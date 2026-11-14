/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         GroupNorm against a CPU reference: ranks 3 and 4, the aligned,
               cached and scalar kernel paths, shift, the residual in either
               order and the epilogue.
=============================================================================*/

#include "vai/nodes/group-norm.h"
#include "eva-runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace vai;
using namespace vai::node;


struct GroupNormNet : public NeuralNet
{
    GroupNorm norm;

    GroupNormNet(const GroupNorm::Config& config)
    : NeuralNet({.oDefaultDtype = Dtype::Fp32})
    , norm(config)
    {
        input("in0") - norm - output("out0");
        if (config.acceptResidual)
            input("residual") - "residual"/norm;
    }
};


// Deterministic values in [-1, 1].
float hashed(uint32_t i, uint32_t salt)
{
    uint32_t h = i * 2654435761u ^ salt * 40503u;
    h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
    return float(h % 20001) / 10000.0f - 1.0f;
}


double activate(double x, Epilogue epilogue)
{
    switch (epilogue)
    {
        case Epilogue::Relu: return std::max(x, 0.0);
        case Epilogue::Silu: return x / (1.0 + std::exp(-x));
        default:             return x;
    }
}


struct Fusion
{
    bool hasShift;
    bool acceptResidual;
    ResidualOrder residualOrder;
    Epilogue epilogue;
    const char* name;
};


// Returns whether the node matched the reference.
bool check(const std::vector<uint32_t>& shape, uint32_t groups, const Fusion& fusion)
{
    const size_t rank = shape.size();
    const uint32_t C = shape[rank - 3];
    const uint32_t HW = shape[rank - 2] * shape[rank - 1];
    uint32_t samples = 1;
    for (size_t axis = 0; axis + 3 < rank; ++axis)
        samples *= shape[axis];
    const uint32_t chPerGroup = C / groups;
    const float eps = 1e-5f;

    std::vector<float> x(size_t(samples) * C * HW), residual(x.size()), scale(C), shift(C, 0.0f);
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = 3.0f + 2.0f * hashed(uint32_t(i), 1);   // an offset mean, as activations have
        residual[i] = hashed(uint32_t(i), 4);
    }
    for (uint32_t c = 0; c < C; ++c)
    {
        scale[c] = 1.0f + 0.5f * hashed(c, 2);
        if (fusion.hasShift)
            shift[c] = 0.5f * hashed(c, 3);
    }

    GroupNormNet net({
        .groups = groups,
        .acceptResidual = fusion.acceptResidual,
        .residualOrder = fusion.residualOrder,
        .epilogue = fusion.epilogue});
    net.norm["scale"] = Tensor({C}, Dtype::Fp32).setFloats(scale);
    if (fusion.hasShift)
        net.norm["shift"] = Tensor({C}, Dtype::Fp32).setFloats(shift);

    TensorDict inputs{{"in0", Tensor(shape, Dtype::Fp32).setFloats(x)}};
    if (fusion.acceptResidual)
        inputs.insert({"residual", Tensor(shape, Dtype::Fp32).setFloats(residual)});
    Tensor out;
    net(inputs, {{"out0", &out}});
    const float* got = out.getData<float>();

    double maxErr = 0.0;
    for (uint32_t s = 0; s < samples; ++s)
        for (uint32_t g = 0; g < groups; ++g)
        {
            const size_t base = (size_t(s) * C + size_t(g) * chPerGroup) * HW;
            const size_t count = size_t(chPerGroup) * HW;
            double mean = 0.0, var = 0.0;
            for (size_t j = 0; j < count; ++j)
                mean += x[base + j];
            mean /= double(count);
            for (size_t j = 0; j < count; ++j)
                var += (x[base + j] - mean) * (x[base + j] - mean);
            var /= double(count);
            const double invStd = 1.0 / std::sqrt(var + eps);
            for (size_t j = 0; j < count; ++j)
            {
                const uint32_t c = g * chPerGroup + uint32_t(j / HW);
                double expected = (x[base + j] - mean) * invStd * scale[c] + shift[c];
                const double r = fusion.acceptResidual ? residual[base + j] : 0.0;
                if (fusion.residualOrder == ResidualOrder::BeforeEpilogue)
                    expected += r;
                expected = activate(expected, fusion.epilogue);
                if (fusion.residualOrder == ResidualOrder::AfterEpilogue)
                    expected += r;
                maxErr = std::max(maxErr, std::fabs(got[base + j] - expected));
            }
        }

    const bool pass = maxErr <= 1e-4;
    printf("[");
    for (size_t axis = 0; axis < rank; ++axis)
        printf(axis ? ", %u" : "%u", shape[axis]);
    printf("]  groups %3u  %-22s  max err %.2e  %s\n", groups, fusion.name, maxErr, pass ? "ok" : "FAIL");
    return pass;
}


int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    getVaiDevice();

    struct Case { std::vector<uint32_t> shape; uint32_t groups; };
    const Case cases[] = {
        {{64, 15, 20}, 4},          // aligned, HW % 4 == 0
        {{32, 7, 9}, 2},            // scalar, HW odd
        {{256, 2, 2}, 16},          // cached: a group fits one quad per thread
        {{2, 64, 15, 20}, 4},       // two samples
        {{3, 32, 7, 9}, 2},
        {{4, 256, 2, 2}, 16},
        {{2, 512, 15, 20}, 32},     // ResNet-18 layer4, 16 channels a group
    };

    const Fusion fusions[] = {
        {true, false, ResidualOrder::AfterEpilogue, Epilogue::None, "shift"},
        {false, false, ResidualOrder::AfterEpilogue, Epilogue::None, "no shift"},
        {true, true, ResidualOrder::AfterEpilogue, Epilogue::Relu, "relu, residual after"},
        {true, true, ResidualOrder::BeforeEpilogue, Epilogue::Relu, "residual before, relu"},
        {true, false, ResidualOrder::AfterEpilogue, Epilogue::Silu, "silu"},
    };

    uint32_t failures = 0;
    for (const Case& c : cases)
        for (const Fusion& fusion : fusions)
            failures += !check(c.shape, c.groups, fusion);

    printf("\n%u failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
