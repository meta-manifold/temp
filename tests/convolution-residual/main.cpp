/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         Convolution's residual against a CPU reference: every kernel /
               stride the plan selector routes differently, both residual orders,
               fp32 and fp16 weights.
=============================================================================*/

#include "vai/nodes/convolution.h"
#include "eva-runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace vai;
using namespace vai::node;


struct ConvNet : public NeuralNet
{
    Convolution conv;

    ConvNet(const NeuralNetConfig& netCfg, const Convolution::Config& config)
    : NeuralNet(netCfg)
    , conv(config)
    {
        input("in0") - conv - output("out0");
        if (config.acceptResidual)
            input("residual") - "residual"/conv;
    }
};


struct Shape
{
    uint32_t ci, co, h, w, kernel, stride;
    const char* path;   // the path the selector prefers for it
};


// Deterministic values in [-1, 1].
float hashed(uint32_t i, uint32_t salt)
{
    uint32_t h = i * 2654435761u ^ salt * 40503u;
    h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
    return float(h % 20001) / 10000.0f - 1.0f;
}


std::vector<float> filled(size_t count, uint32_t salt, float scale)
{
    std::vector<float> v(count);
    for (size_t i = 0; i < count; ++i)
        v[i] = scale * hashed(uint32_t(i), salt);
    return v;
}


// out[co, ho, wo] = epilogue(scale * conv + bias (+ residual)) (+ residual)
std::vector<float> reference(
    const Shape& s, const std::vector<float>& x, const std::vector<float>& weight,
    const std::vector<float>& bias, const std::vector<float>& scale,
    const std::vector<float>* residual, ResidualOrder order)
{
    const uint32_t ho = (s.h - 1u) / s.stride + 1u;
    const uint32_t wo = (s.w - 1u) / s.stride + 1u;
    const int pad = int(s.kernel - 1u) / 2;
    std::vector<float> out(size_t(s.co) * ho * wo);
    for (uint32_t o = 0; o < s.co; ++o)
        for (uint32_t y = 0; y < ho; ++y)
            for (uint32_t x0 = 0; x0 < wo; ++x0)
            {
                double acc = 0.0;
                for (uint32_t c = 0; c < s.ci; ++c)
                    for (uint32_t ky = 0; ky < s.kernel; ++ky)
                        for (uint32_t kx = 0; kx < s.kernel; ++kx)
                        {
                            const int iy = int(y * s.stride + ky) - pad;
                            const int ix = int(x0 * s.stride + kx) - pad;
                            if (iy < 0 || ix < 0 || iy >= int(s.h) || ix >= int(s.w))
                                continue;
                            acc += double(x[(size_t(c) * s.h + iy) * s.w + ix])
                                * weight[((size_t(o) * s.ci + c) * s.kernel + ky) * s.kernel + kx];
                        }
                const size_t index = (size_t(o) * ho + y) * wo + x0;
                double value = acc * scale[o] + bias[o];
                const double r = residual ? (*residual)[index] : 0.0;
                if (order == ResidualOrder::BeforeEpilogue)
                    value += r;
                value = std::max(value, 0.0);
                if (order == ResidualOrder::AfterEpilogue)
                    value += r;
                out[index] = float(value);
            }
    return out;
}


// Returns whether the node matched the reference.
bool check(const Shape& s, Dtype wDtype, bool withResidual, ResidualOrder order)
{
    const uint32_t ho = (s.h - 1u) / s.stride + 1u;
    const uint32_t wo = (s.w - 1u) / s.stride + 1u;
    const float fanIn = float(s.ci * s.kernel * s.kernel);

    const std::vector<float> x = filled(size_t(s.ci) * s.h * s.w, 1, 1.0f);
    const std::vector<float> weight = filled(size_t(s.co) * s.ci * s.kernel * s.kernel, 2, 1.7f / std::sqrt(fanIn));
    const std::vector<float> bias = filled(s.co, 3, 0.5f);
    std::vector<float> scale = filled(s.co, 4, 0.5f);
    for (float& v : scale)
        v += 1.0f;
    const std::vector<float> residual = filled(size_t(s.co) * ho * wo, 5, 1.0f);

    ConvNet net({.wDefaultDtype = wDtype, .oDefaultDtype = Dtype::Fp32}, {
        .inChannels = s.ci,
        .outChannels = s.co,
        .kernelSize = s.kernel,
        .stride = s.stride,
        .acceptResidual = withResidual,
        .residualOrder = order,
        .epilogue = Epilogue::Relu,
    });
    net.conv["weight"] = Tensor({s.co, s.ci, s.kernel, s.kernel}, wDtype).setFloats(weight);
    net.conv["bias"] = Tensor({s.co}, wDtype).setFloats(bias);
    net.conv["scale"] = Tensor({s.co}, wDtype).setFloats(scale);

    const Tensor input = Tensor({s.ci, s.h, s.w}, Dtype::Fp32).setFloats(x);
    const Tensor residualInput = Tensor({s.co, ho, wo}, Dtype::Fp32).setFloats(residual);
    TensorDict inputs{{"in0", input}};
    if (withResidual)
        inputs.insert({"residual", residualInput});
    Tensor out;
    net(inputs, {{"out0", &out}});
    const float* got = out.getData<float>();

    const std::vector<float> expected =
        reference(s, x, weight, bias, scale, withResidual ? &residual : nullptr, order);
    double maxErr = 0.0, maxRef = 0.0;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        maxErr = std::max(maxErr, double(std::fabs(got[i] - expected[i])));
        maxRef = std::max(maxRef, double(std::fabs(expected[i])));
    }
    const double relErr = maxErr / std::max(maxRef, 1e-6);
    const double tolerance = wDtype == Dtype::Fp32 ? 1e-3 : 3e-2;
    const bool pass = relErr <= tolerance;

    const char* residualName = !withResidual ? "none  "
        : order == ResidualOrder::AfterEpilogue ? "after " : "before";
    printf("%-8s %4u %4u %3ux%-3u k%u s%u  %s  %s  rel %.2e  %s\n",
        s.path, s.ci, s.co, s.h, s.w, s.kernel, s.stride,
        wDtype == Dtype::Fp32 ? "fp32" : "fp16", residualName, relErr, pass ? "ok" : "FAIL");
    return pass;
}


int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    getVaiDevice();

    const Shape shapes[] = {
        {64, 64, 20, 30, 1, 1, "K1S1"},
        {32, 48, 17, 13, 1, 1, "K1S1"},
        {512, 256, 15, 20, 1, 1, "K1S1"},
        {2048, 512, 7, 7, 1, 1, "K1S1"},
        {64, 64, 56, 56, 3, 1, "K3S1"},
        {256, 256, 14, 14, 3, 1, "K3S1"},
        {512, 512, 15, 20, 3, 1, "K3S1"},
        {32, 48, 9, 11, 3, 1, "K3S1"},
        {128, 128, 3, 3, 3, 1, "K3S1"},
        {64, 128, 56, 56, 3, 2, "K3S2"},
        {256, 512, 30, 40, 3, 2, "K3S2"},
        {3, 32, 64, 64, 3, 2, "K3S2"},
        {512, 512, 14, 14, 3, 2, "K3S2"},
        {3, 64, 64, 80, 7, 2, "General"},
        {64, 64, 16, 16, 1, 2, "General"},
        {16, 24, 20, 20, 5, 1, "General"},
    };

    uint32_t failures = 0;
    for (const Dtype wDtype : {Dtype::Fp32, Dtype::Fp16})
        for (const Shape& s : shapes)
        {
            failures += !check(s, wDtype, false, ResidualOrder::AfterEpilogue);
            failures += !check(s, wDtype, true, ResidualOrder::AfterEpilogue);
            failures += !check(s, wDtype, true, ResidualOrder::BeforeEpilogue);
        }

    printf("\n%u failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
