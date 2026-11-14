/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         LayerNorm and RmsNorm with the residual against a CPU reference,
               on the aligned (vec4) and the scalar kernel paths, and LayerNorm
               with its scale / shift as inputs.
=============================================================================*/

#include "vai/nodes/layer-norm.h"
#include "vai/nodes/rms-norm.h"
#include "eva-runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace vai;
using namespace vai::node;


template<typename Norm>
struct NormNet : public NeuralNet
{
    Norm norm;

    NormNet(const typename Norm::Config& config)
    : NeuralNet({.oDefaultDtype = Dtype::Fp32})
    , norm(config)
    {
        input("in0") - norm - output("out0");
        if (config.acceptResidual)
            input("residual") - "residual"/norm;
        if (config.coeffsAsInputs)
        {
            input("scale") - "scale"/norm;
            input("shift") - "shift"/norm;
        }
    }
};


// Deterministic values in [-1, 1].
float hashed(uint32_t i, uint32_t salt)
{
    uint32_t h = i * 2654435761u ^ salt * 40503u;
    h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
    return float(h % 20001) / 10000.0f - 1.0f;
}


// Returns whether the node matched the reference.
template<typename Norm>
bool check(const char* name, uint32_t rows, uint32_t features, bool acceptResidual, bool coeffsAsInputs)
{
    constexpr bool layer = std::is_same_v<Norm, LayerNorm>;
    const float eps = 1e-5f;

    std::vector<float> x(size_t(rows) * features), residual(x.size()), scale(features), shift(features);
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = 1.0f + 2.0f * hashed(uint32_t(i), 1);
        residual[i] = hashed(uint32_t(i), 2);
    }
    for (uint32_t j = 0; j < features; ++j)
    {
        scale[j] = 1.0f + 0.5f * hashed(j, 3);
        shift[j] = 0.5f * hashed(j, 4);
    }

    // LayerNorm has a shift; RmsNorm takes one only as an input (adaRMS).
    // Each applies because it is filled or wired below.
    typename Norm::Config config{.eps = eps};
    config.coeffsAsInputs = coeffsAsInputs;
    config.acceptResidual = acceptResidual;
    NormNet<Norm> net(config);

    TensorDict inputs{{"in0", Tensor({rows, features}, Dtype::Fp32).setFloats(x)}};
    if (coeffsAsInputs)
    {
        inputs.insert({"scale", Tensor({1, features}, Dtype::Fp32).setFloats(scale)});
        inputs.insert({"shift", Tensor({1, features}, Dtype::Fp32).setFloats(shift)});
    }
    else
    {
        net.norm["scale"] = Tensor({features}, Dtype::Fp32).setFloats(scale);
        if (layer)
            net.norm["shift"] = Tensor({features}, Dtype::Fp32).setFloats(shift);
    }
    if (acceptResidual)
        inputs.insert({"residual", Tensor({rows, features}, Dtype::Fp32).setFloats(residual)});
    Tensor out;
    net(inputs, {{"out0", &out}});
    const float* got = out.getData<float>();

    double maxErr = 0.0;
    for (uint32_t r = 0; r < rows; ++r)
    {
        const float* row = &x[size_t(r) * features];
        double mean = 0.0, meanSq = 0.0;
        for (uint32_t j = 0; j < features; ++j)
        {
            mean += row[j];
            meanSq += double(row[j]) * row[j];
        }
        mean /= features;
        meanSq /= features;
        const double invStd = layer
            ? 1.0 / std::sqrt(meanSq - mean * mean + eps)
            : 1.0 / std::sqrt(meanSq + eps);
        for (uint32_t j = 0; j < features; ++j)
        {
            const size_t i = size_t(r) * features + j;
            double expected = layer
                ? (row[j] - mean) * invStd * scale[j] + shift[j]
                : row[j] * invStd * scale[j] + (coeffsAsInputs ? shift[j] : 0.0);
            if (acceptResidual)
                expected += residual[i];
            maxErr = std::max(maxErr, std::fabs(got[i] - expected));
        }
    }

    const bool pass = maxErr <= 1e-4;
    printf("%-9s [%3u, %4u]  residual %d  coeffs as inputs %d  max err %.2e  %s\n",
        name, rows, features, acceptResidual, coeffsAsInputs, maxErr, pass ? "ok" : "FAIL");
    return pass;
}


int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    getVaiDevice();

    uint32_t failures = 0;
    for (const uint32_t features : {2560u, 1150u})      // aligned (F % 4 == 0), scalar
        for (const uint32_t rows : {1u, 37u})
            for (const bool acceptResidual : {false, true})
                for (const bool coeffsAsInputs : {false, true})
                {
                    failures += !check<LayerNorm>("LayerNorm", rows, features, acceptResidual, coeffsAsInputs);
                    failures += !check<RmsNorm>("RmsNorm", rows, features, acceptResidual, coeffsAsInputs);
                }

    printf("\n%u failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
