// ACT's action chunk against reference.py, on an input both sides can compute.
//
// The image and the state are closed-form, so reference.py rebuilds them bit for
// bit without sharing a file. This side prints the [actionChunk, actionDim] chunk,
// one row per line; reference.py runs LeRobot's forward in numpy from the raw
// checkpoint and reports the gap.
//
//   test-act [fp32|fp16] > out.txt          default: fp32 weights
//   python tests/act/reference.py out.txt

#include "models/production/act/net.h"
#include "models/production/act/lerobot-weights.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace vai;
namespace act = model::production::act;


int main(int argc, char** argv)
{
    const bool fp16 = argc > 1 && std::strcmp(argv[1], "fp16") == 0;

    auto modelCfg = act::ModelConfig::get({
        .variant = "lerobot/transfer-cube",
        .cameras = {"top"},
        .imageSize = {480, 640},
        .stateDim = 14,
        .actionDim = 14,
        .actionChunk = 100,
    });
    act::Net net({
        .wDefaultDtype = fp16 ? Dtype::Fp16 : Dtype::Fp32,
    }, modelCfg);
    loadWeights<"lerobot">(net);

    // The same closed forms as reference.py, the image normalized with the checkpoint's statistics.
    const auto [H, W] = modelCfg.imageSize;
    std::vector<float> chw(size_t(3) * H * W);
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t i = 0; i < H; ++i)
            for (uint32_t j = 0; j < W; ++j)
            {
                const float v = 0.5f + 0.5f * std::sin(0.05f * i + 0.07f * j + float(c));
                chw[(size_t(c) * H + i) * W + j] = (v - net.imageMean.at("top")[c]) / net.imageStd.at("top")[c];
            }

    std::vector<float> states(modelCfg.stateDim);
    for (uint32_t k = 0; k < modelCfg.stateDim; ++k)
        states[k] = 0.1f * std::sin(float(k));
    net.normalizeState(states);

    Tensor images = Tensor({1, 3, H, W}).setFloats(std::move(chw));
    Tensor state = Tensor({1, modelCfg.stateDim}).setFloats(std::move(states));
    Tensor actions;
    net.predict(images, state, &actions);

    const float* data = actions.getData<float>();
    std::vector<float> a(data, data + size_t(modelCfg.actionChunk) * modelCfg.actionDim);
    net.denormalizeActions(a);
    for (uint32_t t = 0; t < modelCfg.actionChunk; ++t)
    {
        for (uint32_t d = 0; d < modelCfg.actionDim; ++d)
            printf("%.7g ", a[size_t(t) * modelCfg.actionDim + d]);
        printf("\n");
    }
    return 0;
}
