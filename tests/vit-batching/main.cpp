// What batching the cameras buys on pi0's vision tower.
//
// The tower is SigLIP So400m/14 and pi0 feeds it three views per observation.
// One pass over [3, C, S, S] attends each image apart, which is what the batch
// axis is for; the alternative is three passes over [1, C, S, S]. Both do the
// same arithmetic, so the gap is what the GPU loses to short dispatches and to
// the per-forward walk of the graph.
//
// Weights are random: only the shapes reach the kernels, and both runs use the
// same ones anyway.
//
// The two run in alternating blocks, so a drift in clocks or scheduling lands
// on both rather than on whichever went first.
//
//   test-vit-batching [itersPerBlock]      default: 10

#include "models/production/pi0/net.h"
#include "eva-runtime.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace vai;
namespace pi0 = model::production::pi0;


/////////////////////////////////////////////////////////////////////////////////////////
// The tower on its own: patches, the encoder, and the projection pi0 reads.
/////////////////////////////////////////////////////////////////////////////////////////
struct VitNet : public NeuralNet
{
    pi0::ModelConfig modelCfg;

    vai::group::PatchEmbedding patchEmbed;
    pi0::VisionTransformer visionEncoder;
    vai::node::Linear projector;

    VitNet(NeuralNetConfig netCfg, const pi0::ModelConfig& cfg)
    : NeuralNet(netCfg)
    , modelCfg(cfg)
    , patchEmbed({
        .imageChannels = cfg.visionTrf.imageChannels,
        .imageSize = cfg.visionTrf.imageSize,
        .patchSize = cfg.visionTrf.patchSize,
        .embedDim = cfg.visionTrf.embedDim,
        .allowBatch = true})
    , visionEncoder({.base = cfg.visionTrf, .allowBatch = true})
    , projector({
        .inDim = cfg.visionTrf.embedDim,
        .outDim = cfg.languageTrf.embedDim})
    {
        input("inImages") - patchEmbed - visionEncoder - projector - output("outTokens");
    }
};


// Every parameter the tower reads, from one generator. The values are never
// compared, so the only thing that has to be right is the shape.
void randomizeWeights(VitNet& net)
{
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    auto fill = [&](const std::vector<uint32_t>& shape, Dtype dt) {
        size_t n = 1;
        for (uint32_t d : shape) n *= d;
        std::vector<float> v(n);
        for (float& x : v) x = dist(rng);
        return Tensor(shape, dt).setFloats(std::move(v));
    };
    // Weights at the net's wDefaultDtype, norm coefficients at Fp32.
    const Dtype wDt = net.config().wDefaultDtype;
    auto fillLin = [&](auto& lin, uint32_t out, uint32_t in) {
        lin["weight"] = fill({out, in}, wDt);
        lin["bias"]   = fill({out}, wDt);
    };
    auto fillNorm = [&](auto& ln, uint32_t D) {
        ln["scale"] = fill({D}, Dtype::Fp32);
        ln["shift"]  = fill({D}, Dtype::Fp32);
    };

    const auto& modelCfg = net.modelCfg;
    const auto& vision = modelCfg.visionTrf;
    const uint32_t D = vision.embedDim;
    const uint32_t K = modelCfg.visionTrf.imageChannels
                     * modelCfg.visionTrf.patchSize * modelCfg.visionTrf.patchSize;

    fillLin(net.patchEmbed.project, D, K);

    // One image's rows, repeated over the output by the projection itself.
    net.patchEmbed.project["bias"] =
        fill({modelCfg.visionTrf.numPatches(), D}, wDt);

    for (auto& block : net.visionEncoder.blocks)
    {
        fillNorm(block.norm1, D);
        fillNorm(block.norm2, D);
        fillLin(block.mhAttn.projQ,   D, D);
        fillLin(block.mhAttn.projK,   D, D);
        fillLin(block.mhAttn.projV,   D, D);
        fillLin(block.mhAttn.projOut, D, D);
        fillLin(block.ffn.fc1, vision.ffnHiddenDim, D);
        fillLin(block.ffn.fc2, D, vision.ffnHiddenDim);
    }
    fillNorm(net.visionEncoder.finalNorm, D);
    fillLin(net.projector, modelCfg.languageTrf.embedDim, D);
}


int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 10;

    eva::Device device = getVaiDevice();
    auto modelCfg = pi0::ModelConfig::_3B();
    const uint32_t cameras = modelCfg.maxCameras;
    const uint32_t S = modelCfg.visionTrf.imageSize, C = modelCfg.visionTrf.imageChannels;

    NeuralNetConfig netCfg = {
        .device = device, .wDefaultDtype = Dtype::Fp16};

    printf("\n=== pi0 vision tower: %u cameras batched vs one at a time ===\n", cameras);
    printf("SigLIP %ux%u, patch %u, %u blocks, D=%u, %u tokens per image\n\n",
           S, S, modelCfg.visionTrf.patchSize, modelCfg.visionTrf.nBlocks,
           modelCfg.visionTrf.embedDim, modelCfg.visionTrf.numPatches());

    VitNet batched(netCfg, modelCfg);
    VitNet single(netCfg, modelCfg);
    randomizeWeights(batched);
    randomizeWeights(single);

    std::vector<float> pixels(size_t(cameras) * C * S * S);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : pixels) x = dist(rng);

    Tensor allImages = Tensor({cameras, C, S, S}).setFloats(pixels);
    std::vector<Tensor> oneImage(cameras);
    for (uint32_t c = 0; c < cameras; ++c)
    {
        const size_t off = size_t(c) * C * S * S;
        oneImage[c] = Tensor({1, C, S, S}).setFloats(
            std::vector<float>(pixels.begin() + off, pixels.begin() + off + size_t(C) * S * S));
    }

    Tensor out;
    auto runBatched = [&] {
        batched({{"inImages", allImages}}, {{"outTokens", &out}});
        out.getData();
    };
    auto runSingle = [&] {
        for (uint32_t c = 0; c < cameras; ++c)
        {
            single({{"inImages", oneImage[c]}}, {{"outTokens", &out}});
            out.getData();
        }
    };

    for (int i = 0; i < 3; ++i) { runBatched(); runSingle(); }   // pipelines, pools

    // Same weights, same pixels: the two paths must agree to whatever the wider
    // tile's accumulation order costs.
    const uint32_t N = modelCfg.visionTrf.numPatches(), Dllm = modelCfg.languageTrf.embedDim;
    std::vector<float> fromBatch;
    {
        batched({{"inImages", allImages}}, {{"outTokens", &out}});
        const float* p = out.getData<float>();
        fromBatch.assign(p, p + size_t(cameras) * N * Dllm);
    }
    double maxRel = 0.0, maxAbs = 0.0;
    for (uint32_t c = 0; c < cameras; ++c)
    {
        single({{"inImages", oneImage[c]}}, {{"outTokens", &out}});
        const float* got = out.getData<float>();
        const float* ref = fromBatch.data() + size_t(c) * N * Dllm;
        for (size_t i = 0; i < size_t(N) * Dllm; ++i)
        {
            double d = std::fabs(double(got[i]) - double(ref[i]));
            maxAbs = std::max(maxAbs, d);
            maxRel = std::max(maxRel, d / std::max(1e-3, std::fabs(double(ref[i]))));
        }
    }
    printf("  output agreement over %u x %u x %u: maxAbs %.3e, maxRel %.3e\n\n",
           cameras, N, Dllm, maxAbs, maxRel);

    auto timeIt = [&](auto&& fn) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };

    printf("  %-5s %-26s %10s\n", "block", "", "ms");
    double oneAtATime[2], asBatch[2];
    for (int b = 0; b < 2; ++b)
    {
        oneAtATime[b] = timeIt(runSingle);
        printf("  %-5d %-26s %10.3f\n", 2 * b + 1, "one at a time x3", oneAtATime[b]);
        asBatch[b] = timeIt(runBatched);
        printf("  %-5d %-26s %10.3f\n", 2 * b + 2, "batched  [3, C, S, S] x1", asBatch[b]);
    }

    const double msSingle  = 0.5 * (oneAtATime[0] + oneAtATime[1]);
    const double msBatched = 0.5 * (asBatch[0] + asBatch[1]);

    printf("\n  %-32s %10.3f\n", "one at a time x3  (mean)", msSingle);
    printf("  %-32s %10.3f  %.2fx\n", "batched  [3, C, S, S] x1  (mean)",
           msBatched, msSingle / msBatched);
    printf("\n  batching saves %.3f ms per observation (%.1f%%), %d iterations a block\n\n",
           msSingle - msBatched, 100.0 * (msSingle - msBatched) / msSingle, iters);
    return 0;
}
