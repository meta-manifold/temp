// M = 1 .. 8 Linear kernels on Intel, against linear-slab as Linear runs it
// (its spv, ggmlSlabSgSize / ggmlSlabWgSize): O[m, n] = sum_k I[m, k] W[n, k],
// W fp16, I and O fp32. linear-slab reads W as given; the others read W packed
// as Linear::packWeightIntel lays it out for the Intel path:
//
//   tile   linear-tile-intel as Linear runs it (its spv, intelTileParams,
//          intelTileSplitCount, fp16 accumulation), any M
//   vecm   vecm.comp, vec for M rows, fp32 accumulation, any M up to 8
//   vec    vec.comp, the M = 1 kernel the Intel path once ran          (M = 1)
//   vecacc vecacc.comp, vec with one accumulator per unrolled k pair   (M = 1)
//   vec32  vec32.comp, 32 B per thread per k pair                      (M = 1)
//   slabp  slabp.comp, linear-slab's decomposition over the packed W   (M = 1)
//
// Each kernel is one argument, name[:WG[:S]] ("vecy" = vec with its split index
// on y, walked first, so neighbouring workgroups read neighbouring K spans of a
// block; vec / vecm WG 0 = the rule Linear used, vecRuleWg; S = split-K
// count, folded by reduce.comp and timed with it). Every (shape, M) times its
// kernels in P interleaved passes (forward, reverse, shuffled), R back-to-back
// runs per visit for about 10 ms, and checks each against a double-precision
// reference.
//
//   test-linear-vec-intel [-shape K,N ...] [-m M,M,...] [-passes P] [-packedfirst] [kernel ...]

#include "vai/nodes/linear.h"
#include "eva-runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef VEC_SHADER_DIR
#error "VEC_SHADER_DIR must point at the compiled shaders of this folder"
#endif
#ifndef SLAB_SPV
#error "SLAB_SPV must point at linear-slab.comp.spv"
#endif
#ifndef TILE_SPV
#error "TILE_SPV must point at linear-tile-intel.comp.spv"
#endif

using namespace vai;
using namespace vai::node;

namespace {

eva::Buffer deviceBuffer(eva::Device dev, uint64_t bytes)
{
    return dev.createBuffer({
        .size = bytes,
        .usage = eva::BUFFER_USAGE::STORAGE_BUFFER | eva::BUFFER_USAGE::TRANSFER_SRC | eva::BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = eva::MEMORY_PROPERTY::DEVICE_LOCAL,
    });
}

eva::Buffer hostBuffer(eva::Device dev, uint64_t bytes)
{
    return dev.createBuffer({
        .size = bytes,
        .usage = eva::BUFFER_USAGE::TRANSFER_SRC | eva::BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT,
    });
}

uint16_t toHalf(float f)
{
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = int32_t((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return uint16_t(sign);
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    uint32_t h = sign | (uint32_t(e) << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1u))) h += 1;
    return uint16_t(h);
}

float fromHalf(uint16_t h)
{
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu;
    float f = (e == 0) ? std::ldexp(float(m), -24) : std::ldexp(float(m | 0x400u), int(e) - 25);
    return (h & 0x8000u) ? -f : f;
}

// The workgroup size Linear gave vec.comp: 128 threads (8 subgroups) doubled up
// to 1024 until the 32-column blocks' subgroups fill a wave.
uint32_t vecRuleWg(uint32_t N, eva::Device dev)
{
    const uint32_t blocks = CEIL_DIV(N, 32u), wave = intelTileWave(dev);
    uint32_t sgs = 8;
    while (sgs < 64 && blocks * sgs < wave) sgs *= 2;
    return sgs * 16u;
}

struct Kernel
{
    std::string name;
    uint32_t wg = 0, splits = 1;   // vec*: split-K count, partials folded by reduce.comp
    bool onY = false;
    std::string label() const
    {
        std::string l = wg ? (onY ? "vecy" : name) + ":" + std::to_string(wg) : name;
        return splits > 1 ? l + ":" + std::to_string(splits) : l;
    }
};

}   // namespace


int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    eva::Device dev = getVaiDevice();

    std::vector<std::pair<uint32_t, uint32_t>> shapes = {   // (K, N)
        {2048, 2048}, {2048, 512}, {2048, 8192}, {8192, 2048}, {2048, 128256},
        {4096, 4096}, {4096, 1024}, {4096, 14336}, {14336, 4096}, {1024, 2048}, {1024, 8192},
    };
    std::vector<std::pair<uint32_t, uint32_t>> custom;
    std::vector<uint32_t> Ms = { 1 };
    std::vector<Kernel> kernels;
    uint32_t passes = 6;
    bool packedFirst = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "-passes" && i + 1 < argc) passes = uint32_t(std::atoi(argv[++i]));
        else if (a == "-packedfirst") packedFirst = true;
        else if (a == "-m" && i + 1 < argc)
        {
            Ms.clear();
            std::stringstream ss(argv[++i]);
            for (std::string t; std::getline(ss, t, ',');) Ms.push_back(uint32_t(std::atoi(t.c_str())));
        }
        else if (a == "-shape" && i + 1 < argc)
        {
            uint32_t k = 0, n = 0;
            if (std::sscanf(argv[++i], "%u,%u", &k, &n) != 2 || !k || !n || k % 2) { fprintf(stderr, "bad shape (K even): %s\n", argv[i]); return 1; }
            custom.push_back({k, n});
        }
        else
        {
            Kernel kn;
            const size_t colon = a.find(':');
            kn.name = a.substr(0, colon);
            if (kn.name == "vecy") { kn.name = "vec"; kn.onY = true; }
            if (colon != std::string::npos) std::sscanf(a.c_str() + colon + 1, "%u:%u", &kn.wg, &kn.splits);
            kernels.push_back(kn);
        }
    }
    if (!custom.empty()) shapes = custom;
    if (kernels.empty())
        kernels = { {"slab"}, {"vec", 0}, {"vec32", 128}, {"vec32", 256}, {"slabp", 16}, {"slabp", 64} };

    std::map<std::string, eva::SpvBlob> spvs;
    for (const char* n : { "vec", "vecm", "vecacc", "vec32", "slabp", "reduce" })
        spvs.emplace(n, eva::SpvBlob::readFrom((std::string(VEC_SHADER_DIR) + "/" + n + ".comp.spv").c_str()));
    const eva::SpvBlob slabSpv = eva::SpvBlob::readFrom(SLAB_SPV);
    const eva::SpvBlob tileSpv = eva::SpvBlob::readFrom(TILE_SPV);

    eva::CommandBuffer cmd = dev.newCommandBuffer(eva::queue_compute, eva::COMMAND_POOL_CREATE::RESET_COMMAND_BUFFER);
    eva::QueryPool qp = dev.createTimestampQueryPool(2);
    const auto stage = eva::PIPELINE_STAGE::ALL_COMMANDS;
    eva::Buffer dummy = deviceBuffer(dev, 256);
    eva::ComputePipeline reducePipe = dev.createComputePipeline({ .csStage = spvs.at("reduce") });
    const uint32_t maxM = *std::max_element(Ms.begin(), Ms.end());

    std::map<std::string, std::vector<double>> ratios;   // "kernel M" -> time / slab
    for (auto [K, N] : shapes)
    {
        // W rounded to fp16, as uploaded; the packed copy as Linear::packWeightIntel lays it out.
        std::mt19937 rng(K * 7u + N);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        std::vector<uint16_t> wh(size_t(N) * K);
        std::vector<float> x(size_t(maxM) * K);
        for (auto& v : wh) v = toHalf(uni(rng) * 0.05f);
        for (float& v : x) v = uni(rng);
        const uint32_t kPairs = K / 2, blocks = CEIL_DIV(N, 32u);
        std::vector<uint16_t> packed(size_t(blocks) * kPairs * 64, 0);
        for (uint32_t n = 0; n < N; n++)
            for (uint32_t k = 0; k < K; k++)
                packed[((size_t(n / 32) * kPairs + k / 2) * 16 + n % 16) * 4 + (k % 2) * 2 + (n % 32) / 16] = wh[size_t(n) * K + k];
        std::vector<double> ref(size_t(maxM) * N);
        {
            std::vector<float> wrow(K);
            for (uint32_t n = 0; n < N; n++)
            {
                for (uint32_t k = 0; k < K; k++) wrow[k] = fromHalf(wh[size_t(n) * K + k]);
                for (uint32_t m = 0; m < maxM; m++)
                {
                    double s = 0;
                    const float* xr = &x[size_t(m) * K];
                    for (uint32_t k = 0; k < K; k++) s += double(xr[k]) * wrow[k];
                    ref[size_t(m) * N + n] = s;
                }
            }
        }

        const uint64_t wBytes = wh.size() * 2, pBytes = packed.size() * 2, iBytes = x.size() * 4, oBytes = uint64_t(maxM) * N * 4;
        // -packedfirst allocates the packed W first, to tell the kernels from where their buffers land.
        eva::Buffer bufW, bufP;
        if (packedFirst) { bufP = deviceBuffer(dev, pBytes); bufW = deviceBuffer(dev, wBytes); }
        else             { bufW = deviceBuffer(dev, wBytes); bufP = deviceBuffer(dev, pBytes); }
        eva::Buffer bufI = deviceBuffer(dev, iBytes), bufO = deviceBuffer(dev, oBytes);
        {
            eva::Buffer sw = hostBuffer(dev, wBytes), sp = hostBuffer(dev, pBytes), si = hostBuffer(dev, iBytes);
            std::memcpy(sw.map(), wh.data(), wBytes);
            std::memcpy(sp.map(), packed.data(), pBytes);
            std::memcpy(si.map(), x.data(), iBytes);
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
                .copyBuffer(sw(0, wBytes), bufW(0, wBytes)).copyBuffer(sp(0, pBytes), bufP(0, pBytes)).copyBuffer(si(0, iBytes), bufI(0, iBytes))
            .end().submit().wait();
        }
        eva::Buffer readback = hostBuffer(dev, oBytes);
        eva::Buffer bufPart = deviceBuffer(dev, oBytes * 16);   // split-K partials, up to 16 splits

        for (uint32_t M : Ms)
        {
            printf("\nK %u  N %u  M %u  (W %.1f MB)\n", K, N, M, wBytes / 1e6);
            struct Run { std::string label; std::function<void()> record; std::vector<double> ms; double maxRel; };
            std::vector<Run> runs;
            for (const Kernel& kn : kernels)
            {
                if (M > 1 && kn.name != "slab" && kn.name != "vecm" && kn.name != "tile") continue;
                eva::ComputePipeline pipe;
                eva::DescriptorSet ds, rds;
                uint32_t gx = 1, gy = 1, gz = 1, splits = 1;
                std::vector<uint32_t> pc;
                std::string label = kn.label();
                if (kn.name == "slab")
                {
                    const uint32_t sg = ggmlSlabSgSize(dev), wg = ggmlSlabWgSize(K, N, sg, dev);
                    label = "slab(wg" + std::to_string(wg) + ")";
                    pipe = dev.createComputePipeline({
                        .csStage = slabSpv + eva::constant_id<0>(wg) + eva::constant_id<1>(sg) + eva::constant_id<2>(M) + eva::constant_id<3>(2u)
                            + eva::constant_id<4>(0u) + eva::constant_id<5>(0u) + eva::constant_id<6>(0u) + eva::constant_id<7>(0u) + eva::constant_id<8>(0u)
                            + eva::constant_id<SPEC_ID_TYPE_O>(Dtype::Fp32) + eva::constant_id<SPEC_ID_TYPE_I>(Dtype::Fp32)
                            + eva::constant_id<SPEC_ID_TYPE_W>(Dtype::Fp16) + eva::constant_id<SPEC_ID_ALIGNED>(K % 4 == 0 ? 1u : 0u)
                            + eva::constant_id<SPEC_ID_EPILOGUE>(0u),
                        .requiredSubgroupSize = sg,
                    });
                    ds = gVaiDestSetPool(pipe.descSetLayout(0));
                    ds.write({ bufO, bufI, bufW, dummy, dummy });
                    float one = 1.0f; uint32_t oneBits; std::memcpy(&oneBits, &one, 4);
                    pc = { K, N, oneBits };
                    gx = CEIL_DIV(N, 2u);
                    if (gx > dev.maxComputeWorkGroupCount()[0]) { gz = 64; gx = CEIL_DIV(gx, gz); }
                }
                else if (kn.name == "tile")
                {
                    const Linear::PipeParamsIntel t = intelTileParams(M, N, K, Dtype::Fp32, false, dev);
                    splits = intelTileSplitCount(M, N, K, t, dev);
                    const uint32_t splitSize = (splits > 1) ? CEIL_DIV(CEIL_DIV(K, splits), 16u) * 16u : K;
                    splits = CEIL_DIV(K, splitSize);
                    label = "tile(b" + std::to_string(t.tileB) + " sp" + std::to_string(splits) + ")";
                    pipe = dev.createComputePipeline({
                        .csStage = tileSpv + eva::constant_id<7>(t.tileB) + eva::constant_id<10>(t.bandTiles) + eva::constant_id<11>(Dtype::Fp16)
                            + eva::constant_id<SPEC_ID_TYPE_O>(Dtype::Fp32) + eva::constant_id<SPEC_ID_TYPE_I>(Dtype::Fp32)
                            + eva::constant_id<SPEC_ID_TYPE_W>(Dtype::Fp16),
                        .requiredSubgroupSize = 16,
                    });
                    ds = gVaiDestSetPool(pipe.descSetLayout(0));
                    ds.write({ splits > 1 ? bufPart : bufO, bufI, bufP });
                    pc = { M, N, K, splitSize };
                    gx = blocks * CEIL_DIV(CEIL_DIV(M, t.tileB), t.bandTiles) * t.bandTiles;
                    gz = splits;
                }
                else
                {
                    uint32_t wg = kn.wg;
                    if ((kn.name == "vec" || kn.name == "vecm" || kn.name == "vecacc") && wg == 0)
                    { wg = vecRuleWg(N, dev); label = kn.name + "(rule " + std::to_string(wg) + ")"; }
                    if (wg == 0) wg = 128;
                    pipe = dev.createComputePipeline({
                        .csStage = spvs.at(kn.name) + eva::constant_id<0>(wg) + eva::constant_id<1>(kn.onY ? 1u : 0u) + eva::constant_id<2>(M)
                            + eva::constant_id<SPEC_ID_TYPE_O>(Dtype::Fp32) + eva::constant_id<SPEC_ID_TYPE_I>(Dtype::Fp32)
                            + eva::constant_id<SPEC_ID_TYPE_W>(Dtype::Fp16),
                        .requiredSubgroupSize = 16,
                    });
                    ds = gVaiDestSetPool(pipe.descSetLayout(0));
                    splits = (kn.name == "slabp") ? 1u : std::max(kn.splits, 1u);
                    const uint32_t splitSize = (splits > 1) ? CEIL_DIV(CEIL_DIV(K, splits), 64u) * 64u : K;
                    splits = CEIL_DIV(K, splitSize);
                    ds.write({ splits > 1 ? bufPart : bufO, bufI, bufP });
                    pc = { M, N, K, splitSize };
                    gx = (kn.name == "slabp") ? blocks * 16u : blocks;
                    gz = splits;
                    if (kn.onY) { gy = splits; gz = 1; }
                }
                if (splits > 1)
                {
                    rds = gVaiDestSetPool(reducePipe.descSetLayout(0));
                    rds.write({ bufO, bufPart });
                }
                const bool viaReduce = splits > 1;
                const uint32_t MN = M * N;
                auto record = [&cmd, &bufO, &bufPart, &reducePipe, pipe, ds, rds, pc, gx, gy, gz, viaReduce, MN]
                {
                    cmd.bindPipeline(pipe).bindDescSets({ds})
                        .setPushConstants(0, uint32_t(pc.size() * 4), pc.data())
                        .dispatch(gx, gy, gz);
                    if (!viaReduce)
                    {
                        cmd.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / bufO / eva::SYNC_SCOPE::COMPUTE_WRITE);
                        return;
                    }
                    const uint32_t rpc[] = { MN, gy * gz };
                    cmd.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / bufPart / eva::SYNC_SCOPE::COMPUTE_READ)
                        .bindPipeline(reducePipe).bindDescSets({rds})
                        .setPushConstants(0, sizeof(rpc), rpc)
                        .dispatch(CEIL_DIV(MN, 256u), 1, 1)
                        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / bufO / eva::SYNC_SCOPE::COMPUTE_WRITE);
                };
                cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT);
                record();
                cmd.copyBuffer(bufO(0, uint64_t(MN) * 4), readback(0, uint64_t(MN) * 4));
                cmd.end().submit().wait();
                const float* got = (const float*)readback.map();
                double maxErr = 0, maxRef = 0;
                for (uint32_t i = 0; i < MN; i++) { maxErr = std::max(maxErr, std::abs(ref[i] - got[i])); maxRef = std::max(maxRef, std::abs(ref[i])); }
                runs.push_back({ label, record, {}, maxErr / maxRef });
            }

            // R runs per visit for about 10 ms at the first kernel's speed.
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
            cmd.writeTimestamp(stage, qp, 0); runs[0].record(); cmd.writeTimestamp(stage, qp, 1);
            cmd.end().submit().wait();
            const uint32_t R = std::clamp(uint32_t(10.0 / std::max(qp.getElapsedMs(0, 1), 1e-3)), 5u, 2000u);
            std::vector<size_t> order(runs.size());
            std::mt19937 shuffleRng(K + N + M);
            for (uint32_t p = 0; p < passes; p++)
            {
                for (size_t i = 0; i < order.size(); i++) order[i] = i;
                if (p % 3 == 1) std::reverse(order.begin(), order.end());
                if (p % 3 == 2) std::shuffle(order.begin(), order.end(), shuffleRng);
                for (size_t i : order)
                {
                    cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
                    cmd.writeTimestamp(stage, qp, 0);
                    for (uint32_t r = 0; r < R; r++) runs[i].record();
                    cmd.writeTimestamp(stage, qp, 1);
                    cmd.end().submit().wait();
                    runs[i].ms.push_back(qp.getElapsedMs(0, 1) / R);
                }
            }
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            const double slabMs = med(runs[0].ms);
            for (const Run& r : runs)
            {
                const double ms = med(r.ms);
                printf("  %-18s %9.4f ms  %6.1f GB/s  x%5.3f  maxRel %.1e\n", r.label.c_str(), ms, wBytes / (ms * 1e6), ms / slabMs, r.maxRel);
                ratios[r.label.substr(0, r.label.find('(')) + " M" + std::to_string(M)].push_back(ms / slabMs);
            }
        }
    }

    printf("\nagainst %s, over the shapes:\n", kernels[0].name.c_str());
    for (auto& [name, v] : ratios)
    {
        std::sort(v.begin(), v.end());
        printf("  %-14s median x%.3f  min x%.3f  max x%.3f\n", name.c_str(), v[v.size() / 2], v.front(), v.back());
    }
    return 0;
}
