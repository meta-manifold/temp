// Tuning the scalar tile GEMM (a copy of vai/shaders/linear-tile.comp and its
// variants, kept in this folder) on pi0.5's two language FFN projections:
//
//   gate+up  O[922, 32768] = I[922, 2048]  · W[32768, 2048]ᵀ
//   down     O[922,  2048] = I[922, 16384] · W[2048, 16384]ᵀ   (residual left out)
//
// I and O are fp32, W fp16, as the demo runs them. Each configuration is
// built straight from its spec constants, timed with GPU timestamps, and
// checked against a CPU reference on a few sampled rows.
//
// A configuration is one argument, comma separated:
//   shader,wgSize,sgSize,tN,tM,sgpNIter,sgpMIter,wgN,wgK,sgN,acc,reqSg[,opt[,stage32]]
//     shader  1 = linear-tile-1.comp (Linear's scalar tile kernel as copied), 11 =
//             linear-tile-11.comp; "1-swizzle" = linear-tile-1-swizzle.comp (as 1,
//             with opt = the swizzle group). Variants 2 ~ 10 are in the history
//             (7aa97dd8) only. 11 takes only wgSize, sgSize (= forced subgroup),
//             tN (columns per lane), tM (rows per subgroup), sgpMIter (subgroups
//             along N), acc, wgK (k per staged block), opt (0: one split, no
//             reduce pass; s > 0: s splits, partials folded by split-reduce.comp,
//             so 1 runs the reduce pass at one split), sgpNIter (dispatch order:
//             0 M tiles on x, 1 N tiles on x), sgN (W padding in vec4 slots) and
//             stage32 (1: fp32 staging, which also accumulates in fp32); the rest
//             are placeholders. 21 = linear-tile-21.comp (OpenVINO's
//             fully_connected_gpu_bf_tiled) takes tN (TILE_OFM), tM (TILE_B),
//             wgK (TILE_K), sgpNIter (DISPATCH_BSV), sgpMIter (DISPATCH_FSV), sgN
//             (split-K count, 0 or 1 for none), acc and stage32 (1: the bsv x fsv
//             groups run row-fastest) and opt (kernel variant bits, linear-tile-21's
//             constant 13); wgSize and sgSize are 16, the rest placeholders
//     acc     0 fp32, 1 fp16 accumulation
//     reqSg   required subgroup size on the pipeline (0: the driver's choice)
//     opt     constant 13, the variant's own switch (0 when left out)
// "11-node" runs linear-tile-11 with the pick of intel-tile-v11.h for the shape
// (tile, order, split; fp16 accumulation; "11-node32" for fp32 accumulation),
// "21-ov" linear-tile-21 with what OpenVINO's GetAutoTuneParams picks for it
// (fp16, static shape, uncompressed weights); "21-ov,M'" takes OpenVINO's pick
// for M' rows instead (its own token count).
// The tile parameters follow Linear::PipeParamsTile's constructor (wgM, sgM,
// sgpM and sgpN are derived). Without arguments the baseline runs: the M tier
// the demo selects on this device.
//
// -flush (with -passes) evicts the GPU L2 with a 32 MB copy ahead of every timed run
// and times each run on its own, R = 3 ~ 16 runs per visit for about 10 ms.
// -passes P (with -shape) times every configuration of a shape in P passes
// after the usual run: each pass visits all of them once (forward, reverse,
// then shuffled), timing R back-to-back runs per visit, R = max(3, 30 ms / the
// configurations' median time), and prints each visit with its system-clock
// start and end (ms since the epoch) to join with a clock log.
// -quick times one run instead of about half a second of runs. -i16 hands I
// over in fp16 instead of fp32 (the reference stays on the fp32 values). -clk
// brackets each timed configuration with a short fp16 FMA run, a proxy for the
// GPU clock at that moment. -shape M,K,N (repeatable) replaces the two FFN
// shapes; -w32 hands W over in fp32 (linear-tile-7 ~ 10 take fp16 W only). -isa prints
// the driver's ISA disassembly per configuration. -peak runs fma-peak.comp
// instead: the device's measured fp32 / fp16 / f16vec2 FMA throughput.
//
//   test-linear-tune-intel [gu|dn] [-shape M,K,N ...] [-passes P] [-stats] [-quick] [-i16] [-w32] [-clk] [-isa] [config ...]
//   test-linear-tune-intel -peak [-isa]
//   test-linear-tune-intel -walk gx,gy,wgSize,iters,slmWords,reqSg,path ...   (walk-probe.comp)
//   test-linear-tune-intel -share gx,wgSize,reqSg,G,regionVec4,reps ...         (share-probe.comp)
//   test-linear-tune-intel -slm gx,wgSize,mode,iters,tN ...                     (slm-probe.comp)
//   test-linear-tune-intel -rows wgSize,rows,chunk4,rowBytes,strideBytes[,MB] ...  (rows-probe.comp)

#include "vai/nodes/common.h"
#include "intel-tile-v11.h"
#include "eva-runtime.h"
#include <algorithm>
#include <map>
#include <chrono>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef TILE_SHADER_DIR
#error "TILE_SHADER_DIR must point at the compiled linear-tile-N.comp.spv files"
#endif

using namespace vai;

namespace {

struct Shape { std::string what; uint32_t M, K, N; };

struct Config
{
    uint32_t shader = 1;
    uint32_t wgSize, sgSize, tN, tM, sgpNIter, sgpMIter, wgN, wgK, sgN;
    uint32_t acc = 1, reqSg = 0;
    uint32_t opt = 0;   // constant 13, a variant's own switch (2: forceStageF32, 5: mFastest, 6: groupN);
                        // linear-tile-10 / 11 take it as the split-K count, 0 for one split without the reduce pass
    uint32_t stage32 = 0;   // linear-tile-11 constant 14: 1 stages I and W in fp32
    bool swizzle1 = false;   // "1-swizzle": linear-tile-1-swizzle.comp, host-side as 1, opt = swizzle group
    bool probe11 = false;    // "11-probe": linear-tile-11-probe.comp, host-side as 11, records the run order
    bool node11 = false;
    bool ov21 = false;

    // derived, as in Linear::PipeParamsTile
    uint32_t wgM = 0, sgM = 0, sgpM = 0, sgpN = 0;
    std::string why;   // empty when valid

    bool derive(uint32_t maxShared)
    {
        auto fail = [&](const char* r) { why = r; return false; };
        if (shader == 21)   // linear-tile-21: one subgroup of 16, TILE_B x TILE_OFM * 16
        {
            wgSize = sgSize = reqSg = 16;
            wgM = tM; wgN = tN * 16;
            if (16 % wgK) return fail("TILE_K");
            return true;
        }
        if (shader >= 7)   // linear-tile-7 ~ 11: a grid of subgroups (sgpMIter along N), each tM rows x sgSize * tN columns
        {
            if (wgSize % sgSize || (wgSize / sgSize) % sgpMIter) return fail("subgroup grid");
            wgM = wgSize / sgSize / sgpMIter * tM;
            wgN = sgpMIter * sgSize * tN;
            // shared memory: a vec4 slot is 8 bytes (16 with linear-tile-11's fp32 staging);
            // I wgM x KC/4, W wgN x (KC/4 + pad)
            const uint32_t wPad = (shader >= 11) ? sgN : 1u;   // linear-tile-11 takes WPAD from sgN
            const uint32_t slotBytes = (shader >= 11 && stage32) ? 16u : 8u;
            if (shader == 7) wgK = sgSize;   // linear-tile-7 walks K sgSize at a time
            else if (wgK % 4 || (wgM * wgK / 4 + (shader >= 9 ? wgN * (wgK / 4 + wPad) : 0)) * slotBytes > maxShared)
                return fail("KC");
            reqSg = sgSize;    // the lane math needs the real subgroup size
            sgM = tM; sgpM = sgpN = 0;
            if (shader < 11) sgN = wgN;
            return true;
        }
        if (tN % 2) return fail("tN odd");
        if ((wgSize * tM * tN * sgpMIter * sgpNIter) % wgN) return fail("wgM not integral");
        wgM = wgSize * tM * tN * sgpMIter * sgpNIter / wgN;
        if (sgN % sgpNIter) return fail("sgN % sgpNIter");
        sgpN = sgN / sgpNIter;
        if ((sgSize * tM * tN) % sgpN) return fail("sgpM not integral");
        sgpM = sgSize * tM * tN / sgpN;
        sgM = sgpMIter * sgpM;
        if (wgM % sgM || wgN % sgN) return fail("subgroup tiles do not tile the workgroup");
        if (sgpM % tM || sgpN % tN) return fail("thread tiles do not tile the piece");
        if (wgSize != (wgM / sgM) * (wgN / sgN) * sgSize) return fail("wgSize mismatch");
        if (wgK % 8 || wgSize % (wgK / 8)) return fail("loader: wgK");
        const uint32_t loadstride = wgSize * 8 / wgK;
        if (loadstride == 0 || wgM % loadstride || wgN % loadstride) return fail("loader: stride");
        if ((wgN + wgM) * (wgK / 2 + 1) * 4 > maxShared) return fail("shared memory");
        return true;
    }

    std::string str() const
    {
        char b[160];
        snprintf(b, sizeof(b), "s%u wg%-3u sg%-2u t%ux%u it%ux%u tile%ux%u k%-3u sgN%-3u %s%s",
                 shader, wgSize, sgSize, tM, tN, sgpMIter, sgpNIter, wgM, wgN, wgK, sgN,
                 acc ? "f16acc" : "f32acc", (reqSg ? " req" + std::to_string(reqSg) : std::string()).c_str());
        if (opt) snprintf(b + strlen(b), sizeof(b) - strlen(b), " opt%u", opt);
        if (stage32) snprintf(b + strlen(b), sizeof(b) - strlen(b), " stage32");
        if (swizzle1) return std::string("s1sw") + (b + 2) + " swz" + std::to_string(opt);
        if (probe11) return std::string("s11p") + (b + 3) + " order" + std::to_string(sgpNIter);
        if (shader == 21)
            snprintf(b, sizeof(b), "s21 b%u o%u k%u bsv%u fsv%u sp%u mf%u %s", tM, tN, wgK, sgpNIter, sgpMIter,
                     std::max(sgN, 1u), stage32, acc ? "f16acc" : "f32acc");
        if (shader == 21 && opt && !ov21) snprintf(b + strlen(b), sizeof(b) - strlen(b), " v%u", opt);
        return b;
    }
};

Config parse(const std::string& s)
{
    std::vector<uint32_t> v;
    std::stringstream ss(s);
    bool swizzle1 = false, probe11 = false;
    for (std::string tok; std::getline(ss, tok, ',');)
    {
        if (v.empty() && tok == "1-swizzle") { swizzle1 = true; v.push_back(1); continue; }
        if (v.empty() && tok == "11-probe") { probe11 = true; v.push_back(11); continue; }
        if (v.empty() && (tok == "11-node" || tok == "11-node32" || tok == "21-ov"))
        {
            Config c;
            c.shader = (tok == "21-ov") ? 21 : 11;
            c.node11 = c.shader == 11;
            c.acc = (tok == "11-node32") ? 0 : 1;
            c.ov21 = c.shader == 21;
            if (c.ov21 && std::getline(ss, tok, ',')) c.opt = uint32_t(std::stoul(tok));   // the M OpenVINO picks for
            return c;
        }
        v.push_back(uint32_t(std::stoul(tok)));
    }
    if (v.size() < 12 || v.size() > 14) { fprintf(stderr, "bad config: %s\n", s.c_str()); exit(1); }
    Config c;
    c.shader = v[0]; c.wgSize = v[1]; c.sgSize = v[2]; c.tN = v[3]; c.tM = v[4];
    c.sgpNIter = v[5]; c.sgpMIter = v[6]; c.wgN = v[7]; c.wgK = v[8]; c.sgN = v[9];
    c.acc = v[10]; c.reqSg = v[11];
    if (v.size() >= 13) c.opt = v[12];
    if (v.size() >= 14) c.stage32 = v[13];
    c.swizzle1 = swizzle1;
    c.probe11 = probe11;
    return c;
}

uint16_t toHalf(float f)
{
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = int32_t((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0)                       // subnormal or zero
    {
        if (e < -10) return uint16_t(sign);
        m |= 0x800000u;
        const uint32_t shift = uint32_t(14 - e);
        uint32_t h = m >> shift;
        if ((m >> (shift - 1)) & 1u) h += 1;
        return uint16_t(sign | h);
    }
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    uint32_t h = sign | (uint32_t(e) << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1u))) h += 1;   // round to nearest even
    return uint16_t(h);
}

float fromHalf(uint16_t h)
{
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu;
    float f;
    if (e == 0) f = std::ldexp(float(m), -24);
    else if (e == 31) f = INFINITY;
    else f = std::ldexp(float(m | 0x400u), int(e) - 25);
    uint32_t x; std::memcpy(&x, &f, 4); x |= sign; std::memcpy(&f, &x, 4);
    return f;
}

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

// Every statistic the driver reports for the pipeline, name=value.
std::string statSummary(const eva::ComputePipeline& pipe)
{
    std::string out;
    for (const eva::PipelineExecutable& e : pipe.executables())
        out += " [" + e.name + " sg" + std::to_string(e.subgroupSize) + "]";
    for (const eva::PipelineStatistic& s : pipe.statistics())
        out += " " + s.name + "=" + s.value;
    return out;
}

// The driver's text representations (on Intel, the ISA disassembly).
void printIsa(const eva::ComputePipeline& pipe)
{
    const auto irs = pipe.internalRepresentations();
    if (irs.empty())
        printf("---- (the driver reports no internal representation)\n");
    for (const eva::PipelineInternalRepresentation& r : irs)
        if (r.isText)
            printf("---- %s\n%.*s\n", r.name.c_str(), int(r.data.size()), (const char*)r.data.data());
        else
            printf("---- %s: binary, %zu bytes\n", r.name.c_str(), r.data.size());
}

// fma-peak.comp over each mode and subgroup size: independent FMA chains only,
// no memory traffic but one store per thread.
int peakProbe(eva::Device dev, eva::CommandBuffer cmd, eva::QueryPool qp, bool dumpIsa)
{
    const std::string path = std::string(TILE_SHADER_DIR) + "/fma-peak.comp.spv";
    const eva::SpvBlob spv = eva::SpvBlob::readFrom(path.c_str());
    const uint32_t wgSize = 128, wgs = 4096, threads = wgSize * wgs, iters = 2048, chains = 16;
    eva::Buffer bufO = deviceBuffer(dev, uint64_t(threads) * 4);
    const auto stage = eva::PIPELINE_STAGE::ALL_COMMANDS;
    const char* names[] = { "fp32", "fp16", "f16vec2" };

    printf("\n%-8s %5s | %9s %9s | %s\n", "mode", "reqSg", "ms", "TFLOP/s", "compiled");
    for (uint32_t mode = 0; mode < 3; ++mode)
        for (uint32_t reqSg : { 0u, 8u, 16u, 32u })
        {
            eva::ComputePipeline pipe = dev.createComputePipeline({
                .csStage = spv + eva::constant_id<0>(wgSize) + eva::constant_id<1>(mode),
                .requiredSubgroupSize = reqSg,
                .captureStatistics = true,
            });
            eva::DescriptorSet ds = gVaiDestSetPool(pipe.descSetLayout(0));
            ds.write({ bufO });
            struct { uint32_t iters; float a, b; } pc = { iters, 0.999f, 0.001f };
            auto record = [&]
            {
                cmd.bindPipeline(pipe).bindDescSets({ds})
                    .setPushConstants(0, sizeof(pc), &pc)
                    .dispatch(wgs, 1, 1)
                    .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / bufO / eva::SYNC_SCOPE::COMPUTE_WRITE);
            };
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT);
            record();
            cmd.end().submit().wait();

            const uint32_t reps = 10;
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
            cmd.writeTimestamp(stage, qp, 0);
            for (uint32_t r = 0; r < reps; ++r) record();
            cmd.writeTimestamp(stage, qp, 1);
            cmd.end().submit().wait();
            const double ms = qp.getElapsedMs(0, 1) / reps;

            const double flop = double(threads) * iters * chains * 2 * (mode == 2 ? 2 : 1);
            std::string compiled;
            for (const eva::PipelineExecutable& e : pipe.executables()) compiled += "SIMD" + std::to_string(e.subgroupSize) + " ";
            printf("%-8s %5u | %9.3f %9.2f | %s\n", names[mode], reqSg, ms, flop / (ms * 1e9), compiled.c_str());
            if (dumpIsa) printIsa(pipe);
        }
    return 0;
}

// walk-probe.comp on one grid: per workgroup x, y, start ticket, ended-before,
// end ticket, written as CSV. spec = gx,gy,wgSize,iters,slmWords,reqSg,path.
int walkProbe(eva::Device dev, eva::CommandBuffer cmd, eva::QueryPool qp, const std::string& spec)
{
    uint32_t gx, gy, wgSize, iters, slmWords, reqSg;
    char path[1024] = {};
    if (std::sscanf(spec.c_str(), "%u,%u,%u,%u,%u,%u,%1023s", &gx, &gy, &wgSize, &iters, &slmWords, &reqSg, path) != 7)
        { fprintf(stderr, "bad -walk: %s\n", spec.c_str()); return 1; }

    const eva::SpvBlob spv = eva::SpvBlob::readFrom((std::string(TILE_SHADER_DIR) + "/walk-probe.comp.spv").c_str());
    eva::ComputePipeline pipe = dev.createComputePipeline({
        .csStage = spv + eva::constant_id<0>(wgSize) + eva::constant_id<1>(iters) + eva::constant_id<2>(slmWords),
        .requiredSubgroupSize = reqSg,
    });
    const uint64_t words = 2ull + 3ull * gx * gy, bytes = words * 4;
    eva::Buffer buf = deviceBuffer(dev, bytes), stage = hostBuffer(dev, bytes);
    eva::DescriptorSet ds = gVaiDestSetPool(pipe.descSetLayout(0));
    ds.write({ buf });
    const auto ts = eva::PIPELINE_STAGE::ALL_COMMANDS;

    for (int pass = 0; pass < 2; ++pass)   // the first pass warms the pipeline up
    {
        std::memset(stage.map(), 0, bytes);
        cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2)
            .copyBuffer(stage(0, bytes), buf(0, bytes))
            .barrier(eva::SYNC_SCOPE::TRANSFER_DST / buf / eva::SYNC_SCOPE::COMPUTE_READ);
        cmd.writeTimestamp(ts, qp, 0);
        cmd.bindPipeline(pipe).bindDescSets({ds}).dispatch(gx, gy, 1);
        cmd.writeTimestamp(ts, qp, 1);
        cmd.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / buf / eva::SYNC_SCOPE::TRANSFER_SRC)
            .copyBuffer(buf(0, bytes), stage(0, bytes));
        cmd.end().submit().wait();
    }
    const uint32_t* w = (const uint32_t*)stage.map();
    FILE* f = std::fopen(path, "w");
    std::fprintf(f, "x,y,start,ended_before,end\n");
    for (uint32_t y = 0; y < gy; ++y)
        for (uint32_t x = 0; x < gx; ++x)
        {
            const uint64_t id = x + uint64_t(y) * gx;
            std::fprintf(f, "%u,%u,%u,%u,%u\n", x, y, w[2 + 3 * id], w[3 + 3 * id], w[4 + 3 * id]);
        }
    std::fclose(f);
    printf("walk %ux%u wg%u iters%u slm%u sg%u: %.3f ms -> %s\n", gx, gy, wgSize, iters, slmWords, reqSg,
           qp.getElapsedMs(0, 1), path);
    return 0;
}

// share-probe.comp: one dispatch of gx workgroups per sharer stride 1, 2, 4,
// 8, 16, each timed. spec = gx,wgSize,reqSg,G,regionVec4,reps.
int shareProbe(eva::Device dev, eva::CommandBuffer cmd, eva::QueryPool qp, const std::string& spec)
{
    uint32_t gx, wgSize, reqSg, g, region, reps;
    if (std::sscanf(spec.c_str(), "%u,%u,%u,%u,%u,%u", &gx, &wgSize, &reqSg, &g, &region, &reps) != 6)
        { fprintf(stderr, "bad -share: %s\n", spec.c_str()); return 1; }

    const eva::SpvBlob spv = eva::SpvBlob::readFrom((std::string(TILE_SHADER_DIR) + "/share-probe.comp.spv").c_str());
    const uint64_t dataBytes = uint64_t(gx / g) * region * 16, outBytes = uint64_t(gx) * wgSize * 4;
    eva::Buffer data = deviceBuffer(dev, dataBytes), out = deviceBuffer(dev, outBytes);
    const uint32_t strides[] = { 1, 2, 4, 8, 16 };
    double ms[5] = {};
    for (uint32_t si = 0; si < 5; ++si)
    {
        eva::ComputePipeline pipe = dev.createComputePipeline({
            .csStage = spv + eva::constant_id<0>(wgSize) + eva::constant_id<1>(g) + eva::constant_id<2>(strides[si])
                           + eva::constant_id<3>(region) + eva::constant_id<4>(reps),
            .requiredSubgroupSize = reqSg,
        });
        eva::DescriptorSet ds = gVaiDestSetPool(pipe.descSetLayout(0));
        ds.write({ data, out });
        std::vector<double> runs;
        for (int it = 0; it < 21; ++it)   // the first run warms up
        {
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
            cmd.writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, qp, 0);
            cmd.bindPipeline(pipe).bindDescSets({ds}).dispatch(gx, 1, 1);
            cmd.writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, qp, 1);
            cmd.end().submit().wait();
            if (it) runs.push_back(qp.getElapsedMs(0, 1));
        }
        std::sort(runs.begin(), runs.end());
        ms[si] = runs[runs.size() / 2];
    }
    printf("share %u wgs wg%u sg%u G%u region %u KB reps%u, ms by stride 1 / 2 / 4 / 8 / 16: %.3f %.3f %.3f %.3f %.3f\n",
           gx, wgSize, reqSg, g, region * 16 / 1024, reps, ms[0], ms[1], ms[2], ms[3], ms[4]);
    return 0;
}

// slm-probe.comp: one dispatch of gx workgroups (subgroup 16) per W row stride
// 2, 3, 4 uvec2 slots (WPAD 0, 1, 2 at wgK 8), each timed. spec = gx,wgSize,mode,iters,tN.
int slmProbe(eva::Device dev, eva::CommandBuffer cmd, eva::QueryPool qp, const std::string& spec)
{
    uint32_t gx, wgSize, mode, iters, tn;
    if (std::sscanf(spec.c_str(), "%u,%u,%u,%u,%u", &gx, &wgSize, &mode, &iters, &tn) != 5)
        { fprintf(stderr, "bad -slm: %s\n", spec.c_str()); return 1; }

    const eva::SpvBlob spv = eva::SpvBlob::readFrom((std::string(TILE_SHADER_DIR) + "/slm-probe.comp.spv").c_str());
    eva::Buffer out = deviceBuffer(dev, uint64_t(gx) * wgSize * 4);
    printf("slm %s wg%u x %u iters %u tN %u, ms by stride 2 / 3 / 4:", mode ? "write" : "read", wgSize, gx, iters, tn);
    for (uint32_t stride = 2; stride <= 4; ++stride)
    {
        eva::ComputePipeline pipe = dev.createComputePipeline({
            .csStage = spv + eva::constant_id<0>(wgSize) + eva::constant_id<1>(stride) + eva::constant_id<2>(mode)
                           + eva::constant_id<3>(iters) + eva::constant_id<4>(tn)
                           + eva::constant_id<5>(std::getenv("SLM_OFF") ? uint32_t(std::atoi(std::getenv("SLM_OFF"))) : 0u),
            .requiredSubgroupSize = 16,
            .captureStatistics = true,
        });
        eva::DescriptorSet ds = gVaiDestSetPool(pipe.descSetLayout(0));
        ds.write({ out });
        std::vector<double> runs;
        for (int it = 0; it < 21; ++it)   // the first run warms up
        {
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
            cmd.writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, qp, 0);
            cmd.bindPipeline(pipe).bindDescSets({ds}).dispatch(gx, 1, 1);
            cmd.writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, qp, 1);
            cmd.end().submit().wait();
            if (it) runs.push_back(qp.getElapsedMs(0, 1));
        }
        std::sort(runs.begin(), runs.end());
        printf("  %.3f", runs[runs.size() / 2]);
        if (std::getenv("SLM_STATS")) printf(" [%s]", statSummary(pipe).c_str());
    }
    printf("\n");
    return 0;
}

// rows-probe.comp: the W stream alone. A buffer of MB megabytes (default 256) in
// rows strideBytes apart; each workgroup stages `rows` of them, chunk4 vec4s a
// row per step, along rowBytes of each. Prints the rows one wave keeps in flight,
// their address span and the read rate (median of 9 timed runs).
int rowsProbe(eva::Device dev, eva::CommandBuffer cmd, eva::QueryPool qp, const std::string& spec)
{
    uint32_t wgSize, rows, chunk4, rowBytes, strideBytes, mb = 256;
    if (std::sscanf(spec.c_str(), "%u,%u,%u,%u,%u,%u", &wgSize, &rows, &chunk4, &rowBytes, &strideBytes, &mb) < 5
        || rowBytes > strideBytes || rowBytes % (16 * chunk4) || strideBytes % 16)
        { fprintf(stderr, "bad -rows: %s\n", spec.c_str()); return 1; }

    const uint64_t totalRows = (uint64_t(mb) << 20) / strideBytes / rows * rows;
    const uint32_t gx = uint32_t(totalRows / rows);
    eva::Buffer data = deviceBuffer(dev, totalRows * strideBytes);
    eva::Buffer out = deviceBuffer(dev, uint64_t(gx) * wgSize * 4);
    const eva::SpvBlob spv = eva::SpvBlob::readFrom((std::string(TILE_SHADER_DIR) + "/rows-probe.comp.spv").c_str());
    eva::ComputePipeline pipe = dev.createComputePipeline({
        .csStage = spv + eva::constant_id<0>(wgSize) + eva::constant_id<1>(rows) + eva::constant_id<2>(chunk4),
        .requiredSubgroupSize = 16,
    });
    eva::DescriptorSet ds = gVaiDestSetPool(pipe.descSetLayout(0));
    ds.write({ data, out });
    struct { uint32_t stride4, steps; } pc = { strideBytes / 16, rowBytes / (16 * chunk4) };

    std::vector<double> runs;
    for (int it = 0; it < 10; ++it)   // the first run warms up
    {
        cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
        cmd.writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, qp, 0);
        cmd.bindPipeline(pipe).bindDescSets({ds}).setPushConstants(0, sizeof(pc), &pc).dispatch(gx, 1, 1);
        cmd.writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, qp, 1);
        cmd.end().submit().wait();
        if (it) runs.push_back(qp.getElapsedMs(0, 1));
    }
    std::sort(runs.begin(), runs.end());
    const double ms = runs[runs.size() / 2];
    const uint32_t inFlight = 8u * (128u / (wgSize / 16u)) * rows;   // Meteor Lake: 8 Xe-cores x 128 threads
    printf("rows wg%u r%u c%u row %u B stride %u B: in flight %u rows, span %.1f MB, %.3f ms, %.1f GB/s\n",
           wgSize, rows, chunk4, rowBytes, strideBytes, inFlight, double(inFlight) * strideBytes / (1 << 20), ms,
           double(totalRows) * rowBytes / (ms * 1e6));
    return 0;
}

// OpenVINO's FullyConnected_bf_tiled::GetAutoTuneParams for fp16 input, a
// static shape and uncompressed weights (fully_connected_gpu_bf_tiled.cpp): the
// first case TuneParamsSelector::VerifyTuneParams accepts, in OpenVINO's order.
struct OvTune { uint32_t tileB, tileOfm, tileK, bsv, fsv; };

bool ovVerify(uint32_t M, uint32_t N, const OvTune& t)
{
    const uint32_t simd = 16;
    if (M % (t.tileB * t.bsv) != 0)
    {
        if (t.bsv != 1 || M == 1) return false;
        uint32_t tile = simd;
        while (M % tile != 0) tile--;
        if (tile > 1) return false;
    }
    if (CEIL_DIV(N, t.tileOfm * simd) % t.fsv != 0) return false;
    if (N <= (t.tileOfm / 2) * simd) return false;
    if (t.tileOfm * simd > 64) return false;
    const uint32_t regBytes = t.tileB * t.tileOfm * simd * 2 + t.tileB * simd * 2 + t.tileOfm * t.tileK * simd * 2;
    return regBytes <= 128 * 32;
}

OvTune ovPick(uint32_t M, uint32_t N)
{
    uint32_t maxOfm = 1;
    while (maxOfm * 2 * 16 <= N && maxOfm < 4) maxOfm *= 2;
    const uint32_t ofm = std::min(maxOfm, 2u);
    const OvTune cases[] = {
        { 8, ofm, 2, 16, 2 }, { 8, ofm, 2, 16, 1 }, { 16, ofm, 2, 4, 2 }, { 8, ofm, 2, 8, 1 },
        { 16, ofm, 2, 2, 2 }, { 8, ofm, 2, 4, 1 }, { 16, ofm, 2, 1, 1 }, { 8, ofm, 2, 1, 1 },
    };
    for (const OvTune& t : cases)
        if (ovVerify(M, N, t)) return t;
    OvTune last = { 8, ofm, 2, 16, 1 };   // the last case: the largest tile_b and dispatch_bsv dividing M
    while (M % last.tileB != 0) last.tileB--;
    while (M % (last.tileB * last.bsv) != 0) last.bsv--;
    if (ovVerify(M, N, last)) return last;
    return { 1, 1, 1, 1, 1 };
}

}   // namespace


int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    eva::Device dev = getVaiDevice();

    std::vector<Shape> shapes = {
        { "gate+up", 922,  2048, 32768 },
        { "down",    922, 16384,  2048 },
    };
    bool dumpStats = false, quick = false, dumpIsa = false, peak = false, i16 = false, clk = false, w32 = false;
    uint32_t passes = 0;    // -passes P: time every configuration of a shape in P interleaved passes
    bool flush = false;     // -flush (with -passes): a 32 MB copy evicts the GPU L2 ahead of every timed run
    std::string probeDir;   // -probe <dir>: where linear-tile-11-probe writes its per-workgroup records
    std::vector<std::string> walks;   // -walk gx,gy,wgSize,iters,slmWords,reqSg,path (repeatable)
    std::vector<std::string> shares;  // -share gx,wgSize,reqSg,G,regionVec4,reps (repeatable)
    std::vector<std::string> slms;    // -slm gx,wgSize,mode,iters,tN (repeatable)
    std::vector<std::string> rowss;   // -rows wgSize,rows,chunk4,rowBytes,strideBytes[,MB] (repeatable)
    std::vector<Shape> custom;
    std::vector<Config> configs;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "gu")          shapes.erase(shapes.begin() + 1);
        else if (a == "dn")     shapes.erase(shapes.begin());
        else if (a == "-stats") dumpStats = true;
        else if (a == "-quick") quick = true;
        else if (a == "-isa")   dumpIsa = true;
        else if (a == "-peak")  peak = true;
        else if (a == "-i16")   i16 = true;
        else if (a == "-clk")   clk = true;
        else if (a == "-w32")   w32 = true;
        else if (a == "-probe" && i + 1 < argc) probeDir = argv[++i];
        else if (a == "-passes" && i + 1 < argc) passes = uint32_t(std::atoi(argv[++i]));
        else if (a == "-flush") flush = true;
        else if (a == "-walk" && i + 1 < argc)  walks.push_back(argv[++i]);
        else if (a == "-share" && i + 1 < argc) shares.push_back(argv[++i]);
        else if (a == "-slm" && i + 1 < argc)   slms.push_back(argv[++i]);
        else if (a == "-rows" && i + 1 < argc)  rowss.push_back(argv[++i]);
        else if (a == "-shape" && i + 1 < argc)
        {
            uint32_t m = 0, k = 0, n = 0;
            if (std::sscanf(argv[++i], "%u,%u,%u", &m, &k, &n) != 3 || !m || !k || !n)
                { fprintf(stderr, "bad shape: %s\n", argv[i]); return 1; }
            custom.push_back({ std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n), m, k, n });
        }
        else                    configs.push_back(parse(a));
    }
    if (!custom.empty())
        shapes = custom;
    if (configs.empty())   // the demo's pick here: M tier on a 32-wide default subgroup, fp16 accumulation
        configs.push_back(parse("1,128,32,4,2,2,2,64,32,32,1,0"));

    const uint32_t maxShared = dev.maxComputeSharedMemorySize();
    printf("device subgroup %u (min %u, max %u), shared %u bytes\n",
           dev.subgroupSize(), dev.minSubgroupSize(), dev.maxSubgroupSize(), maxShared);

    eva::CommandBuffer cmd = dev.newCommandBuffer(eva::queue_compute, eva::COMMAND_POOL_CREATE::RESET_COMMAND_BUFFER);
    eva::QueryPool qp = dev.createTimestampQueryPool(2);
    const auto stage = eva::PIPELINE_STAGE::ALL_COMMANDS;
    if (peak)
        return peakProbe(dev, cmd, qp, dumpIsa);
    if (!walks.empty())
    {
        for (const std::string& w : walks)
            if (walkProbe(dev, cmd, qp, w)) return 1;
        return 0;
    }
    if (!shares.empty())
    {
        for (const std::string& w : shares)
            if (shareProbe(dev, cmd, qp, w)) return 1;
        return 0;
    }
    if (!slms.empty())
    {
        for (const std::string& w : slms)
            if (slmProbe(dev, cmd, qp, w)) return 1;
        return 0;
    }
    if (!rowss.empty())
    {
        for (const std::string& w : rowss)
            if (rowsProbe(dev, cmd, qp, w)) return 1;
        return 0;
    }

    // split-reduce.comp for linear-tile-10 / 11's partials.
    eva::ComputePipeline reducePipe = dev.createComputePipeline({
        .csStage = eva::SpvBlob::readFrom((std::string(TILE_SHADER_DIR) + "/split-reduce.comp.spv").c_str()),
    });

    // -clk: a short fp16 FMA run (fma-peak.comp) before and after each timed
    // configuration. Its throughput tracks the GPU clock at that moment.
    eva::ComputePipeline clkPipe;
    eva::DescriptorSet clkSet;
    eva::Buffer clkBuf;
    const uint32_t clkWgs = 1024, clkIters = 1024;
    if (clk)
    {
        const std::string path = std::string(TILE_SHADER_DIR) + "/fma-peak.comp.spv";
        clkPipe = dev.createComputePipeline({
            .csStage = eva::SpvBlob::readFrom(path.c_str()) + eva::constant_id<0>(128u) + eva::constant_id<1>(1u),
            .requiredSubgroupSize = 32,
        });
        clkBuf = deviceBuffer(dev, uint64_t(clkWgs) * 128 * 4);
        clkSet = gVaiDestSetPool(clkPipe.descSetLayout(0));
        clkSet.write({ clkBuf });
    }
    auto clockProxy = [&]() -> double
    {
        struct { uint32_t iters; float a, b; } pc = { clkIters, 0.999f, 0.001f };
        const uint32_t reps = 5;
        cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
        cmd.writeTimestamp(stage, qp, 0);
        for (uint32_t r = 0; r < reps; ++r)
            cmd.bindPipeline(clkPipe).bindDescSets({clkSet})
                .setPushConstants(0, sizeof(pc), &pc)
                .dispatch(clkWgs, 1, 1)
                .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / clkBuf / eva::SYNC_SCOPE::COMPUTE_WRITE);
        cmd.writeTimestamp(stage, qp, 1);
        cmd.end().submit().wait();
        const double flop = double(clkWgs) * 128 * clkIters * 16 * 2 * reps;
        return flop / (qp.getElapsedMs(0, 1) * 1e9);
    };

    const eva::SpvBlob swizzle1Spv = eva::SpvBlob::readFrom((std::string(TILE_SHADER_DIR) + "/linear-tile-1-swizzle.comp.spv").c_str());
    const eva::SpvBlob probe11Spv = eva::SpvBlob::readFrom((std::string(TILE_SHADER_DIR) + "/linear-tile-11-probe.comp.spv").c_str());
    std::map<uint32_t, eva::SpvBlob> spvs;   // the linear-tile-N.comp variants kept
    for (uint32_t s : { 1u, 11u, 21u })
        spvs.emplace(s, eva::SpvBlob::readFrom((std::string(TILE_SHADER_DIR) + "/linear-tile-" + std::to_string(s) + ".comp.spv").c_str()));

    for (const Shape& sh : shapes)
    {
        // Inputs, weights (fp16) and the CPU reference on sampled rows.
        std::mt19937 rng(sh.K * 7u + sh.N);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        std::vector<float> x(size_t(sh.M) * sh.K);
        std::vector<float> wf(size_t(sh.N) * sh.K);   // the weights as uploaded (fp16-rounded unless -w32)
        std::vector<uint16_t> wh(w32 ? 0 : wf.size());
        for (float& v : x) v = uni(rng);
        for (float& v : wf) v = uni(rng) * 0.05f;
        if (!w32)
            for (size_t i = 0; i < wf.size(); ++i) { wh[i] = toHalf(wf[i]); wf[i] = fromHalf(wh[i]); }

        const uint32_t rows[] = { 0, 1, 2, sh.M / 2, sh.M - 2, sh.M - 1 };
        std::vector<double> ref(std::size(rows) * size_t(sh.N));
        {
            std::vector<float> wrow(sh.K);
            for (uint32_t n = 0; n < sh.N; ++n)
            {
                for (uint32_t k = 0; k < sh.K; ++k) wrow[k] = wf[size_t(n) * sh.K + k];
                for (size_t r = 0; r < std::size(rows); ++r)
                {
                    const float* xr = &x[size_t(rows[r]) * sh.K];
                    double acc = 0;
                    for (uint32_t k = 0; k < sh.K; ++k) acc += double(xr[k]) * wrow[k];
                    ref[r * sh.N + n] = acc;
                }
            }
        }

        const uint64_t iBytes = uint64_t(sh.M) * sh.K * (i16 ? 2 : 4), wBytes = uint64_t(sh.N) * sh.K * (w32 ? 4 : 2), oBytes = uint64_t(sh.M) * sh.N * 4;
        eva::Buffer bufI = deviceBuffer(dev, iBytes), bufW = deviceBuffer(dev, wBytes), bufO = deviceBuffer(dev, oBytes);
        {
            eva::Buffer stI = hostBuffer(dev, iBytes), stW = hostBuffer(dev, wBytes);
            if (i16)   // I handed over in fp16, as a conversion pass ahead of the GEMM would
            {
                std::vector<uint16_t> xh(x.size());
                for (size_t i = 0; i < x.size(); ++i) xh[i] = toHalf(x[i]);
                std::memcpy(stI.map(), xh.data(), iBytes);
            }
            else
                std::memcpy(stI.map(), x.data(), iBytes);
            std::memcpy(stW.map(), w32 ? (const void*)wf.data() : (const void*)wh.data(), wBytes);
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
                .copyBuffer(stI(0, iBytes), bufI(0, iBytes))
                .copyBuffer(stW(0, wBytes), bufW(0, wBytes))
            .end().submit().wait();
        }
        eva::Buffer readback = hostBuffer(dev, oBytes);

        // linear-tile-21: W in osv = tileOfm * 16 column blocks, a lane's tileK x tileOfm weights of a k group
        // together: W'[n / osv, k / tileK, n % 16, (k % tileK) * tileOfm + (n % osv) / 16], padded with zeros.
        std::map<uint32_t, eva::Buffer> osvW;
        auto reorderedW = [&](uint32_t tileOfm, uint32_t tileK) -> eva::Buffer
        {
            const uint32_t key = tileOfm * 16 + tileK, osv = tileOfm * 16, kGroups = CEIL_DIV(sh.K, tileK);
            if (osvW.count(key)) return osvW.at(key);
            const uint64_t elems = uint64_t(CEIL_DIV(sh.N, osv)) * kGroups * 16 * tileK * tileOfm, bytes = elems * (w32 ? 4 : 2);
            eva::Buffer st = hostBuffer(dev, bytes), dst = deviceBuffer(dev, bytes);
            uint8_t* p = (uint8_t*)st.map();
            std::memset(p, 0, bytes);
            for (uint32_t n = 0; n < sh.N; ++n)
                for (uint32_t k = 0; k < sh.K; ++k)
                {
                    const uint64_t d = ((uint64_t(n / osv) * kGroups + k / tileK) * 16 + n % 16) * tileK * tileOfm
                                     + (k % tileK) * tileOfm + (n % osv) / 16;
                    const uint64_t s = uint64_t(n) * sh.K + k;
                    if (w32) std::memcpy(p + d * 4, &wf[s], 4);
                    else     std::memcpy(p + d * 2, &wh[s], 2);
                }
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).copyBuffer(st(0, bytes), dst(0, bytes)).end().submit().wait();
            return osvW.emplace(key, dst).first->second;
        };

        const double gflop = 2.0 * sh.M * sh.K * sh.N / 1e9;
        printf("\n%s  M %u  K %u  N %u  (%.1f GFLOP)\n", sh.what.c_str(), sh.M, sh.K, sh.N, gflop);

        struct Visit { std::string name; std::function<void()> record; double ms; };
        std::vector<Visit> visits;   // -passes: every configuration of this shape, timed again below

        // One partials buffer per shape, sized for the largest split asked, shared by every configuration:
        // they run one at a time, and one buffer per configuration (all alive for -passes) overflowed the
        // iGPU's memory on large shapes, paging buffers and slowing the configurations measured after.
        std::vector<Config> resolved = configs;
        for (Config& c : resolved)
        {
            if (c.node11)   // intel-tile-v11.h: tile, order and split
            {
                const Dtype iDtype = i16 ? Dtype::Fp16 : Dtype::Fp32, wDtype = w32 ? Dtype::Fp32 : Dtype::Fp16;
                const bool stageF32 = !c.acc && iDtype == Dtype::Fp32 && wDtype == Dtype::Fp32;   // linear-tile-11's staging rule
                const v11::IntelTilePick ip = v11::intelTileParams(sh.M, sh.N, sh.K, !c.acc, stageF32, iDtype, wDtype, dev);
                const node::Linear::PipeParamsTile& t = ip.tile;
                c.wgSize = t.wgSize; c.sgSize = c.reqSg = t.sgSize; c.tN = t.tN; c.tM = t.tM;
                c.sgpNIter = v11::intelTileNFirst(t, wDtype) ? 0u : 1u;
                c.sgpMIter = t.wgN / (t.sgSize * t.tN);
                c.wgN = t.wgN; c.wgK = t.wgK; c.sgN = 0; c.stage32 = stageF32;
                c.opt = ip.splits > 1 ? ip.splits : 0u;
            }
            if (c.ov21)
            {
                const OvTune t = ovPick(c.opt ? c.opt : sh.M, sh.N);
                c.tN = t.tileOfm; c.tM = t.tileB; c.wgK = t.tileK; c.sgpNIter = t.bsv; c.sgpMIter = t.fsv; c.acc = 1;
            }
        }
        uint32_t maxSplitReq = 0;
        for (const Config& c : resolved)
        {
            if (c.shader == 10 || c.shader == 11) maxSplitReq = std::max(maxSplitReq, std::max(c.opt, 1u));
            if (c.shader == 21)                   maxSplitReq = std::max(maxSplitReq, std::max(c.sgN, 1u));
        }
        eva::Buffer sharedP = maxSplitReq ? deviceBuffer(dev, uint64_t(maxSplitReq) * oBytes) : eva::Buffer();

        for (Config c : resolved)
        {
            if (!c.derive(maxShared) || (c.shader < 11 && sh.K % c.wgK))
            {
                printf("  %-58s  skipped: %s\n", c.str().c_str(), c.why.empty() ? "K % wgK" : c.why.c_str());
                continue;
            }
            if (w32 && c.shader >= 7 && c.shader <= 10)
            {
                printf("  %-58s  skipped: fp16 W only\n", c.str().c_str());
                continue;
            }
            if (c.reqSg && (c.reqSg < dev.minSubgroupSize() || c.reqSg > dev.maxSubgroupSize()))
            {
                printf("  %-58s  skipped: subgroup %u unsupported\n", c.str().c_str(), c.reqSg);
                continue;
            }
            // linear-tile-1-swizzle: opt is the swizzle group, 1 .. N tiles (constant 13)
            if (c.swizzle1)
                c.opt = std::clamp(c.opt, 1u, uint32_t(CEIL_DIV(sh.N, c.wgN)));
            // fp32 I and W make linear-tile-1 ~ 6 stage in fp32, which accumulates in fp32 (as Linear passes it)
            const bool stageF32 = c.shader <= 6 && w32 && !i16;
            // linear-tile-11 picks vec4 loads when K allows them, masked scalar loads otherwise
            const uint32_t alignedK = (c.shader == 21) ? (sh.K % c.wgK == 0 ? 1u : 0u)   // linear-tile-21: tileK-wide W rows
                                    : (c.shader >= 11) ? ((sh.K % c.wgK == 0 && sh.K % 4 == 0) ? 1u : 0u) : 1u;
            if (!c.swizzle1 && !c.probe11 && !spvs.count(c.shader))
            {
                printf("  %-58s  skipped: no shader %u\n", c.str().c_str(), c.shader);
                continue;
            }

            // clang-format off
            eva::ComputePipeline pipe = dev.createComputePipeline({
                .csStage = (c.swizzle1 ? swizzle1Spv : c.probe11 ? probe11Spv : spvs.at(c.shader))
                    + eva::constant_id<0>(c.wgSize)
                    + eva::constant_id<1>(c.sgSize)
                    + eva::constant_id<2>(c.wgM)
                    + eva::constant_id<3>(c.wgN)
                    + eva::constant_id<4>(c.wgK)
                    + eva::constant_id<5>(c.sgM)
                    + eva::constant_id<6>(c.sgN)
                    + eva::constant_id<7>(c.tM)
                    + eva::constant_id<8>(c.tN)
                    + eva::constant_id<9>(c.sgpMIter)
                    + eva::constant_id<10>(c.shader == 11 && !c.probe11 ? (c.sgpNIter ? 0u : 1u) : c.sgpNIter)   // 11: nFirst = !order
                    + eva::constant_id<11>(stageF32 || !c.acc ? Dtype::Fp32 : Dtype::Fp16)
                    + eva::constant_id<12>(0u)
                    + eva::constant_id<13>(c.opt)
                    + eva::constant_id<14>(c.stage32)
                    + eva::constant_id<SPEC_ID_TYPE_O>(Dtype::Fp32)
                    + eva::constant_id<SPEC_ID_TYPE_I>(i16 ? Dtype::Fp16 : Dtype::Fp32)
                    + eva::constant_id<SPEC_ID_TYPE_W>(w32 ? Dtype::Fp32 : Dtype::Fp16)
                    + eva::constant_id<SPEC_ID_ALIGNED>(alignedK),
                .requiredSubgroupSize = c.reqSg,
                .captureStatistics = dumpStats || dumpIsa,
            });
            // clang-format on
            // linear-tile-10 split-K: K spans of a KC multiple, partials P[split, M, N] in fp32,
            // then split-reduce.comp folds them into O.
            // opt 0: one split, no reduce pass; opt s > 0: s splits through the reduce pass.
            uint32_t splitSize = sh.K, splits = 1;
            const uint32_t splitReq = (c.shader == 10 || c.shader == 11) ? std::max(c.opt, 1u)
                                    : (c.shader == 21) ? std::max(c.sgN, 1u) : 1;
            if (splitReq > 1)
            {
                const uint32_t align = (c.shader == 21) ? 16u : c.wgK;   // linear-tile-21: whole 16-k trips
                splitSize = CEIL_DIV(CEIL_DIV(sh.K, splitReq), align) * align;
                splits = CEIL_DIV(sh.K, splitSize);
            }
            const bool viaReduce = splits > 1 || ((c.shader == 10 || c.shader == 11) && c.opt);
            eva::Buffer bufP;
            eva::DescriptorSet reduceSet;
            if (viaReduce)
            {
                bufP = sharedP;   // splits <= the requested split <= maxSplitReq
                reduceSet = gVaiDestSetPool(reducePipe.descSetLayout(0));
                reduceSet.write({ bufO, bufP });
            }

            eva::DescriptorSet ds = gVaiDestSetPool(pipe.descSetLayout(0));
            // linear-tile-11-probe: counters + 8 words per tile and split on binding 3
            const uint32_t probeWords = 2u + 8u * CEIL_DIV(sh.M, c.wgM) * CEIL_DIV(sh.N, c.wgN) * splits;
            eva::Buffer bufProbe = c.probe11 ? deviceBuffer(dev, uint64_t(probeWords) * 4) : eva::Buffer();
            if (c.probe11)           ds.write({ viaReduce ? bufP : bufO, bufI, bufW, bufProbe });
            else if (c.shader == 21) ds.write({ viaReduce ? bufP : bufO, bufI, (c.opt & 4u) && !c.ov21 ? bufW
                                                                           : (c.opt & 8u) && !c.ov21 ? reorderedW(1, 2) : reorderedW(c.tN, c.wgK) });
            else                     ds.write({ viaReduce ? bufP : bufO, bufI, bufW });

            struct { uint32_t M, N, K, splitSize; } pc = { sh.M, sh.N, sh.K, splitSize };
            struct { uint32_t MN, splits; } reducePc = { sh.M * sh.N, splits };
            uint32_t gx = CEIL_DIV(sh.N, c.wgN), gy = CEIL_DIV(sh.M, c.wgM);
            if (c.shader == 5 && c.opt) std::swap(gx, gy);   // linear-tile-5 mFastest: x walks M tiles
            if (c.shader == 6) { gx *= gy; gy = 1; }          // linear-tile-6: one flat axis, grouped order
            if (c.shader >= 7 && c.shader <= 11 && !(c.shader == 11 && c.sgpNIter))
                std::swap(gx, gy);                              // linear-tile-7 ~ 11: x walks M tiles (11 with order 1: N tiles)
            if (c.probe11 && c.sgpNIter >= 2)                   // linear-tile-11-probe order 2 / 3: flat grid, N- / M-first
            {
                gx = gx * gy;                                   // gx = N tiles, gy = M tiles here
                gy = 1;
            }
            if (c.shader == 21)                                 // linear-tile-21: flat tile index x * gy + y, folded past the x limit
            {
                const uint32_t total = CEIL_DIV(CEIL_DIV(sh.N, c.wgN), c.sgpMIter) * c.sgpMIter
                                     * CEIL_DIV(CEIL_DIV(sh.M, c.wgM), c.sgpNIter) * c.sgpNIter;
                gy = (total <= dev.maxComputeWorkGroupCount()[0])
                   ? 1u : CEIL_DIV(total, CEIL_DIV(total, dev.maxComputeWorkGroupCount()[1]));
                gx = CEIL_DIV(total, gy);
            }
            if (c.swizzle1)                                     // linear-tile-1-swizzle: groups of opt N tiles
            {
                gy = CEIL_DIV(gx, c.opt);                       // gx = N tiles here
                gx = CEIL_DIV(sh.M, c.wgM) * c.opt;
            }
            // clang-format off
            auto record = [&cmd, &reducePipe, &bufO, &sh, pipe, ds, pc, reducePc, gx, gy, splits, viaReduce, bufP, reduceSet]
            {
                if (!viaReduce)
                {
                    cmd.bindPipeline(pipe).bindDescSets({ds})
                        .setPushConstants(0, sizeof(pc), &pc)
                        .dispatch(gx, gy, 1)
                        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / bufO / eva::SYNC_SCOPE::COMPUTE_WRITE);
                    return;
                }
                cmd.bindPipeline(pipe).bindDescSets({ds})
                    .setPushConstants(0, sizeof(pc), &pc)
                    .dispatch(gx, gy, splits)
                    .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / bufP / eva::SYNC_SCOPE::COMPUTE_READ)
                    .bindPipeline(reducePipe).bindDescSets({reduceSet})
                    .setPushConstants(0, sizeof(reducePc), &reducePc)
                    .dispatch(CEIL_DIV(sh.M * sh.N, 1024u), 1, 1)   // split-reduce.comp: four elements per invocation
                    .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / bufO / eva::SYNC_SCOPE::COMPUTE_WRITE);
            };
            // clang-format on

            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT);
            record();
            cmd.end().submit().wait();

            // About half a second of GPU work at the baseline's speed.
            const double clkBefore = clk ? clockProxy() : 0.0;
            const uint32_t reps = (quick || passes) ? 1u : std::clamp(uint32_t(500.0 / (gflop / 1.1)), 3u, 50u);
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
            cmd.writeTimestamp(stage, qp, 0);
            for (uint32_t r = 0; r < reps; ++r) record();
            cmd.writeTimestamp(stage, qp, 1);
            cmd.end().submit().wait();
            const double ms = qp.getElapsedMs(0, 1) / reps;
            const double clkAfter = clk ? clockProxy() : 0.0;

            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
                .copyBuffer(bufO(0, oBytes), readback(0, oBytes))
            .end().submit().wait();
            const float* got = (const float*)readback.map();
            double maxErr = 0, maxRef = 0;
            for (size_t r = 0; r < std::size(rows); ++r)
                for (uint32_t n = 0; n < sh.N; ++n)
                {
                    const double rv = ref[r * sh.N + n];
                    maxErr = std::max(maxErr, std::abs(rv - got[size_t(rows[r]) * sh.N + n]));
                    maxRef = std::max(maxRef, std::abs(rv));
                }

            printf("  %-58s  %8.3f ms  %5.2f TFLOP/s  maxRel %.1e%s\n", c.str().c_str(), ms, gflop / ms,
                   maxErr / maxRef, dumpStats ? statSummary(pipe).c_str() : "");
            if (clk)
                printf("    clock proxy (fp16 FMA TFLOP/s) before %.2f  after %.2f\n", clkBefore, clkAfter);

            // linear-tile-11-probe: zero the counters, run once more, write m,n,start,ended-before,end per tile.
            if (c.probe11 && !probeDir.empty())
            {
                const uint64_t bytes = uint64_t(probeWords) * 4;
                eva::Buffer stage = hostBuffer(dev, bytes);
                std::memset(stage.map(), 0, bytes);
                cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
                    .copyBuffer(stage(0, bytes), bufProbe(0, bytes))
                    .barrier(eva::SYNC_SCOPE::TRANSFER_DST / bufProbe / eva::SYNC_SCOPE::COMPUTE_READ);
                record();
                cmd.copyBuffer(bufProbe(0, bytes), stage(0, bytes));
                cmd.end().submit().wait();
                const uint32_t* w = (const uint32_t*)stage.map();
                const uint32_t tilesN = CEIL_DIV(sh.N, c.wgN), tiles = (probeWords - 2) / 8;
                const std::string path = probeDir + "/" + std::to_string(sh.M) + "x" + std::to_string(sh.K) + "x"
                                       + std::to_string(sh.N) + "_order" + std::to_string(c.sgpNIter)
                                       + "_split" + std::to_string(splits) + ".csv";
                FILE* f = std::fopen(path.c_str(), "w");
                const uint32_t tilesMN = tilesN * CEIL_DIV(sh.M, c.wgM);
                std::fprintf(f, "z,m,n,start,ended_before,end,staging,compute,whole,t_first,t_end\n");
                for (uint32_t t = 0; t < tiles; ++t)
                    std::fprintf(f, "%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n", t / tilesMN, t % tilesMN / tilesN, t % tilesN, w[2 + 8 * t], w[3 + 8 * t],
                                 w[4 + 8 * t], w[5 + 8 * t], w[6 + 8 * t], w[7 + 8 * t], w[8 + 8 * t], w[9 + 8 * t]);
                std::fclose(f);
            }
            if (dumpIsa) printIsa(pipe);
            if (passes) visits.push_back({ c.str(), record, ms });
        }

        if (passes && !visits.empty())
        {
            std::vector<double> t;
            for (const Visit& v : visits) t.push_back(v.ms);
            std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
            // -flush: each run starts with W and I out of the GPU L2 (4 MB) and is timed on its own,
            // so fewer runs per visit.
            constexpr uint32_t maxFlushRuns = 16;
            const uint32_t R = flush ? std::clamp(uint32_t(std::ceil(10.0 / std::max(t[t.size() / 2], 1e-4))), 3u, maxFlushRuns)
                                     : std::max(3u, uint32_t(std::ceil(30.0 / std::max(t[t.size() / 2], 1e-4))));
            constexpr uint64_t flushBytes = uint64_t(32) << 20;
            eva::Buffer flushA = flush ? deviceBuffer(dev, flushBytes) : eva::Buffer();
            eva::Buffer flushB = flush ? deviceBuffer(dev, flushBytes) : eva::Buffer();
            eva::QueryPool qpF = flush ? dev.createTimestampQueryPool(2 * maxFlushRuns) : eva::QueryPool();
            printf("  passes %u, %u runs per visit\n", passes, R);
            auto epochMs = [] {
                return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count();
            };
            std::vector<size_t> order(visits.size());
            std::mt19937 shuffleRng(sh.M * 131u + sh.K * 7u + sh.N);
            for (uint32_t p = 0; p < passes; ++p)
            {
                for (size_t i = 0; i < order.size(); ++i) order[i] = i;
                if (p == 1) std::reverse(order.begin(), order.end());
                if (p >= 2) std::shuffle(order.begin(), order.end(), shuffleRng);
                for (size_t i : order)
                {
                    const double t0 = epochMs();
                    double ms = 0;
                    if (flush)
                    {
                        cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qpF, 0, 2 * R);
                        for (uint32_t r = 0; r < R; ++r)
                        {
                            cmd.copyBuffer(flushA(0, flushBytes), flushB(0, flushBytes))
                               .barrier(eva::SYNC_SCOPE::TRANSFER_DST / flushB / eva::SYNC_SCOPE::COMPUTE_READ);
                            cmd.writeTimestamp(stage, qpF, 2 * r);
                            visits[i].record();
                            cmd.writeTimestamp(stage, qpF, 2 * r + 1);
                        }
                        cmd.end().submit().wait();
                        for (uint32_t r = 0; r < R; ++r) ms += qpF.getElapsedMs(2 * r, 2 * r + 1);
                        ms /= R;
                    }
                    else
                    {
                        cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
                        cmd.writeTimestamp(stage, qp, 0);
                        for (uint32_t r = 0; r < R; ++r) visits[i].record();
                        cmd.writeTimestamp(stage, qp, 1);
                        cmd.end().submit().wait();
                        ms = qp.getElapsedMs(0, 1) / R;
                    }
                    const double t1 = epochMs();
                    printf("  pass %u  %-58s  %9.4f ms  at %.1f %.1f\n", p, visits[i].name.c_str(), ms, t0, t1);
                }
            }
        }
    }
    return 0;
}
