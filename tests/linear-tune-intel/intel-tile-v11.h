// The Intel tile choice vai/nodes/linear.h made for linear-tile-11 (S8 / S15,
// tests/linear-tune-intel/6-structural-tile-heuristic.md) before Linear took
// linear-tile-21: the "11-node" / "11-node32" baseline of this harness.
#pragma once
#include "vai/nodes/linear.h"

namespace v11 {

using namespace vai;

// Intel scalar path: linear-tile-11.comp. Taken
// over the ggml scalar pick when the device can pin subgroup 16: for every
// tile-path M, either accumulation.
inline bool useIntelTile(uint32_t M, bool accF32, const node::Linear::PipeParamsTile& ggmlPick, eva::Device device)
{
    return device.vendorID() == eva::VENDOR_ID::INTEL && ggmlPick.tK == 0
        && device.minSubgroupSize() <= 16u && 16u <= device.maxSubgroupSize();
}

// Intel tile choice (tests/linear-tune-intel/6-structural-tile-heuristic.md):
// fp16 accumulation by S8, fp32 accumulation by S15. A candidate is the tile
// (wgSize / sgSize / sgn * tM) x (sgn * sgSize * tN) at SIMD sgSize, one per
// size from the pool measurement.
struct IntelTileCand
{
    uint32_t wgSize, sgSize, tM, tN, sgn;
};
constexpr IntelTileCand kIntelF16Acc[] = {          // fp16 accumulation, every pool size
    { 256, 16,  2,  2,  1 },                   //  32 x  32
    { 256, 16,  2,  3,  1 },                   //  32 x  48
    { 128, 16,  4,  4,  1 },                   //  32 x  64
    { 256, 16,  2,  6,  1 },                   //  32 x  96
    { 256, 16,  4,  4,  2 },                   //  32 x 128
    { 512, 16,  2,  6,  2 },                   //  32 x 192
    { 512, 16,  4,  4,  4 },                   //  32 x 256
    { 256, 16,  3,  2,  1 },                   //  48 x  32
    { 256, 16,  3,  3,  1 },                   //  48 x  48
    { 256, 16,  6,  2,  2 },                   //  48 x  64
    { 256, 16,  3,  6,  1 },                   //  48 x  96
    { 512, 16,  6,  2,  4 },                   //  48 x 128
    { 512, 16,  6,  3,  4 },                   //  48 x 192
    { 512, 16,  6,  4,  4 },                   //  48 x 256
    { 256, 16,  8,  1,  2 },                   //  64 x  32
    { 256, 16,  4,  3,  1 },                   //  64 x  48
    { 512, 16,  4,  2,  2 },                   //  64 x  64
    { 512, 16,  4,  3,  2 },                   //  64 x  96
    { 512, 16,  4,  4,  2 },                   //  64 x 128
    { 512, 16,  8,  3,  4 },                   //  64 x 192
    { 512, 16,  8,  4,  4 },                   //  64 x 256
    { 256, 16,  6,  2,  1 },                   //  96 x  32
    { 256, 16,  6,  3,  1 },                   //  96 x  48
    { 512, 16,  6,  2,  2 },                   //  96 x  64
    { 512, 16,  3,  6,  1 },                   //  96 x  96
    { 512, 16,  6,  4,  2 },                   //  96 x 128
    { 512, 16,  6,  6,  2 },                   //  96 x 192
    { 512, 16,  6,  8,  2 },                   //  96 x 256
    { 256, 16,  8,  2,  1 },                   // 128 x  32
    { 256, 16,  8,  3,  1 },                   // 128 x  48
    { 512, 16,  4,  4,  1 },                   // 128 x  64
    { 512, 16,  4,  6,  1 },                   // 128 x  96
    { 512, 16,  4,  8,  1 },                   // 128 x 128
    { 512, 16,  8,  6,  2 },                   // 128 x 192
    { 512, 16,  8,  8,  2 },                   // 128 x 256
    { 256, 16, 12,  2,  1 },                   // 192 x  32
    { 512, 16,  6,  3,  1 },                   // 192 x  48
    { 512, 16,  6,  4,  1 },                   // 192 x  64
    { 512, 16,  6,  6,  1 },                   // 192 x  96
    { 512, 16,  6,  8,  1 },                   // 192 x 128
    { 256, 16, 16,  2,  1 },                   // 256 x  32
    { 512, 16,  8,  3,  1 },                   // 256 x  48
    { 512, 16,  8,  4,  1 },                   // 256 x  64
    { 512, 16,  8,  6,  1 },                   // 256 x  96
    { 512, 16,  8,  8,  1 },                   // 256 x 128
};
// fp32 accumulation: wgSize <= 256.
constexpr IntelTileCand kIntelF32AccF16Stage[] = {   // fp16 staging
    { 128, 16,  4,  2,  1 },                   //  32 x  32
    { 256, 16,  2,  3,  1 },                   //  32 x  48
    { 128, 16,  8,  2,  2 },                   //  32 x  64
    { 256, 16,  4,  3,  2 },                   //  32 x  96
    { 256, 16,  8,  2,  4 },                   //  32 x 128
    { 128, 16,  6,  2,  1 },                   //  48 x  32
    { 256, 16,  3,  3,  1 },                   //  48 x  48
    { 256, 16,  6,  2,  2 },                   //  48 x  64
    { 128, 16,  8,  2,  1 },                   //  64 x  32
    { 256, 16,  4,  3,  1 },                   //  64 x  48
    { 256, 16,  8,  2,  2 },                   //  64 x  64
    { 256, 16,  6,  2,  1 },                   //  96 x  32
    { 256, 16,  8,  2,  1 },                   // 128 x  32
    { 128,  8,  4,  2,  2 },                   //  32 x  32
    { 128,  8,  4,  3,  2 },                   //  32 x  48
    { 128,  8,  4,  4,  2 },                   //  32 x  64
    { 128,  8,  8,  3,  4 },                   //  32 x  96
    { 128,  8,  8,  4,  4 },                   //  32 x 128
    { 256,  8,  4,  6,  4 },                   //  32 x 192
    { 256,  8,  8,  4,  8 },                   //  32 x 256
    { 128,  8,  3,  4,  1 },                   //  48 x  32
    { 128,  8,  3,  6,  1 },                   //  48 x  48
    { 128,  8,  6,  4,  2 },                   //  48 x  64
    { 128,  8, 12,  3,  4 },                   //  48 x  96
    { 256,  8,  6,  4,  4 },                   //  48 x 128
    { 256,  8, 12,  3,  8 },                   //  48 x 192
    { 256,  8,  3, 16,  2 },                   //  48 x 256
    { 128,  8,  4,  4,  1 },                   //  64 x  32
    { 128,  8,  4,  6,  1 },                   //  64 x  48
    { 128,  8,  8,  4,  2 },                   //  64 x  64
    { 256,  8,  8,  3,  4 },                   //  64 x  96
    { 256,  8,  8,  4,  4 },                   //  64 x 128
    { 256,  8,  4, 12,  2 },                   //  64 x 192
    { 256,  8,  4, 16,  2 },                   //  64 x 256
    { 128,  8,  6,  4,  1 },                   //  96 x  32
    { 128,  8, 12,  3,  2 },                   //  96 x  48
    { 256,  8,  6,  4,  2 },                   //  96 x  64
    { 256,  8, 12,  3,  4 },                   //  96 x  96
    { 256,  8, 12,  4,  4 },                   //  96 x 128
    { 256,  8,  6, 12,  2 },                   //  96 x 192
    { 128,  8,  8,  4,  1 },                   // 128 x  32
    { 256,  8,  8,  3,  2 },                   // 128 x  48
    { 256,  8,  8,  4,  2 },                   // 128 x  64
    { 256,  8,  4, 12,  1 },                   // 128 x  96
    { 256,  8,  4, 16,  1 },                   // 128 x 128
    { 256,  8,  6,  4,  1 },                   // 192 x  32
    { 256,  8, 12,  3,  2 },                   // 192 x  48
    { 256,  8, 24,  2,  4 },                   // 192 x  64
    { 256,  8,  6, 12,  1 },                   // 192 x  96
    { 256,  8,  8,  4,  1 },                   // 256 x  32
    { 256,  8, 16,  3,  2 },                   // 256 x  48
};
constexpr IntelTileCand kIntelF32AccF32Stage[] = {   // I and W fp32, fp32 staging
    { 128, 16,  8,  1,  2 },                   //  32 x  32
    { 128, 16,  4,  3,  1 },                   //  32 x  48
    { 128, 16,  8,  2,  2 },                   //  32 x  64
    { 256, 16,  4,  3,  2 },                   //  32 x  96
    { 256, 16,  8,  2,  4 },                   //  32 x 128
    { 128, 16,  6,  2,  1 },                   //  48 x  32
    { 128, 16,  6,  3,  1 },                   //  48 x  48
    { 256, 16, 12,  1,  4 },                   //  48 x  64
    { 256, 16,  6,  3,  2 },                   //  48 x  96
    { 256, 16, 24,  1,  8 },                   //  48 x 128
    { 128, 16,  8,  2,  1 },                   //  64 x  32
    { 256, 16,  4,  3,  1 },                   //  64 x  48
    { 256, 16,  8,  2,  2 },                   //  64 x  64
    { 256, 16,  6,  2,  1 },                   //  96 x  32
    { 256, 16,  6,  3,  1 },                   //  96 x  48
    { 256, 16, 24,  1,  4 },                   //  96 x  64
    { 256, 16,  8,  2,  1 },                   // 128 x  32
    { 128,  8,  4,  2,  2 },                   //  32 x  32
    { 128,  8,  4,  3,  2 },                   //  32 x  48
    { 128,  8,  8,  2,  4 },                   //  32 x  64
    { 128,  8,  8,  3,  4 },                   //  32 x  96
    { 128,  8,  8,  4,  4 },                   //  32 x 128
    { 256,  8,  8,  3,  8 },                   //  32 x 192
    { 256,  8,  8,  4,  8 },                   //  32 x 256
    { 128,  8,  6,  2,  2 },                   //  48 x  32
    { 128,  8,  6,  3,  2 },                   //  48 x  48
    { 128,  8, 12,  2,  4 },                   //  48 x  64
    { 128,  8, 12,  3,  4 },                   //  48 x  96
    { 256,  8,  6,  4,  4 },                   //  48 x 128
    { 256,  8, 12,  3,  8 },                   //  48 x 192
    { 256,  8, 24,  2, 16 },                   //  48 x 256
    { 128,  8,  8,  2,  2 },                   //  64 x  32
    { 128,  8,  8,  3,  2 },                   //  64 x  48
    { 128,  8,  8,  4,  2 },                   //  64 x  64
    { 128,  8, 16,  3,  4 },                   //  64 x  96
    { 256,  8,  8,  4,  4 },                   //  64 x 128
    { 256,  8, 16,  3,  8 },                   //  64 x 192
    { 128,  8,  6,  4,  1 },                   //  96 x  32
    { 128,  8, 12,  3,  2 },                   //  96 x  48
    { 256,  8, 12,  2,  4 },                   //  96 x  64
    { 256,  8, 12,  3,  4 },                   //  96 x  96
    { 256,  8, 24,  2,  8 },                   //  96 x 128
    { 128,  8,  8,  4,  1 },                   // 128 x  32
    { 128,  8, 16,  3,  2 },                   // 128 x  48
    { 256,  8,  8,  4,  2 },                   // 128 x  64
    { 256,  8, 16,  3,  4 },                   // 128 x  96
    { 256,  8, 12,  2,  2 },                   // 192 x  32
    { 256,  8, 12,  3,  2 },                   // 192 x  48
    { 256,  8, 24,  2,  4 },                   // 192 x  64
    { 256,  8,  8,  4,  1 },                   // 256 x  32
    { 256,  8, 16,  3,  2 },                   // 256 x  48
};

struct IntelTilePick
{
    node::Linear::PipeParamsTile tile;
    uint32_t splits;
};

// What both rules weigh a candidate by:
//   - splits: the fewest that fill the slots of the GPU once, and at least
//     K / k0, with k spans of at least 64;
//   - pad: the useful share of the padded output;
//   - fill = x / (0.5 + 0.5 x) for the share x of the slots the workgroups
//     fill (a lone wave still takes half a full one).
struct IntelTileFit
{
    uint32_t wgM, wgN, splits;
    float pad, fill;
};

inline IntelTileFit intelTileFit(const IntelTileCand& c, uint32_t M, uint32_t N, uint32_t K, float k0, eva::Device device)
{
    // Xe-cores; 8 (Meteor Lake H) for a part eva does not list.
    const uint32_t cores       = device.computeBlockCount() ? device.computeBlockCount() : 8u;
    const uint32_t coreThreads = 128;                                       // 16 EUs x 8 threads (Xe-LPG)
    const uint32_t wgM = c.wgSize / c.sgSize / c.sgn * c.tM, wgN = c.sgn * c.sgSize * c.tN;
    const uint32_t slots = cores * (coreThreads / (c.wgSize / c.sgSize));
    const uint32_t tiles = CEIL_DIV(M, wgM) * CEIL_DIV(N, wgN);
    const float want = std::max(float(slots) / float(tiles), float(K) / k0);
    uint32_t s = 1;
    while (float(s * 2) <= want && s < 16u) s *= 2;                         // power-of-two floor
    while (s > 1 && K / s < 64u) s /= 2;
    const float pad = float(M) * float(N) / (float(CEIL_DIV(M, wgM) * wgM) * float(CEIL_DIV(N, wgN) * wgN));
    const float x   = std::min(1.0f, float(tiles * s) / float(slots));
    return { wgM, wgN, s, pad, x / (0.5f + 0.5f * x) };
}

inline IntelTilePick intelTilePick(const IntelTileCand& c, uint32_t splits, uint32_t K, Dtype iDtype)
{
    // Whole k-blocks keep the 4-wide load path; otherwise drop to 4.
    const uint32_t wgK = (K % 8u == 0) ? 8u : 4u;
    const uint32_t wgN = c.sgn * c.sgSize * c.tN;
    // ( wgSize, sgSize, tN, tM, sgpNIter, sgpMIter, wgN, wgK, sgN )
    IntelTilePick pick = { node::Linear::PipeParamsTile(c.wgSize, c.sgSize, c.tN, c.tM, 1u, 1u, wgN, wgK, c.sgSize * c.tN), splits };
    pick.tile.iDtype = iDtype;
    return pick;
}

// fp16 accumulation (S8): the largest r x pad x fill with k0 = 2048, where
// r = 4 tM tN / (tM + tN) is a lane's FMAs per shared-memory load. Ties go to
// the fewer subgroups along N, then to the larger tile.
inline IntelTilePick intelTileParamsF16(uint32_t M, uint32_t N, uint32_t K, Dtype iDtype, eva::Device device)
{
    const IntelTileCand* best = nullptr;
    IntelTileFit bestFit = {};
    float bestScore = -1.0f;
    for (const IntelTileCand& c : kIntelF16Acc)
    {
        const IntelTileFit f = intelTileFit(c, M, N, K, 2048.0f, device);
        const float score = 4.0f * float(c.tM * c.tN) / float(c.tM + c.tN) * f.pad * f.fill;
        if (score > bestScore || (score == bestScore && (c.sgn < best->sgn
                                  || (c.sgn == best->sgn && f.wgM * f.wgN > bestFit.wgM * bestFit.wgN))))
            { best = &c; bestFit = f; bestScore = score; }
    }
    return intelTilePick(*best, bestFit.splits, K, iDtype);
}

// fp32 accumulation (S15), in two steps:
//   1. SIMD width: SIMD8 when 2 M N K reaches 2.8 GFLOP (fp16 staging) or 0.22
//      GFLOP (fp32 staging) and M >= 64, where its wider lane tiles pay; SIMD16
//      otherwise.
//   2. Tile: the largest e x pad x fill with k0 = 1024 (fp32 FMAs run at half
//      the fp16 rate), where e is a lane's FMAs per memory message in a
//      k-block: its shared-memory reads (a broadcast I slot, a per-lane W slot)
//      and its staging trips (a global load and a shared store each), every
//      access counted in 64 B lines. Ties go to the squarer tile.
inline IntelTilePick intelTileParamsF32(uint32_t M, uint32_t N, uint32_t K, bool stageF32, Dtype iDtype, Dtype wDtype,
                                        eva::Device device)
{
    const IntelTileCand* cands = stageF32 ? kIntelF32AccF32Stage : kIntelF32AccF16Stage;
    const size_t count = stageF32 ? std::size(kIntelF32AccF32Stage) : std::size(kIntelF32AccF16Stage);
    const double simd8Flop = stageF32 ? 0.22e9 : 2.8e9;
    const uint32_t sg = (2.0 * M * N * K >= simd8Flop && M >= 64u && device.minSubgroupSize() <= 8u) ? 8u : 16u;
    const uint32_t kc = (K % 8u == 0) ? 2u : 1u;                            // 4-element slots per staged row (wgK / 4)
    const uint32_t bS = stageF32 ? 4u : 2u, bI = uint32_t(bytesPerElement(iDtype)), bW = uint32_t(bytesPerElement(wDtype));
    const auto lines = [](uint32_t bytes) { return std::max(1.0f, float(bytes) / 64.0f); };   // one subgroup access

    const IntelTileCand* best = nullptr;
    IntelTileFit bestFit = {};
    float bestScore = -1.0f, bestSquare = 0.0f;
    for (size_t i = 0; i < count; i++)
    {
        const IntelTileCand& c = cands[i];
        if (c.sgSize != sg) continue;
        const IntelTileFit f = intelTileFit(c, M, N, K, 1024.0f, device);
        const float w     = lines(c.sgSize * 4u * bS);                                 // a per-lane W slot read
        const float reads = float(kc) * (float(c.tM) + w * float(c.tN));               // a broadcast I slot read is one line
        const float trips = float(CEIL_DIV(f.wgM * kc, c.wgSize)) * (lines(c.sgSize * 4u * bI) + w)
                          + float(CEIL_DIV(f.wgN * kc, c.wgSize)) * (lines(c.sgSize * 4u * bW) + w);
        const float e      = 4.0f * float(kc * c.tM * c.tN) / (reads + trips);
        const float score  = e * f.pad * f.fill;
        const float square = float(f.wgM * f.wgN) / float(f.wgM + f.wgN);
        if (score > bestScore || (score == bestScore && square > bestSquare))
            { best = &c; bestFit = f; bestScore = score; bestSquare = square; }
    }
    return intelTilePick(*best, bestFit.splits, K, iDtype);
}

inline IntelTilePick intelTileParams(uint32_t M, uint32_t N, uint32_t K, bool accF32, bool stageF32,
                                     Dtype iDtype, Dtype wDtype, eva::Device device)
{
    return accF32 ? intelTileParamsF32(M, N, K, stageF32, iDtype, wDtype, device)
                  : intelTileParamsF16(M, N, K, iDtype, device);
}

// Workgroup order (tests/linear-tune-intel/2-workgroup-order.md): tiles run
// next to each other share the operand a tile reads more bytes of per k.
// N-first shares I, M-first shares W; ties go M-first. Intel walks a 2D grid
// y-fastest, so N-first puts the M tiles on x.
inline bool intelTileNFirst(const node::Linear::PipeParamsTile& t, Dtype wDtype)
{
    return t.wgM * bytesPerElement(t.iDtype) > t.wgN * bytesPerElement(wDtype);
}

}   // namespace v11
