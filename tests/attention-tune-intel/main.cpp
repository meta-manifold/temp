// Attention tuning harness for Intel GPUs without matrix engines: an
// attention-fa-intel*.comp variant at given tile parameters on pi0.5's three
// attention shapes (or -shape ones), checked against a double-precision host
// reference and timed with GPU timestamps (median of interleaved passes).
//
//   test-attention-tune-intel [-shape B,Hq,Hkv,Tq,Tk,Dh ...] [-passes P] [-nocheck] [config ...]
//
// config  variant,wgSize,G,TQ,BK[,TMS,TMP]: attention-fa-intel<variant>.comp ("" =
//         the base file) with G query heads per KV head and TQ tokens a workgroup,
//         BK keys a block; versions 2 to 5 also take the rows per subgroup of S
//         (TMS) and of PV (TMP); versions 6 and 7 need BK = 16 * wgSize / G / TQ.
//         G must equal Hq / Hkv. ATTN_STATS=1 prints every pipeline statistic,
//         not only the scratch (spill) size.
// node    node,wgSize,G,TQ,BK[,TMS,TMP,kCount]: vai's attention-fa-intel.comp bound as the
//         Attention node binds it (TQ and BK follow from wgSize and G). Keys past Tq come from
//         the cache; TMS = 1 caches a Tq == Tk shape too; kCount > 1 leaves the partials
//         unreduced, so only its time counts. ATTN_NODE_SPV points at another build of it.
//
// Q [B, Hq, Tq, Dh], K / V [B, Hkv, Tk, Dh] in fp16; O [B, Tq, Hq, Dh] in fp32.

#include "vai/nodes/common.h"
#include "eva-runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef ATTN_SHADER_DIR
#error "ATTN_SHADER_DIR must point at the compiled attention-fa-intel*.comp.spv files"
#endif

using namespace vai;

namespace {

struct Shape { std::string what; uint32_t B, Hq, Hkv, Tq, Tk, Dh; };
struct Config { std::string variant; uint32_t wgSize, G, TQ, BK, TMS = 8, TMP = 8, SKIP = 0; std::string str() const {
    return "v" + (variant.empty() ? std::string("1") : variant) + " wg" + std::to_string(wgSize) + " G" + std::to_string(G)
         + " TQ" + std::to_string(TQ) + " BK" + std::to_string(BK)
         + (variant.empty() ? std::string() : " S" + std::to_string(TMS) + " P" + std::to_string(TMP) + (SKIP ? " skip" + std::to_string(SKIP) : std::string())); } };

uint16_t toHalf(float f)
{
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = int32_t((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return uint16_t(sign);                    // inputs here never go subnormal
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    uint32_t h = sign | (uint32_t(e) << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1u))) h += 1;
    return uint16_t(h);
}

eva::Buffer deviceBuffer(eva::Device dev, uint64_t bytes)
{
    return dev.createBuffer({ .size = bytes,
        .usage = eva::BUFFER_USAGE::STORAGE_BUFFER | eva::BUFFER_USAGE::TRANSFER_SRC | eva::BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = eva::MEMORY_PROPERTY::DEVICE_LOCAL });
}

eva::Buffer hostBuffer(eva::Device dev, uint64_t bytes)
{
    return dev.createBuffer({ .size = bytes,
        .usage = eva::BUFFER_USAGE::TRANSFER_SRC | eva::BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT });
}

// Values on the fp16 grid in [-1, 1], so host and GPU see the same inputs.
std::vector<float> makeData(size_t n, uint32_t salt)
{
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i)
    {
        uint32_t h = uint32_t(i) * 2654435761u ^ salt * 97u; h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
        v[i] = float(h % 1025u) / 512.0f - 1.0f;
    }
    return v;
}

// O [B, Tq, Hq, Dh] in double.
std::vector<double> reference(const Shape& s, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v)
{
    std::vector<double> o(size_t(s.B) * s.Tq * s.Hq * s.Dh), p(s.Tk);
    const double scale = 1.0 / std::sqrt(double(s.Dh));
    const uint32_t G = s.Hq / s.Hkv;
    for (uint32_t b = 0; b < s.B; ++b)
        for (uint32_t h = 0; h < s.Hq; ++h)
            for (uint32_t t = 0; t < s.Tq; ++t)
            {
                const float* qr = &q[((size_t(b) * s.Hq + h) * s.Tq + t) * s.Dh];
                const size_t kvb = (size_t(b) * s.Hkv + h / G) * s.Tk;
                double mx = -1e300, sum = 0.0;
                for (uint32_t j = 0; j < s.Tk; ++j)
                {
                    const float* kr = &k[(kvb + j) * s.Dh]; double d = 0.0;
                    for (uint32_t x = 0; x < s.Dh; ++x) d += double(qr[x]) * kr[x];
                    p[j] = d * scale; mx = std::max(mx, p[j]);
                }
                for (uint32_t j = 0; j < s.Tk; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
                double* orow = &o[((size_t(b) * s.Tq + t) * s.Hq + h) * s.Dh];
                for (uint32_t j = 0; j < s.Tk; ++j)
                {
                    const float* vr = &v[(kvb + j) * s.Dh]; const double w = p[j] / sum;
                    for (uint32_t x = 0; x < s.Dh; ++x) orow[x] += w * vr[x];
                }
            }
    return o;
}

Config parse(const std::string& a)
{
    Config c{};
    const size_t comma = a.find(',');
    c.variant = a.substr(0, comma);
    if (std::sscanf(a.c_str() + comma + 1, "%u,%u,%u,%u,%u,%u,%u", &c.wgSize, &c.G, &c.TQ, &c.BK, &c.TMS, &c.TMP, &c.SKIP) < 4)
        { fprintf(stderr, "bad config: %s\n", a.c_str()); std::exit(1); }
    return c;
}

}   // namespace


int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    eva::Device dev = getVaiDevice();

    std::vector<Shape> shapes = {
        { "vision",  3, 16, 16, 256, 256,  72 },
        { "prefill", 1,  8,  1, 922, 922, 256 },
        { "denoise", 1,  8,  1,  50, 972, 256 },
    };
    std::vector<Shape> custom;
    std::vector<Config> configs;
    uint32_t passes = 5;
    bool check = true;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "-passes" && i + 1 < argc) passes = uint32_t(std::atoi(argv[++i]));
        else if (a == "-nocheck") check = false;
        else if (a == "-shape" && i + 1 < argc)
        {
            Shape s{ "custom" };
            if (std::sscanf(argv[++i], "%u,%u,%u,%u,%u,%u", &s.B, &s.Hq, &s.Hkv, &s.Tq, &s.Tk, &s.Dh) != 6)
                { fprintf(stderr, "bad shape: %s\n", argv[i]); return 1; }
            custom.push_back(s);
        }
        else configs.push_back(parse(a));
    }
    if (!custom.empty()) shapes = custom;
    if (configs.empty()) configs.push_back(parse(",128,8,2,32"));

    eva::CommandBuffer cmd = dev.newCommandBuffer(eva::queue_compute, eva::COMMAND_POOL_CREATE::RESET_COMMAND_BUFFER);
    eva::QueryPool qp = dev.createTimestampQueryPool(2);

    for (const Shape& s : shapes)
    {
        const size_t nQ = size_t(s.B) * s.Hq * s.Tq * s.Dh, nKV = size_t(s.B) * s.Hkv * s.Tk * s.Dh, nO = size_t(s.B) * s.Tq * s.Hq * s.Dh;
        std::vector<float> q = makeData(nQ, 1), k = makeData(nKV, 2), v = makeData(nKV, 3);
        // ATTN_QSCALE=2^n sharpens the softmax; a power of two keeps Q on the fp16 grid.
        if (const char* qs = getenv("ATTN_QSCALE")) for (float& x : q) x *= float(std::atof(qs));
        std::vector<uint16_t> qh(nQ), kh(nKV), vh(nKV);
        for (size_t i = 0; i < nQ; ++i)  qh[i] = toHalf(q[i]);
        for (size_t i = 0; i < nKV; ++i) { kh[i] = toHalf(k[i]); vh[i] = toHalf(v[i]); }
        eva::Buffer bQ = deviceBuffer(dev, nQ * 2), bK = deviceBuffer(dev, nKV * 2), bV = deviceBuffer(dev, nKV * 2), bO = deviceBuffer(dev, nO * 4);
        eva::Buffer stage = hostBuffer(dev, std::max(nQ * 2, std::max(nKV * 2, nO * 4)));
        auto upload = [&](eva::Buffer dst, const void* src, size_t bytes) {
            std::memcpy(stage.map(), src, bytes);
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).copyBuffer(stage(0, bytes), dst(0, bytes)).end().submit().wait();
        };
        upload(bQ, qh.data(), nQ * 2); upload(bK, kh.data(), nKV * 2); upload(bV, vh.data(), nKV * 2);
        // ATTN_DUMP=dir writes the fp16 inputs and each config's fp32 O, for comparisons outside the harness.
        const char* dumpDir = getenv("ATTN_DUMP");
        const std::string tag = std::to_string(s.B) + "_" + std::to_string(s.Hq) + "_" + std::to_string(s.Hkv) + "_"
                              + std::to_string(s.Tq) + "_" + std::to_string(s.Tk) + "_" + std::to_string(s.Dh);
        auto dump = [&](const std::string& name, const void* data, size_t bytes) {
            if (FILE* f = fopen((std::string(dumpDir) + "/" + tag + "_" + name + ".bin").c_str(), "wb")) { fwrite(data, 1, bytes, f); fclose(f); }
        };
        if (dumpDir) { dump("q", qh.data(), nQ * 2); dump("k", kh.data(), nKV * 2); dump("v", vh.data(), nKV * 2); }
        const std::vector<double> ref = check ? reference(s, q, k, v) : std::vector<double>();
        double refMax = 0.0; for (double x : ref) refMax = std::max(refMax, std::abs(x));
        const double gflop = 4.0 * s.B * s.Hq * s.Tq * s.Tk * s.Dh / 1e9;
        printf("\n%s  B %u  Hq %u  Hkv %u  Tq %u  Tk %u  Dh %u  (%.2f GFLOP)\n", s.what.c_str(), s.B, s.Hq, s.Hkv, s.Tq, s.Tk, s.Dh, gflop);

        struct Run { Config c; eva::ComputePipeline pipe; eva::DescriptorSet ds; uint32_t gx; std::vector<double> ms; eva::Buffer part, cnt; };
        // "node": the cache is bK / bV themselves, which the append rewrites with the same rows.
        const uint32_t cachedKeys = s.Tk - s.Tq;
        eva::Buffer kNew = bK, vNew = bV;
        if (cachedKeys > 0)
        {
            kNew = deviceBuffer(dev, size_t(s.B) * s.Hkv * s.Tq * s.Dh * 2);
            vNew = deviceBuffer(dev, size_t(s.B) * s.Hkv * s.Tq * s.Dh * 2);
            std::vector<uint16_t> kn, vn;
            for (uint32_t h = 0; h < s.B * s.Hkv; ++h)
            {
                kn.insert(kn.end(), kh.begin() + (size_t(h) * s.Tk + cachedKeys) * s.Dh, kh.begin() + (size_t(h) + 1) * s.Tk * s.Dh);
                vn.insert(vn.end(), vh.begin() + (size_t(h) * s.Tk + cachedKeys) * s.Dh, vh.begin() + (size_t(h) + 1) * s.Tk * s.Dh);
            }
            upload(kNew, kn.data(), kn.size() * 2);
            upload(vNew, vn.data(), vn.size() * 2);
        }
        auto uniformBuffer = [&](const void* data, size_t bytes) {
            eva::Buffer b = dev.createBuffer({ .size = bytes, .usage = eva::BUFFER_USAGE::UNIFORM_BUFFER,
                .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT });
            std::memcpy(b.map(), data, bytes);
            return b;
        };
        eva::Buffer counterUb = uniformBuffer(&cachedKeys, 4);
        uint32_t maskData[2 + 31] = { 1u, 0u };   // one non-causal block (common/causal_mask.h)
        eva::Buffer maskUb = uniformBuffer(maskData, sizeof(maskData));
        eva::Buffer dummy = deviceBuffer(dev, 256);
        std::vector<Run> runs;
        for (const Config& c : configs)
        {
            const uint32_t R = c.G * c.TQ, nsg = c.wgSize / 16, dh4 = s.Dh / 4;
            bool ok = (s.Hq / s.Hkv) % c.G == 0 && R % nsg == 0 && c.BK % 16 == 0 && c.wgSize % 16 == 0;
            size_t slm = 0;
            if (c.variant == "node")
            {
                if (c.G != s.Hq / s.Hkv || c.G > 16) { printf("  %-40s skipped: G\n", c.str().c_str()); continue; }
                const uint32_t caching = (cachedKeys > 0 || c.TMS == 1) ? 1u : 0u;
                Run r{ c };
                r.c.TQ = 16 / c.G;
                r.pipe = dev.createComputePipeline({
                    .csStage = eva::SpvBlob::readFrom(getenv("ATTN_NODE_SPV") ? getenv("ATTN_NODE_SPV") : ATTN_NODE_SPV)
                             + eva::constant_id<0>(c.wgSize) + eva::constant_id<1>(r.c.TQ) + eva::constant_id<3>(s.Dh)
                             + eva::constant_id<4>(0u) + eva::constant_id<10>(0u) + eva::constant_id<12>(0u)
                             + eva::constant_id<13>(1u) + eva::constant_id<14>(1u) + eva::constant_id<15>(caching)
                             + eva::constant_id<100>(0u),
                    .requiredSubgroupSize = 16,
                    .captureStatistics = true,
                });
                for (const eva::PipelineStatistic& st : r.pipe.statistics())
                    if (getenv("ATTN_STATS") || st.name.find("Scratch") != std::string::npos)
                        printf("  %-40s %s=%s\n", r.c.str().c_str(), st.name.c_str(), st.value.c_str());
                r.ds = gVaiDestSetPool(r.pipe.descSetLayout(0));
                {
                    // SKIP carries the node's split count; the partials stay unreduced (time only).
                    const size_t rows = size_t(s.B) * s.Tq * s.Hq, splits = std::max(1u, c.SKIP);
                    r.part = splits > 1 ? deviceBuffer(dev, rows * splits * s.Dh * 4) : dummy;
                    r.cnt = splits > 1 ? deviceBuffer(dev, rows * splits * 8) : dummy;
                    r.ds.write({ bO, bQ, kNew, vNew, bK, bV, counterUb, r.part, r.cnt, maskUb });
                }
                r.gx = (s.Tq + r.c.TQ - 1) / r.c.TQ * std::max(1u, c.SKIP);
                runs.push_back(r);
                continue;
            }
            if (c.variant.empty()) slm = (size_t(R) * dh4 + 2 * size_t(dh4) * c.BK) * 8 + size_t(R) * c.BK * 2;
            else if (c.variant == "6" || c.variant == "7" || c.variant == "8" || c.variant == "9" || c.variant == "10" || c.variant == "11" || c.variant == "12" || c.variant == "13" || c.variant == "14" || c.variant == "15")
            {
                const uint32_t rg = R / 16, ds = (s.Dh / 4 + 15) / 16;
                slm = size_t(R) * s.Dh * 2 + size_t(R) * (c.BK + 4) * 2 + size_t(R) * 4 + (rg ? size_t(nsg / rg) * R * 4 : 0);
                ok = (s.Hq / s.Hkv) % c.G == 0 && R % 16 == 0 && nsg % rg == 0 && c.BK == 16 * (nsg / rg)
                  && s.Dh % 4 == 0 && nsg % ds == 0 && R % (nsg / ds) == 0;
            }
            else
            {
                slm = size_t(dh4) * c.BK * 8 + size_t(R) * c.BK * 4 + size_t(R) * 8;
                if (c.variant != "2") slm += size_t(dh4) * c.BK * 8;   // V staged too
                ok = ok && R % c.TMS == 0 && R % c.TMP == 0 && nsg % (R / c.TMS) == 0 && nsg % (R / c.TMP) == 0
                        && c.BK % (16 * (nsg / (R / c.TMS))) == 0;
            }
            if (!ok || slm > dev.maxComputeSharedMemorySize())
                { printf("  %-40s skipped: %s\n", c.str().c_str(), ok ? ("shared memory " + std::to_string(slm)).c_str() : "tiling"); continue; }
            const std::string path = std::string(ATTN_SHADER_DIR) + "/attention-fa-intel" + c.variant + ".comp.spv";
            Run r{ c };
            r.pipe = dev.createComputePipeline({
                .csStage = eva::SpvBlob::readFrom(path.c_str()) + eva::constant_id<0>(c.wgSize) + eva::constant_id<1>(s.Dh)
                         + eva::constant_id<2>(c.G) + eva::constant_id<3>(c.TQ) + eva::constant_id<4>(c.BK)
                         + eva::constant_id<5>(c.TMS) + eva::constant_id<6>(c.TMP) + eva::constant_id<7>(c.SKIP),
                .requiredSubgroupSize = 16,
                .captureStatistics = true,
            });
            for (const eva::PipelineStatistic& st : r.pipe.statistics())
                if (getenv("ATTN_STATS") || st.name.find("Scratch") != std::string::npos)
                    printf("  %-40s %s=%s\n", c.str().c_str(), st.name.c_str(), st.value.c_str());
            r.ds = gVaiDestSetPool(r.pipe.descSetLayout(0));
            r.gx = (s.Tq + c.TQ - 1) / c.TQ;
            // clang-format off
            if (c.variant == "9" || c.variant == "10")   // split-K: TMS is the split count; the parts and the per-tile counts
            // clang-format on
            {
                const size_t rows = size_t(r.gx) * c.TMS * (s.Hq / c.G) * s.B * R;
                r.part = deviceBuffer(dev, rows * (dh4 + 1) * 16);
                r.cnt = deviceBuffer(dev, size_t(r.gx) * (s.Hq / c.G) * s.B * 4);
                const std::vector<uint32_t> zeros(size_t(r.gx) * (s.Hq / c.G) * s.B, 0u);
                upload(r.cnt, zeros.data(), zeros.size() * 4);
                r.ds.write({ bO, bQ, bK, bV, r.part, r.cnt });
                r.gx *= c.TMS;
            }
            else r.ds.write({ bO, bQ, bK, bV });
            runs.push_back(r);
        }
        auto record = [&](Run& r, uint32_t reps) {
            struct { uint32_t Hq, Hkv, Tq, Tk; float scale; } pc = { s.Hq, s.Hq / r.c.G, s.Tq, s.Tk, 1.0f / std::sqrt(float(s.Dh)) };
            // the node's push constants (Attention::run): Dh, Hq, groupSize, Tq, Tkv, Tmax, gqaRatio, kCount, scale
            struct { uint32_t Dh, Hq, groupSize, Tq, Tkv, Tmax, gqaRatio, kCount; float scale; } pcNode =
                { s.Dh, s.Hq, r.c.G, s.Tq, s.Tq, s.Tk, 0u, std::max(1u, r.c.SKIP), 1.0f / std::sqrt(float(s.Dh)) };
            cmd.bindPipeline(r.pipe).bindDescSets({ r.ds });
            if (r.c.variant == "node") cmd.setPushConstants(0, sizeof(pcNode), &pcNode);
            else cmd.setPushConstants(0, sizeof(pc), &pc);
            for (uint32_t i = 0; i < reps; ++i)
                cmd.dispatch(r.gx, s.Hq / r.c.G, s.B).barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / eva::SYNC_SCOPE::COMPUTE_WRITE)
                   .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / eva::SYNC_SCOPE::COMPUTE_READ);
        };
        for (Run& r : runs)
        {
            if (r.c.G != s.Hq / s.Hkv) { printf("  %-40s skipped: G must equal Hq / Hkv\n", r.c.str().c_str()); r.gx = 0; continue; }
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT); record(r, 1); cmd.end().submit().wait();
            if (check || dumpDir)
                cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).copyBuffer(bO(0, nO * 4), stage(0, nO * 4)).end().submit().wait();
            if (dumpDir)
            {
                std::string name = r.c.str();
                for (char& ch : name) if (ch == ' ') ch = '_';
                dump(name, stage.map(), nO * 4);
            }
            if (check)
            {
                const float* o = (const float*)stage.map(); double err = 0.0;
                for (size_t i = 0; i < nO; ++i) err = std::max(err, std::abs(double(o[i]) - ref[i]));
                printf("  %-40s max abs err %.2e (%.2e of max |O|)\n", r.c.str().c_str(), err, err / refMax);
            }
        }
        for (uint32_t p = 0; p < passes; ++p)
            for (Run& r : runs)
            {
                if (!r.gx) continue;
                const uint32_t reps = (gflop < 1.0) ? 10 : 2;
                cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
                cmd.writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, qp, 0);
                record(r, reps);
                cmd.writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, qp, 1);
                cmd.end().submit().wait();
                r.ms.push_back(qp.getElapsedMs(0, 1) / reps);
            }
        for (Run& r : runs)
        {
            if (r.ms.empty()) continue;
            std::sort(r.ms.begin(), r.ms.end());
            const double ms = r.ms[r.ms.size() / 2];
            printf("  %-40s %8.3f ms  %5.2f TFLOP/s\n", r.c.str().c_str(), ms, gflop / ms);
        }
    }
    return 0;
}
