// Every Linear pi0.5 runs per inference, one at a time, with the demo's dtypes
// (fp16 weights, fp32 activations) and each projection's bias / epilogue /
// residual. Prints GPU time per call, TFLOP/s, weight read rate, and the time
// one inference spends in that shape (calls x time). The condition table is
// built once at load and is not here.
//
// "split" sweeps the action expert's projections over explicit split counts
// (splitSize = K / split) next to the automatic choice, and checks each result
// against a CPU reference.
//
// "verify" checks Linear against a double-precision reference across M (1 up),
// K and N off the block sizes, bias / GeluTanh / residual, fp16 and fp32 W and
// accumulation.
//
//   test-linear-pi05
//   test-linear-pi05 split
//   test-linear-pi05 verify

#include "vai/nodes/linear.h"
#include "eva-runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace vai;
using namespace vai::node;


struct LinearModel : public NeuralNet
{
    using NeuralNet::dtypeKey;   // the tests set dtypes from outside
    Linear node;

    LinearModel(Linear::Config config, NeuralNetConfig netConfig)
    : NeuralNet(netConfig), node(config)
    {
        input("in0") - node - output("out0");
        if (config.acceptResidual)
            input("residual") - "residual"/node;
    }
};


struct Shape
{
    const char* tower;
    const char* what;
    uint32_t M, K, N;
    uint32_t calls;          // per inference: layers x per-layer count (x steps)
    bool bias = false;
    Epilogue epilogue = Epilogue::None;
    bool residual = false;
};


// One action-expert projection at one split: GPU ms per call and the max error
// relative to the largest reference value. splitSize 0 is the automatic choice.
struct SplitResult { double ms, maxRel; uint32_t splitCount; };

SplitResult runSplit(uint32_t M, uint32_t K, uint32_t N, uint32_t splitSize,
                     const std::vector<float>& x, const std::vector<float>& w, const std::vector<double>& ref)
{
    LinearModel model({ .inDim = K, .outDim = N,
                        .splitSize = splitSize },
                      { .wDefaultDtype = Dtype::Fp16 });
    model.node.setODtype(Dtype::Fp32, model.dtypeKey());
    model.node.setProfileTag("lin");
    model.node["weight"] = Tensor({N, K}, Dtype::Fp16).setFloats(w);
    Tensor in = Tensor({M, K}, Dtype::Fp32).setFloats(x);
    Tensor out;
    auto run = [&]() { model({{"in0", in}}, {{"out0", &out}}); out.getData(); };

    for (int i = 0; i < 3; ++i) run();
    model.resetProfileRecords();
    for (int i = 0; i < 300; ++i) run();
    const double ms = model.getGpuTimeMs("lin");

    const float* got = (const float*)out.getData();
    double maxErr = 0, maxRef = 0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        maxErr = std::max(maxErr, std::abs(ref[i] - got[i]));
        maxRef = std::max(maxRef, std::abs(ref[i]));
    }
    const uint32_t splitCount = splitSize > 0 ? CEIL_DIV(K, splitSize)
        : ggmlGuessSplitCount(M, N, K, ggmlMulmmTileParams(M, N, K, Dtype::Fp32, Dtype::Fp16, getVaiDevice()),
                              getVaiDevice());
    return { ms, maxErr / maxRef, splitCount };
}


int splitSweep()
{
    struct Proj { const char* what; uint32_t K, N; };
    const std::vector<Proj> projs = {
        { "q",       1024, 2048 },
        { "k / v",   1024,  256 },
        { "o",       2048, 1024 },
        { "gate+up", 1024, 8192 },
        { "down",    4096, 1024 },
    };
    const uint32_t M = 50;
    const uint32_t splits[] = { 1, 2, 4, 8, 16 };

    printf("%-8s %5s %5s |", "linear", "K", "N");
    for (uint32_t s : splits) printf(" %15s", ("split " + std::to_string(s)).c_str());
    printf(" | %20s\n", "auto (split)");

    for (const Proj& p : projs)
    {
        std::mt19937 rng(p.K * 7u + p.N);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        std::vector<float> w(size_t(p.N) * p.K), x(size_t(M) * p.K);
        for (float& v : w) v = uni(rng) * 0.05f;
        for (float& v : x) v = uni(rng);
        std::vector<double> ref(size_t(M) * p.N);
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t n = 0; n < p.N; ++n)
            {
                double acc = 0;
                for (uint32_t k = 0; k < p.K; ++k)
                    acc += double(x[size_t(m) * p.K + k]) * w[size_t(n) * p.K + k];
                ref[size_t(m) * p.N + n] = acc;
            }

        const double gflop = 2.0 * M * p.K * p.N / 1e9;
        std::string errs;
        printf("%-8s %5u %5u |", p.what, p.K, p.N);
        for (uint32_t s : splits)
        {
            const SplitResult r = runSplit(M, p.K, p.N, p.K / s, x, w, ref);
            printf("   %6.3f (%4.2f)", r.ms, gflop / r.ms);
            char e[32]; snprintf(e, sizeof(e), " %15.1e", r.maxRel); errs += e;
        }
        const SplitResult a = runSplit(M, p.K, p.N, 0, x, w, ref);
        printf(" |  %6.3f (%4.2f) [%2u]\n", a.ms, gflop / a.ms, a.splitCount);
        char e[32]; snprintf(e, sizeof(e), " %15.1e", a.maxRel);
        printf("%-8s %11s |%s | %20s\n", "", "maxRel", errs.c_str(), e);
    }
    return 0;
}


// W and bias are multiples of 1/1024, exact in fp16, so the reference takes
// them as they are.
int verify()
{
    struct Case { uint32_t M, K, N; bool bias; Epilogue epi; bool residual; Dtype wDtype, accDtype; };
    const Case cases[] = {
        {   1,   768,   512, false, Epilogue::None,     false, Dtype::Fp16, Dtype::Fp16 },
        {   1,  4305,    77, true,  Epilogue::GeluTanh, true,  Dtype::Fp16, Dtype::Fp16 },
        {   1,   588,  1000, true,  Epilogue::None,     false, Dtype::Fp32, Dtype::Fp32 },
        {   1, 16384,  2048, false, Epilogue::None,     true,  Dtype::Fp16, Dtype::Fp16 },
        {   1,  2048,   256, false, Epilogue::None,     false, Dtype::Fp16, Dtype::Fp16 },
        {   2,  4305,    77, true,  Epilogue::GeluTanh, true,  Dtype::Fp16, Dtype::Fp16 },
        {   4,   588,  1000, true,  Epilogue::None,     false, Dtype::Fp32, Dtype::Fp32 },
        {   4, 16384,  2048, false, Epilogue::None,     true,  Dtype::Fp16, Dtype::Fp16 },
        {   5,  2048,   500, true,  Epilogue::None,     false, Dtype::Fp16, Dtype::Fp16 },
        {   3,   770,   500, true,  Epilogue::None,     false, Dtype::Fp16, Dtype::Fp16 },
        {   8,     6,   512, false, Epilogue::None,     true,  Dtype::Fp16, Dtype::Fp32 },
        {   9,  1152,  4304, true,  Epilogue::GeluTanh, false, Dtype::Fp16, Dtype::Fp16 },
        {  50,  1024,   256, false, Epilogue::None,     true,  Dtype::Fp16, Dtype::Fp16 },
        {  50,    32,  1024, true,  Epilogue::None,     false, Dtype::Fp16, Dtype::Fp32 },
        {  37,  4305,    77, false, Epilogue::None,     false, Dtype::Fp16, Dtype::Fp16 },
        { 100,   588,  1000, true,  Epilogue::None,     true,  Dtype::Fp32, Dtype::Fp32 },
        { 922,  2048,  2048, false, Epilogue::None,     false, Dtype::Fp16, Dtype::Fp32 },
        { 922,  2048,  2048, false, Epilogue::None,     true,  Dtype::Fp16, Dtype::Fp16 },
        { 768,  1152,  4304, true,  Epilogue::GeluTanh, false, Dtype::Fp16, Dtype::Fp16 },
    };
    int fails = 0;
    for (const Case& c : cases)
    {
        std::mt19937 rng(c.M * 131u + c.K * 7u + c.N);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        auto grid = [&](float scale) { return std::round(uni(rng) * scale) / 1024.0f; };
        std::vector<float> w(size_t(c.N) * c.K), x(size_t(c.M) * c.K), b(c.N), r(size_t(c.M) * c.N);
        for (float& v : w) v = grid(48.0f);
        for (float& v : x) v = uni(rng);
        for (float& v : b) v = grid(512.0f);
        for (float& v : r) v = uni(rng);

        LinearModel model({ .inDim = c.K, .outDim = c.N, .acceptResidual = c.residual,
                            .epilogue = c.epi },
                          { .wDefaultDtype = c.wDtype });
        model.node.setODtype(Dtype::Fp32, model.dtypeKey());
        model.node.setAccDtype(c.accDtype, model.dtypeKey());
        model.node["weight"] = Tensor({c.N, c.K}, c.wDtype).setFloats(w);
        if (c.bias) model.node["bias"] = Tensor({c.N}, c.wDtype).setFloats(b);
        Tensor in = Tensor({c.M, c.K}, Dtype::Fp32).setFloats(x), res = Tensor({c.M, c.N}, Dtype::Fp32).setFloats(r), out;
        if (c.residual) model({{"in0", in}, {"residual", res}}, {{"out0", &out}});
        else            model({{"in0", in}}, {{"out0", &out}});
        const float* got = (const float*)out.getData();

        double maxErr = 0, maxRef = 0;
        for (uint32_t m = 0; m < c.M; ++m)
            for (uint32_t n = 0; n < c.N; ++n)
            {
                double v = c.bias ? b[n] : 0.0;
                for (uint32_t k = 0; k < c.K; ++k) v += double(x[size_t(m) * c.K + k]) * w[size_t(n) * c.K + k];
                if (c.epi == Epilogue::GeluTanh)
                    v = 0.5 * v * (1.0 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
                if (c.residual) v += r[size_t(m) * c.N + n];
                maxErr = std::max(maxErr, std::abs(v - got[size_t(m) * c.N + n]));
                maxRef = std::max(maxRef, std::abs(v));
            }
        const double rel = maxErr / maxRef, tol = (c.accDtype == Dtype::Fp16) ? 2e-2 : 1e-4;
        fails += rel > tol;
        printf("%5u x %5u x %5u  bias %d  epi %u  res %d  W %s  acc %s   maxRel %.1e  %s\n", c.M, c.K, c.N, c.bias,
               (uint32_t) c.epi, c.residual, c.wDtype == Dtype::Fp16 ? "f16" : "f32", c.accDtype == Dtype::Fp16 ? "f16" : "f32",
               rel, rel > tol ? "FAIL" : "ok");
    }
    printf("%s\n", fails ? "FAILED" : "all ok");
    return fails ? 1 : 0;
}


int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    getVaiDevice();


    if (argc > 1 && std::strcmp(argv[1], "split") == 0)
        return splitSweep();
    if (argc > 1 && std::strcmp(argv[1], "verify") == 0)
        return verify();

    // 3 cameras x 256 patches, a 154-token prompt (768 + 154 = 922), a 50-action chunk, 10 steps
    const std::vector<Shape> shapes = {
        { "vision",   "patch embed",   768,   588,  1152,   1, true },
        { "vision",   "q / k / v",     768,  1152,  1152,  81, true },                       // 27 x 3
        { "vision",   "o",             768,  1152,  1152,  27, true, Epilogue::None, true },
        { "vision",   "fc1",           768,  1152,  4304,  27, true, Epilogue::GeluTanh },
        { "vision",   "fc2",           768,  4304,  1152,  27, true, Epilogue::None, true },
        { "vision",   "projector",     768,  1152,  2048,   1, true },
        { "language", "q",             922,  2048,  2048,  18 },
        { "language", "k / v",         922,  2048,   256,  36 },                             // 18 x 2
        { "language", "o",             922,  2048,  2048,  18, false, Epilogue::None, true },
        { "language", "gate+up",       922,  2048, 32768,  18 },
        { "language", "down",          922, 16384,  2048,  18, false, Epilogue::None, true },
        { "action",   "action proj",    50,    32,  1024,  10, true },
        { "action",   "q",              50,  1024,  2048, 180 },                             // 18 x 10
        { "action",   "k / v",          50,  1024,   256, 360 },                             // 18 x 2 x 10
        { "action",   "o",              50,  2048,  1024, 180 },
        { "action",   "gate+up",        50,  1024,  8192, 180 },
        { "action",   "down",           50,  4096,  1024, 180 },
        { "action",   "velocity head",  50,  1024,    32,  10, true, Epilogue::None, true },
    };

    printf("%-8s %-13s %4s %5s %5s %4s | %5s %9s %8s %8s %10s\n",
           "tower", "linear", "M", "K", "N", "+", "calls", "ms/call", "TFLOP/s", "W GB/s", "ms/infer");

    std::map<std::string, double> towerMs;
    std::vector<std::string> towers;
    for (const Shape& s : shapes)
    {
        LinearModel model({ .inDim = s.K, .outDim = s.N, .acceptResidual = s.residual,
                            .epilogue = s.epilogue },
                          { .wDefaultDtype = Dtype::Fp16 });
        model.node.setODtype(Dtype::Fp32, model.dtypeKey());
        model.node.setProfileTag("lin");
        model.node["weight"] = Tensor({s.N, s.K}, Dtype::Fp16)
            .setFloats(std::vector<float>(size_t(s.N) * s.K, 0.01f));
        if (s.bias)
            model.node["bias"] = Tensor({s.N}, Dtype::Fp16).setFloats(std::vector<float>(s.N, 0.1f));

        Tensor in = Tensor({s.M, s.K}, Dtype::Fp32).setFloats(std::vector<float>(size_t(s.M) * s.K, 0.5f));
        Tensor res = Tensor({s.M, s.N}, Dtype::Fp32).setFloats(std::vector<float>(size_t(s.M) * s.N, 0.5f));
        Tensor out;
        auto run = [&]() {
            if (s.residual) model({{"in0", in}, {"residual", res}}, {{"out0", &out}});
            else            model({{"in0", in}}, {{"out0", &out}});
            out.getData();
        };

        // Warm up, then size the timed run to about 200 ms of GPU work.
        for (int i = 0; i < 3; ++i) run();
        model.resetProfileRecords();
        run();
        const int iters = std::clamp(int(200.0 / std::max(model.getGpuTimeMs("lin"), 1e-3)), 10, 500);
        model.resetProfileRecords();
        for (int i = 0; i < iters; ++i) run();

        const double ms = model.getGpuTimeMs("lin");
        const double tflops = 2.0 * s.M * s.K * s.N / (ms * 1e9);
        const double wGBs = 2.0 * s.N * s.K / (ms * 1e6);
        // clang-format off
        const char* flags = s.residual ? (s.epilogue != Epilogue::None ? "ber" : (s.bias ? "br" : "r"))
                          : s.epilogue != Epilogue::None ? "be" : s.bias ? "b" : "";
        // clang-format on
        printf("%-8s %-13s %4u %5u %5u %4s | %5u %9.3f %8.2f %8.1f %10.1f\n",
               s.tower, s.what, s.M, s.K, s.N, flags, s.calls, ms, tflops, wGBs, ms * s.calls);

        if (!towerMs.count(s.tower)) towers.push_back(s.tower);
        towerMs[s.tower] += ms * s.calls;
    }

    printf("\nms per inference:");
    for (const std::string& t : towers) printf("  %s %.1f", t.c_str(), towerMs[t]);
    printf("\n");
    return 0;
}
