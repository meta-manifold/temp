// Concat's contiguous case (each input one run of the output) three ways:
// concat-1.comp (one thread per output element), concat-2.comp under
// contiguous=true, and buffer copies. Each is recorded the way Concat records
// it and timed over repeated runs; a run whose outputs differ is reported.
//
//   test-concat-contiguous

#include "vai/nodes/common.h"
#include "eva-runtime.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if !defined(CONCAT_1_SPV) || !defined(CONCAT_2_SPV)
#error "CONCAT_1_SPV / CONCAT_2_SPV must point at the compiled concat-1 / concat-2 .spv"
#endif

using namespace vai;

namespace {

constexpr uint32_t maxInputs = 8;

struct Case {
    std::vector<uint32_t> rows;   // each input's extent along axis 0
    uint32_t D;                   // the trailing extent
    Dtype dtype;
};

struct Constants {
    uint32_t totalElements, innerSize, concatDimOut, inputCount;
    uint32_t prefixSum[maxInputs];
};

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

}   // namespace


int main()
{
    eva::Device dev = getVaiDevice();
    eva::CommandBuffer cmd = dev.newCommandBuffer(eva::queue_compute, eva::COMMAND_POOL_CREATE::RESET_COMMAND_BUFFER);
    eva::QueryPool qp = dev.createTimestampQueryPool(6);
    const auto stage = eva::PIPELINE_STAGE::ALL_COMMANDS;

    const eva::SpvBlob spv1 = eva::SpvBlob::readFrom(CONCAT_1_SPV);
    const eva::SpvBlob spv2 = eva::SpvBlob::readFrom(CONCAT_2_SPV);

    const std::vector<Case> cases = {
        { {1, 50},                               1024, Dtype::Fp32 },   // pi0's state token + action tokens
        { {768, 17},                             2048, Dtype::Fp32 },   // three cameras' patches + prompt
        { {768, 17},                             2048, Dtype::Fp16 },
        { {4096, 4096},                          4096, Dtype::Fp32 },   // two large equal halves, bandwidth-bound
        { {4096, 4096},                          4096, Dtype::Fp16 },
        { std::vector<uint32_t>(8, 512),         1024, Dtype::Fp16 },   // the shader's full eight inputs
        { {3, 5},                                1023, Dtype::Fp32 },   // widths no vec2 / vec4 divides: scalar
        { {32, 16},                             25600, Dtype::Fp32 },   // YOLOv8n: backbone.p2.
        { {32, 16},                             25600, Dtype::Fp16 },
        { {64, 32, 32},                          6400, Dtype::Fp32 },   // YOLOv8n: backbone.p3.
        { {64, 32, 32},                          6400, Dtype::Fp16 },
        { {128, 64, 64},                         1600, Dtype::Fp32 },   // YOLOv8n: backbone.p4.
        { {128, 64, 64},                         1600, Dtype::Fp16 },
        { {256, 128},                             400, Dtype::Fp32 },   // YOLOv8n: backbone.p5Pre, neck.out_p5.
        { {256, 128},                             400, Dtype::Fp16 },
        { {128, 64},                             1600, Dtype::Fp32 },   // YOLOv8n: neck.temp_p4, neck.out_p4.
        { {128, 64},                             1600, Dtype::Fp16 },
        { {64, 32},                              6400, Dtype::Fp32 },   // YOLOv8n: neck.out_p3.
        { {64, 32},                              6400, Dtype::Fp16 },
        { {256, 128},                            1600, Dtype::Fp32 },   // YOLOv8n: neck.concat1. YOLOv8s: neck.temp_p4, neck.out_p4.
        { {256, 128},                            1600, Dtype::Fp16 },
        { {128, 64},                             6400, Dtype::Fp32 },   // YOLOv8n: neck.concat2. YOLOv8s: neck.out_p3.
        { {128, 64},                             6400, Dtype::Fp16 },
        { {64, 128},                             1600, Dtype::Fp32 },   // YOLOv8n: neck.concat3.
        { {64, 128},                             1600, Dtype::Fp16 },
        { {128, 256},                             400, Dtype::Fp32 },   // YOLOv8n: neck.concat4.
        { {128, 256},                             400, Dtype::Fp16 },
        { {64, 32},                             25600, Dtype::Fp32 },   // YOLOv8s: backbone.p2.
        { {64, 32},                             25600, Dtype::Fp16 },
        { {128, 64, 64},                         6400, Dtype::Fp32 },   // YOLOv8s: backbone.p3.
        { {128, 64, 64},                         6400, Dtype::Fp16 },
        { {256, 128, 128},                       1600, Dtype::Fp32 },   // YOLOv8s: backbone.p4.
        { {256, 128, 128},                       1600, Dtype::Fp16 },
        { {512, 256},                             400, Dtype::Fp32 },   // YOLOv8s: backbone.p5Pre, neck.out_p5.
        { {512, 256},                             400, Dtype::Fp16 },
        { {512, 256},                            1600, Dtype::Fp32 },   // YOLOv8s: neck.concat1.
        { {512, 256},                            1600, Dtype::Fp16 },
        { {256, 128},                            6400, Dtype::Fp32 },   // YOLOv8s: neck.concat2.
        { {256, 128},                            6400, Dtype::Fp16 },
        { {128, 256},                            1600, Dtype::Fp32 },   // YOLOv8s: neck.concat3.
        { {128, 256},                            1600, Dtype::Fp16 },
        { {256, 512},                             400, Dtype::Fp32 },   // YOLOv8s: neck.concat4.
        { {256, 512},                             400, Dtype::Fp16 },
        { {96, 48, 48},                         25600, Dtype::Fp32 },   // YOLOv8m: backbone.p2.
        { {96, 48, 48},                         25600, Dtype::Fp16 },
        { {192, 96, 96, 96, 96},                 6400, Dtype::Fp32 },   // YOLOv8m: backbone.p3.
        { {192, 96, 96, 96, 96},                 6400, Dtype::Fp16 },
        { {384, 192, 192, 192, 192},             1600, Dtype::Fp32 },   // YOLOv8m: backbone.p4.
        { {384, 192, 192, 192, 192},             1600, Dtype::Fp16 },
        { {576, 288, 288},                        400, Dtype::Fp32 },   // YOLOv8m: backbone.p5Pre, neck.out_p5.
        { {576, 288, 288},                        400, Dtype::Fp16 },
        { {384, 192, 192},                       1600, Dtype::Fp32 },   // YOLOv8m: neck.temp_p4, neck.out_p4.
        { {384, 192, 192},                       1600, Dtype::Fp16 },
        { {192, 96, 96},                         6400, Dtype::Fp32 },   // YOLOv8m: neck.out_p3.
        { {192, 96, 96},                         6400, Dtype::Fp16 },
        { {576, 384},                            1600, Dtype::Fp32 },   // YOLOv8m: neck.concat1.
        { {576, 384},                            1600, Dtype::Fp16 },
        { {384, 192},                            6400, Dtype::Fp32 },   // YOLOv8m: neck.concat2.
        { {384, 192},                            6400, Dtype::Fp16 },
        { {192, 384},                            1600, Dtype::Fp32 },   // YOLOv8m: neck.concat3.
        { {192, 384},                            1600, Dtype::Fp16 },
        { {384, 576},                             400, Dtype::Fp32 },   // YOLOv8m: neck.concat4.
        { {384, 576},                             400, Dtype::Fp16 },
        { {128, 64, 64, 64},                    25600, Dtype::Fp32 },   // YOLOv8l: backbone.p2.
        { {128, 64, 64, 64},                    25600, Dtype::Fp16 },
        { {256, 128, 128, 128, 128, 128, 128},   6400, Dtype::Fp32 },   // YOLOv8l: backbone.p3.
        { {256, 128, 128, 128, 128, 128, 128},   6400, Dtype::Fp16 },
        { {512, 256, 256, 256, 256, 256, 256},   1600, Dtype::Fp32 },   // YOLOv8l: backbone.p4.
        { {512, 256, 256, 256, 256, 256, 256},   1600, Dtype::Fp16 },
        { {512, 256, 256, 256},                   400, Dtype::Fp32 },   // YOLOv8l: backbone.p5Pre, neck.out_p5.
        { {512, 256, 256, 256},                   400, Dtype::Fp16 },
        { {512, 256, 256, 256},                  1600, Dtype::Fp32 },   // YOLOv8l: neck.temp_p4, neck.out_p4.
        { {512, 256, 256, 256},                  1600, Dtype::Fp16 },
        { {256, 128, 128, 128},                  6400, Dtype::Fp32 },   // YOLOv8l: neck.out_p3.
        { {256, 128, 128, 128},                  6400, Dtype::Fp16 },
        { {512, 512},                            1600, Dtype::Fp32 },   // YOLOv8l: neck.concat1. YOLOXl: neck.concatP4.
        { {512, 512},                            1600, Dtype::Fp16 },
        { {512, 256},                            6400, Dtype::Fp32 },   // YOLOv8l: neck.concat2.
        { {512, 256},                            6400, Dtype::Fp16 },
        { {256, 512},                            1600, Dtype::Fp32 },   // YOLOv8l: neck.concat3.
        { {256, 512},                            1600, Dtype::Fp16 },
        { {512, 512},                             400, Dtype::Fp32 },   // YOLOv8l: neck.concat4. YOLOXl: backbone.dark5.csp, neck.concatN4, neck.c3N4.
        { {512, 512},                             400, Dtype::Fp16 },
        { {160, 80, 80, 80},                    25600, Dtype::Fp32 },   // YOLOv8x: backbone.p2.
        { {160, 80, 80, 80},                    25600, Dtype::Fp16 },
        { {320, 160, 160, 160, 160, 160, 160},   6400, Dtype::Fp32 },   // YOLOv8x: backbone.p3.
        { {320, 160, 160, 160, 160, 160, 160},   6400, Dtype::Fp16 },
        { {640, 320, 320, 320, 320, 320, 320},   1600, Dtype::Fp32 },   // YOLOv8x: backbone.p4.
        { {640, 320, 320, 320, 320, 320, 320},   1600, Dtype::Fp16 },
        { {640, 320, 320, 320},                   400, Dtype::Fp32 },   // YOLOv8x: backbone.p5Pre, neck.out_p5.
        { {640, 320, 320, 320},                   400, Dtype::Fp16 },
        { {640, 320, 320, 320},                  1600, Dtype::Fp32 },   // YOLOv8x: neck.temp_p4, neck.out_p4.
        { {640, 320, 320, 320},                  1600, Dtype::Fp16 },
        { {320, 160, 160, 160},                  6400, Dtype::Fp32 },   // YOLOv8x: neck.out_p3.
        { {320, 160, 160, 160},                  6400, Dtype::Fp16 },
        { {640, 640},                            1600, Dtype::Fp32 },   // YOLOv8x: neck.concat1. YOLOXx: neck.concatP4.
        { {640, 640},                            1600, Dtype::Fp16 },
        { {640, 320},                            6400, Dtype::Fp32 },   // YOLOv8x: neck.concat2.
        { {640, 320},                            6400, Dtype::Fp16 },
        { {320, 640},                            1600, Dtype::Fp32 },   // YOLOv8x: neck.concat3.
        { {320, 640},                            1600, Dtype::Fp16 },
        { {640, 640},                             400, Dtype::Fp32 },   // YOLOv8x: neck.concat4. YOLOXx: backbone.dark5.csp, neck.concatN4, neck.c3N4.
        { {640, 640},                             400, Dtype::Fp16 },
        { {32, 32},                             25600, Dtype::Fp32 },   // YOLOXs: backbone.dark2.csp.
        { {32, 32},                             25600, Dtype::Fp16 },
        { {64, 64},                              6400, Dtype::Fp32 },   // YOLOXs: backbone.dark3.csp, neck.c3P3.
        { {64, 64},                              6400, Dtype::Fp16 },
        { {128, 128},                            1600, Dtype::Fp32 },   // YOLOXs: backbone.dark4.csp, neck.concatN3, neck.c3P4, neck.c3N3.
        { {128, 128},                            1600, Dtype::Fp16 },
        { {256, 256},                             400, Dtype::Fp32 },   // YOLOXs: backbone.dark5.csp, neck.concatN4, neck.c3N4. RT-DETRv2 S / RT-DETRv2 M / RT-DETRv2 L: encoder.panConcat1.
        { {256, 256},                             400, Dtype::Fp16 },
        { {256, 256},                            1600, Dtype::Fp32 },   // YOLOXs: neck.concatP4. YOLOXl: backbone.dark4.csp, neck.concatN3, neck.c3P4, neck.c3N3. RT-DETRv2 S / RT-DETRv2 M / RT-DETRv2 L: encoder.fpnConcat0, encoder.panConcat0.
        { {256, 256},                            1600, Dtype::Fp16 },
        { {128, 128},                            6400, Dtype::Fp32 },   // YOLOXs: neck.concatP3. YOLOXl: backbone.dark3.csp, neck.c3P3.
        { {128, 128},                            6400, Dtype::Fp16 },
        { {48, 48},                             25600, Dtype::Fp32 },   // YOLOXm: backbone.dark2.csp.
        { {48, 48},                             25600, Dtype::Fp16 },
        { {96, 96},                              6400, Dtype::Fp32 },   // YOLOXm: backbone.dark3.csp, neck.c3P3.
        { {96, 96},                              6400, Dtype::Fp16 },
        { {192, 192},                            1600, Dtype::Fp32 },   // YOLOXm: backbone.dark4.csp, neck.concatN3, neck.c3P4, neck.c3N3.
        { {192, 192},                            1600, Dtype::Fp16 },
        { {384, 384},                             400, Dtype::Fp32 },   // YOLOXm: backbone.dark5.csp, neck.concatN4, neck.c3N4. RT-DETRv2 X: encoder.panConcat1.
        { {384, 384},                             400, Dtype::Fp16 },
        { {384, 384},                            1600, Dtype::Fp32 },   // YOLOXm: neck.concatP4. RT-DETRv2 X: encoder.fpnConcat0, encoder.panConcat0.
        { {384, 384},                            1600, Dtype::Fp16 },
        { {192, 192},                            6400, Dtype::Fp32 },   // YOLOXm: neck.concatP3.
        { {192, 192},                            6400, Dtype::Fp16 },
        { {64, 64},                             25600, Dtype::Fp32 },   // YOLOXl: backbone.dark2.csp.
        { {64, 64},                             25600, Dtype::Fp16 },
        { {256, 256},                            6400, Dtype::Fp32 },   // YOLOXl: neck.concatP3. RT-DETRv2 S / RT-DETRv2 M / RT-DETRv2 L: encoder.fpnConcat1.
        { {256, 256},                            6400, Dtype::Fp16 },
        { {80, 80},                             25600, Dtype::Fp32 },   // YOLOXx: backbone.dark2.csp.
        { {80, 80},                             25600, Dtype::Fp16 },
        { {160, 160},                            6400, Dtype::Fp32 },   // YOLOXx: backbone.dark3.csp, neck.c3P3.
        { {160, 160},                            6400, Dtype::Fp16 },
        { {320, 320},                            1600, Dtype::Fp32 },   // YOLOXx: backbone.dark4.csp, neck.concatN3, neck.c3P4, neck.c3N3.
        { {320, 320},                            1600, Dtype::Fp16 },
        { {320, 320},                            6400, Dtype::Fp32 },   // YOLOXx: neck.concatP3.
        { {320, 320},                            6400, Dtype::Fp16 },
        { {6400, 1600, 400},                      256, Dtype::Fp32 },   // RT-DETRv2 S / RT-DETRv2 M / RT-DETRv2 L / RT-DETRv2 X: memory.
        { {6400, 1600, 400},                      256, Dtype::Fp16 },
        { {384, 384},                            6400, Dtype::Fp32 },   // RT-DETRv2 X: encoder.fpnConcat1.
        { {384, 384},                            6400, Dtype::Fp16 },
    };

    // pack is concat-2's elements per invocation; GB/s counts each output byte
    // read once and written once.
    printf("%-5s  %-36s %9s %5s | %21s | %21s | %21s\n",
           "dtype", "inputs", "MB", "pack", "concat-1 us (GB/s)", "concat-2 us (GB/s)", "copy us (GB/s)");

    for (const Case& c : cases)
    {
        const uint32_t n = uint32_t(c.rows.size());
        const uint32_t esz = uint32_t(bytesPerElement(c.dtype));
        uint32_t totalRows = 0;
        for (uint32_t r : c.rows) totalRows += r;
        const uint64_t totalElems = uint64_t(totalRows) * c.D;
        const uint64_t outBytes = totalElems * esz;

        // Inputs with distinct contents, uploaded once.
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<eva::Buffer> ins(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            const uint64_t elems = uint64_t(c.rows[i]) * c.D;
            std::vector<float> f(elems);
            for (float& v : f) v = dist(rng);
            eva::Buffer staging = hostBuffer(dev, elems * esz);
            convertDtype(f.data(), Dtype::Fp32, staging.map(), c.dtype, elems);
            ins[i] = deviceBuffer(dev, elems * esz);
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
                .copyBuffer(staging(0, elems * esz), ins[i](0, elems * esz))
            .end().submit().wait();
        }
        eva::Buffer outs[3] = { deviceBuffer(dev, outBytes), deviceBuffer(dev, outBytes), deviceBuffer(dev, outBytes) };

        // concat-1: one thread per output element, axis prefix sums.
        eva::ComputePipeline pipe1 = dev.createComputePipeline({
            .csStage = spv1
                + eva::constant_id<SPEC_ID_TYPE_O>(c.dtype)
                + eva::constant_id<SPEC_ID_TYPE_I>(c.dtype),
        });
        Constants consts1 = { uint32_t(totalElems), c.D, totalRows, n, {} };
        for (uint32_t i = 0, s = 0; i < n; ++i)
            consts1.prefixSum[i] = (s += c.rows[i]);

        // concat-2: contiguous=true, the widest pack every input divides into.
        uint32_t packable = 1;
        for (uint32_t width : {4u, 2u})
        {
            bool aligned = true;
            for (uint32_t r : c.rows) aligned &= (uint64_t(r) * c.D) % width == 0;
            if (aligned) { packable = width; break; }
        }
        eva::ComputePipeline pipe2 = dev.createComputePipeline({
            .csStage = spv2
                + eva::constant_id<0>(packable)
                + eva::constant_id<1>(1u)
                + eva::constant_id<SPEC_ID_TYPE_O>(c.dtype)
                + eva::constant_id<SPEC_ID_TYPE_I>(c.dtype),
        });
        Constants consts2 = { uint32_t(totalElems), c.D, totalRows, n, {} };
        uint32_t maxCount = 0;
        for (uint32_t i = 0, s = 0; i < n; ++i)
        {
            const uint32_t count = uint32_t(uint64_t(c.rows[i]) * c.D / packable);
            consts2.prefixSum[i] = (s += count);
            maxCount = std::max(maxCount, count);
        }

        auto descSet = [&](eva::ComputePipeline pipe, eva::Buffer out)
        {
            std::vector<eva::Descriptor> descs(maxInputs + 1, eva::Buffer());
            descs[0] = out;
            for (uint32_t i = 0; i < n; ++i) descs[i + 1] = ins[i];
            eva::DescriptorSet ds = gVaiDestSetPool(pipe.descSetLayout(0));
            ds.write(descs);
            return ds;
        };
        eva::DescriptorSet set1 = descSet(pipe1, outs[0]);
        eva::DescriptorSet set2 = descSet(pipe2, outs[1]);

        auto record1 = [&]
        {
            cmd.bindPipeline(pipe1).bindDescSets({set1})
                .setPushConstants(0, sizeof(consts1), &consts1)
                .dispatch2(uint32_t(totalElems))
                .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outs[0] / eva::SYNC_SCOPE::COMPUTE_READ);
        };
        auto record2 = [&]
        {
            cmd.bindPipeline(pipe2).bindDescSets({set2})
                .setPushConstants(0, sizeof(consts2), &consts2)
                .dispatch2(maxCount, n)
                .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outs[1] / eva::SYNC_SCOPE::COMPUTE_READ);
        };
        auto recordCopy = [&]
        {
            uint64_t dst = 0;
            for (uint32_t i = 0; i < n; ++i)
            {
                const uint64_t bytes = uint64_t(c.rows[i]) * c.D * esz;
                cmd.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / ins[i] / eva::SYNC_SCOPE::TRANSFER_SRC)
                    .copyBuffer(ins[i](0, bytes), outs[2](dst, bytes));
                dst += bytes;
            }
            cmd.barrier(eva::SYNC_SCOPE::TRANSFER_DST / outs[2] / eva::SYNC_SCOPE::COMPUTE_READ);
        };

        // Enough repetitions that the slowest run is tens of milliseconds.
        const uint32_t reps = uint32_t(std::clamp<uint64_t>(4'000'000'000ull / std::max<uint64_t>(outBytes, 1), 20, 2000));

        // Warm each case for at least one second of GPU work before timing.
        double warmupMs = 0.0;
        while (warmupMs < 1000.0)
        {
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 2);
            cmd.writeTimestamp(stage, qp, 0);
            for (uint32_t r = 0; r < reps; ++r)
            {
                record1(); record2(); recordCopy();
            }
            cmd.writeTimestamp(stage, qp, 1).end().submit().wait();
            warmupMs += qp.getElapsedMs(0, 1);
        }

        cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT).resetQueryPool(qp, 0, 6);
        cmd.writeTimestamp(stage, qp, 0);
        for (uint32_t r = 0; r < reps; ++r) record1();
        cmd.writeTimestamp(stage, qp, 1);
        cmd.writeTimestamp(stage, qp, 2);
        for (uint32_t r = 0; r < reps; ++r) record2();
        cmd.writeTimestamp(stage, qp, 3);
        cmd.writeTimestamp(stage, qp, 4);
        for (uint32_t r = 0; r < reps; ++r) recordCopy();
        cmd.writeTimestamp(stage, qp, 5);
        cmd.end().submit().wait();

        const double us[3] = {
            qp.getElapsedMs(0, 1) * 1e3 / reps,
            qp.getElapsedMs(2, 3) * 1e3 / reps,
            qp.getElapsedMs(4, 5) * 1e3 / reps,
        };

        // All three outputs must hold the same bytes.
        std::vector<std::vector<uint8_t>> host(3, std::vector<uint8_t>(outBytes));
        for (int k = 0; k < 3; ++k)
        {
            eva::Buffer readback = hostBuffer(dev, outBytes);
            cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
                .copyBuffer(outs[k](0, outBytes), readback(0, outBytes))
            .end().submit().wait();
            std::memcpy(host[k].data(), readback.map(), outBytes);
        }
        auto shape = [&](uint32_t rows) { return "[" + std::to_string(rows) + "," + std::to_string(c.D) + "]"; };
        std::string inputs;
        for (uint32_t i = 0; i < n;)
        {
            uint32_t end = i + 1;
            while (n > 2 && end < n && c.rows[end] == c.rows[i]) ++end;
            if (i) inputs += " + ";
            if (end - i > 1) inputs += std::to_string(end - i) + " x ";
            inputs += shape(c.rows[i]);
            i = end;
        }

        auto gbps = [&](double t) { return 2.0 * double(outBytes) / (t * 1e3); };
        printf("%-5s  %-36s %9.2f %5u | %11.1f (%7.1f) | %11.1f (%7.1f) | %11.1f (%7.1f)\n",
               c.dtype == Dtype::Fp32 ? "fp32" : "fp16", inputs.c_str(), outBytes / 1e6, packable,
               us[0], gbps(us[0]), us[1], gbps(us[1]), us[2], gbps(us[2]));
        if (host[0] != host[2] || host[1] != host[2])
            printf("       outputs differ\n");
    }

    return 0;
}
