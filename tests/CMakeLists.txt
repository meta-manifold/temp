cmake_minimum_required(VERSION 3.16)

project(vai-tests)

# One folder per test, one executable per folder. To add a test: make a folder,
# drop its sources in, and append a block here.

# ---------------------------------------------------------------------------
# linear-nodes — LinearNode against the NaiveLinearNode reference, side by side
# ---------------------------------------------------------------------------
add_executable(test-linear-nodes
    linear-nodes/main.cpp
)

target_link_libraries(test-linear-nodes PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# convolution-residual — Convolution's residual, both orders, against a CPU
# reference on every path the plan selector takes
# ---------------------------------------------------------------------------
add_executable(test-convolution-residual
    convolution-residual/main.cpp
)

target_link_libraries(test-convolution-residual PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# group-norm — GroupNorm against a CPU reference, ranks 3 and 4, every kernel path
# ---------------------------------------------------------------------------
add_executable(test-group-norm
    group-norm/main.cpp
)

target_link_libraries(test-group-norm PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# norm-residual — LayerNorm and RmsNorm with the residual, against a CPU reference
# ---------------------------------------------------------------------------
add_executable(test-norm-residual
    norm-residual/main.cpp
)

target_link_libraries(test-norm-residual PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# linear-slab — linear-slab vs test-linear-slab2, the A/B geometry variant
# (a subgroup owns each slab instead of a workgroup). slab2 is not part of vAi,
# so it is built here and loaded by path through test-nodes.h.
# ---------------------------------------------------------------------------
set(LINEAR_SLAB_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/linear-slab)

# main.cpp verifies and benchmarks; main2.cpp benchmarks only. Swap which one
# is active — they both define main().
add_executable(test-linear-slab
    # linear-slab/main.cpp
    linear-slab/main2.cpp

    linear-slab/test-linear-slab.comp
    linear-slab/test-linear-slab2.comp
    linear-slab/test-linear-slab-v0.comp
    linear-slab/test-linear-slab-v2.comp
    linear-slab/test-linear-slab-v5.comp
    linear-slab/test-stream.comp
)

eva_auto_compile_shaders(test-linear-slab ${LINEAR_SLAB_SHADER_DIR})

target_link_libraries(test-linear-slab PRIVATE
    vAi
)

target_compile_definitions(test-linear-slab PRIVATE
    TEST_LINEAR_SLAB_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab.comp.spv"
    TEST_LINEAR_SLAB2_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab2.comp.spv"
    TEST_LINEAR_SLAB_V0_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab-v0.comp.spv"
    TEST_LINEAR_SLAB_V2_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab-v2.comp.spv"
    TEST_LINEAR_SLAB_V5_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab-v5.comp.spv"
    TEST_STREAM_SPV="${LINEAR_SLAB_SHADER_DIR}/test-stream.comp.spv"
)

# ---------------------------------------------------------------------------
# vai-vs-native — what the framework costs on top of the kernel: the same
# ggml-style-matmul .spv driven by raw Vulkan and by a vAi graph. The raw path
# calls Vulkan directly, so it needs the loader on top of what vAi pulls in.
#
# The kernel is a benchmark reference, not part of vAi, so it lives with the
# test and is compiled here rather than by eva's vai/shaders sweep. Both paths
# load this one .spv, which is what makes the comparison host-side only.
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)

set(VAI_VS_NATIVE_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/vai-vs-native)

add_executable(test-vai-vs-native
    vai-vs-native/main.cpp
    vai-vs-native/on-native.cpp
    vai-vs-native/on-vai.cpp
    vai-vs-native/comparison-native.h
    vai-vs-native/test-linear-node.h

    vai-vs-native/ggml-style-matmul.comp
)

eva_auto_compile_shaders(test-vai-vs-native ${VAI_VS_NATIVE_SHADER_DIR})

target_include_directories(test-vai-vs-native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vai-vs-native
)

target_link_libraries(test-vai-vs-native PRIVATE
    vAi
    Vulkan::Vulkan
)

target_compile_definitions(test-vai-vs-native PRIVATE
    GGML_STYLE_MATMUL_SPV="${VAI_VS_NATIVE_SHADER_DIR}/ggml-style-matmul.comp.spv"
)

# ---------------------------------------------------------------------------
# attention — AttentionNode's per-(shape, device) paths (coopmat FA,
# scalar FA, GQA-packed decode, split-K, naive fallback) vs a host reference,
# plus a shape bench. The kernels are vAi's own, so no shader compile here.
# ---------------------------------------------------------------------------
add_executable(test-attention
    attention/main.cpp
)

target_link_libraries(test-attention PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# linear-cm — LinearNode's coopmat path (linear-cm) vs the NaiveLinearNode
# reference, plus the device-local-input benchmark (graph inputs are pinned,
# which distorts staging-bound GEMM timings).
# ---------------------------------------------------------------------------
add_executable(test-linear-cm
    linear-cm/main.cpp
)

target_link_libraries(test-linear-cm PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# test-robust-buffer-access — the same tile GEMM with the loaders' row-tail
# guards kept vs folded away behind VK_EXT_pipeline_robustness. vai::node::Linear
# derives that choice from the device, so the A/B needs a local copy of the
# node with the option forced; it lives beside the test and uses vAi's kernels.
# ---------------------------------------------------------------------------
# main.cpp is the A/B table over model shapes; main2.cpp sweeps M across a
# workgroup-tile boundary to split the two costs folding the guards away pays.
# Swap which one is active — they both define main().
add_executable(test-robust-buffer-access
    # test-robust-buffer-access/main.cpp
    test-robust-buffer-access/main2.cpp

    test-robust-buffer-access/bench-common.h
    test-robust-buffer-access/test-robust-linear-node.h
)

target_link_libraries(test-robust-buffer-access PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# tokenizer — Tokenizer vs HuggingFace tokenizers on the tokenizer.json files
# under data/, as fixed cases carrying the reference ids. Host-side only, so it
# is the one test here that needs no device.
# ---------------------------------------------------------------------------
add_executable(test-tokenizer
    tokenizer/main.cpp
)

target_link_libraries(test-tokenizer PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# llama3-decode-tps — decode throughput of the llama3 sample model, prefill
# excluded. The model owns its definition and its checkpoint loader, so this
# test and models/production/llama3/demo.cpp can never end up on different weights.
# ---------------------------------------------------------------------------
add_executable(test-llama3-decode-tps
    llama3-decode-tps/main.cpp
)

target_link_libraries(test-llama3-decode-tps PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# vit-batching - pi0's vision tower over its three cameras in one pass against
# three passes of one. Same arithmetic either way, so the gap is dispatch size
# and the per-forward graph walk. Weights are random; only shapes matter.
# ---------------------------------------------------------------------------
add_executable(test-vit-batching
    vit-batching/main.cpp
)

target_link_libraries(test-vit-batching PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# chunked-prefill — the generated ids against the chunk size.
# ---------------------------------------------------------------------------
add_executable(test-chunked-prefill
    chunked-prefill/main.cpp
)

target_link_libraries(test-chunked-prefill PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# gemma3 — the Gemma 3 4B VLM against PyTorch, layer by layer and step by step.
# Dumps every layer boundary of one image prefill plus N decode steps; the
# PyTorch side (dump_torch.py) and the table (compare.py) live next to it. The
# model owns its definition and its checkpoint loader, so this test and
# models/production/gemma3/demo.cpp can never end up on different weights.
# ---------------------------------------------------------------------------
add_executable(test-gemma3-dump
    gemma3/dump-vai.cpp
)

target_include_directories(test-gemma3-dump PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3          # npy.h
)

target_link_libraries(test-gemma3-dump PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# concat-contiguous — Concat's contiguous case three ways: concat-1.comp (one
# thread per output element), concat-2.comp under contiguous=true, and buffer
# copies, timed and compared byte for byte. Both kernels live with the test.
# ---------------------------------------------------------------------------
set(CONCAT_CONTIGUOUS_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/concat-contiguous)

add_executable(test-concat-contiguous
    concat-contiguous/main.cpp

    concat-contiguous/concat-1.comp
    concat-contiguous/concat-2.comp
)

eva_auto_compile_shaders(test-concat-contiguous ${CONCAT_CONTIGUOUS_SHADER_DIR})

target_link_libraries(test-concat-contiguous PRIVATE
    vAi
)

target_compile_definitions(test-concat-contiguous PRIVATE
    CONCAT_1_SPV="${CONCAT_CONTIGUOUS_SHADER_DIR}/concat-1.comp.spv"
    CONCAT_2_SPV="${CONCAT_CONTIGUOUS_SHADER_DIR}/concat-2.comp.spv"
)

# ---------------------------------------------------------------------------
# linear-pi05 — every Linear pi0.5 runs per inference with its bias / epilogue /
# residual: GPU time, TFLOP/s, weight read rate, time per inference.
# ---------------------------------------------------------------------------
add_executable(test-linear-pi05
    linear-pi05/main.cpp
)

target_link_libraries(test-linear-pi05 PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# linear-tune-intel — the scalar tile GEMM (linear-tile-N.comp, copies and
# variants of vai/shaders/linear-tile.comp) on pi0.5's language FFN shapes over
# configurations given on the command line: time, TFLOP/s, error, driver stats.
# ---------------------------------------------------------------------------
set(LINEAR_TUNE_INTEL_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/linear-tune-intel)

file(GLOB LINEAR_TUNE_INTEL_SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/linear-tune-intel/*.comp)

add_executable(test-linear-tune-intel
    linear-tune-intel/main.cpp
    ${LINEAR_TUNE_INTEL_SHADERS}
)

eva_auto_compile_shaders(test-linear-tune-intel ${LINEAR_TUNE_INTEL_SHADER_DIR})

target_link_libraries(test-linear-tune-intel PRIVATE
    vAi
)

target_compile_definitions(test-linear-tune-intel PRIVATE
    TILE_SHADER_DIR="${LINEAR_TUNE_INTEL_SHADER_DIR}"
)

# ---------------------------------------------------------------------------
# linear-vec-intel — M = 1 .. 8 Linear kernels over the Intel path's packed W
# (linear-tile-intel, vec.comp and variants) against linear-slab as Linear runs it.
# ---------------------------------------------------------------------------
set(LINEAR_VEC_INTEL_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/linear-vec-intel)

file(GLOB LINEAR_VEC_INTEL_SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/linear-vec-intel/*.comp)

add_executable(test-linear-vec-intel
    linear-vec-intel/main.cpp
    ${LINEAR_VEC_INTEL_SHADERS}
)

eva_auto_compile_shaders(test-linear-vec-intel ${LINEAR_VEC_INTEL_SHADER_DIR})

target_link_libraries(test-linear-vec-intel PRIVATE
    vAi
)

target_compile_definitions(test-linear-vec-intel PRIVATE
    VEC_SHADER_DIR="${LINEAR_VEC_INTEL_SHADER_DIR}"
    SLAB_SPV="${CMAKE_BINARY_DIR}/shaders/linear-slab.comp.spv"
    TILE_SPV="${CMAKE_BINARY_DIR}/shaders/linear-tile-intel.comp.spv"
)

# ---------------------------------------------------------------------------
# attention-tune-intel — flash-attention variants for Intel GPUs without
# matrix engines (attention-fa-intel*.comp) on pi0.5's attention shapes,
# checked against a host reference and timed.
# ---------------------------------------------------------------------------
set(ATTENTION_TUNE_INTEL_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/attention-tune-intel)

file(GLOB ATTENTION_TUNE_INTEL_SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/attention-tune-intel/*.comp)

add_executable(test-attention-tune-intel
    attention-tune-intel/main.cpp
    ${ATTENTION_TUNE_INTEL_SHADERS}
)

eva_auto_compile_shaders(test-attention-tune-intel ${ATTENTION_TUNE_INTEL_SHADER_DIR})

target_link_libraries(test-attention-tune-intel PRIVATE
    vAi
)

target_compile_definitions(test-attention-tune-intel PRIVATE
    ATTN_SHADER_DIR="${ATTENTION_TUNE_INTEL_SHADER_DIR}"
    ATTN_NODE_SPV="${CMAKE_BINARY_DIR}/shaders/attention-fa-intel.comp.spv"
)

# ---------------------------------------------------------------------------
# act — ACT's action chunk against a numpy forward of the raw LeRobot
# checkpoint (act/reference.py), on a closed-form image and state.
# ---------------------------------------------------------------------------
add_executable(test-act
    act/main.cpp
)

target_link_libraries(test-act PRIVATE
    vAi
)
