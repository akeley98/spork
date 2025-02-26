#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

void cutlass_synclog_setup();
void cutlass_synclog_print();

enum class TestDataCode
{
    random = 0,
    identity = 1,
    tiled_numbers = 2,
};

enum class AlgorithmCode
{
    cublas,
    cutlass,
    exo,
    mine_output_stationary,
    mine_split_k_inner,
    mine_split_k_outer,
    mine_stream_k_early_tma,
    mine_stream_k_late_tma,
};

constexpr uint32_t algorithm_count = 1u + static_cast<uint32_t>(AlgorithmCode::mine_stream_k_late_tma);

inline uint32_t algorithm_code_bit(AlgorithmCode code)
{
    return 1u << static_cast<uint32_t>(code);
}

inline const char* algorithm_name(AlgorithmCode code)
{
    switch (code) {
      case AlgorithmCode::mine_output_stationary:
        return "mine (output stationary)";
      case AlgorithmCode::cublas:
        return "cublas";
      case AlgorithmCode::cutlass:
        return "cutlass";
      case AlgorithmCode::mine_split_k_inner:
        return "mine (split k inner)";
      case AlgorithmCode::mine_split_k_outer:
        return "mine (split k outer)";
      case AlgorithmCode::mine_stream_k_early_tma:
        return "mine (stream k early TMA)";
      case AlgorithmCode::mine_stream_k_late_tma:
        return "mine (stream k late TMA)";
      case AlgorithmCode::exo:
        return "exo";
    }
    return "unknown";
}

struct TestParams
{
    uint32_t M, N, K;
    TestDataCode test_data_code_A, test_data_code_B;
    uint32_t algorithm_code_bits;
};

void gemm_test(TestParams params, cudaStream_t stream);
