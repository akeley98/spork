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

enum class TestKModes
{
    all = 0,
    output_stationary = 1,
    split_k_outer = 2,
    split_k_inner = 3,
    stream_k_early_tma = 4,
    stream_k_late_tma = 5,
};

struct TestParams
{
    uint32_t M, N, K;
    TestDataCode test_data_code_A, test_data_code_B;
    TestKModes test_k_modes;
};

void gemm_test(TestParams params, cudaStream_t stream);
