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

struct TestParams
{
    uint32_t M, N, K;
    TestDataCode test_data_code_A, test_data_code_B;
};

void gemm_test(TestParams params, cudaStream_t stream);
