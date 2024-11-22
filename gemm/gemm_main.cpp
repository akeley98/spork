#include "gemm_test.h"

int main()
{
    TestParams params{};
    params.M = 64;
    params.N = 128;
    params.K = 64;
    params.test_data_code_A = TestDataCode::tiled_numbers;
    params.test_data_code_B = TestDataCode::identity;
    gemm_test(params, {});
    cudaDeviceSynchronize();
}
