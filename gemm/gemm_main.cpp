#include "gemm_test.h"

int main()
{
    auto sized_tests = [] (uint32_t M, uint32_t N, uint32_t K)
    {
        TestParams params{};
        params.M = M;
        params.N = N;
        params.K = K;
        {
            params.test_data_code_A = TestDataCode::tiled_numbers;
            params.test_data_code_B = TestDataCode::identity;
            gemm_test(params, {});
        }
        {
            params.test_data_code_A = TestDataCode::identity;
            params.test_data_code_B = TestDataCode::tiled_numbers;
            gemm_test(params, {});
        }
        {
            params.test_data_code_A = TestDataCode::random;
            params.test_data_code_B = TestDataCode::random;
            gemm_test(params, {});
        }
    };
    sized_tests(128, 128, 16);
    sized_tests(256, 128, 16);
    sized_tests(128, 256, 16);
    sized_tests(256, 256, 64);
    sized_tests(256, 256, 192);
    sized_tests(1280, 768, 640);
    sized_tests(1280, 2560, 640);
    cudaDeviceSynchronize();
}
