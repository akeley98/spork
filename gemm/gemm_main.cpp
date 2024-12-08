#include "gemm_test.h"

int main()
{
    auto sized_tests = [] (uint32_t M, uint32_t N, uint32_t K, bool nontrivial_only)
    {
        TestParams params{};
        params.M = M;
        params.N = N;
        params.K = K;
        if (!nontrivial_only) {
            params.test_data_code_A = TestDataCode::tiled_numbers;
            params.test_data_code_B = TestDataCode::identity;
            gemm_test(params, {});
        }
        if (!nontrivial_only) {
            params.test_data_code_A = TestDataCode::identity;
            params.test_data_code_B = TestDataCode::tiled_numbers;
            gemm_test(params, {});
        }
        if (true) {
            params.test_data_code_A = TestDataCode::random;
            params.test_data_code_B = TestDataCode::random;
            gemm_test(params, {});
        }
    };

    sized_tests(256, 512, 3 * 128, false);
    sized_tests(512, 256, 3 * 128, false);
    sized_tests(256, 256, 3 * 512, false);
    sized_tests(256, 1024, 3 * 128, false);
    sized_tests(1280, 768, 3 * 256, false);
    sized_tests(1280, 2560, 3 * 256, false);
    sized_tests(12800, 2560, 3 * 2048, true);
    sized_tests(12800, 25600, 3 * 512, true);
    sized_tests(2048, 2048, 3 * 4096, true);
    sized_tests(1024, 1024, 3 * 4096, true);
    sized_tests(256, 512, 3 * 4096, true);
    cudaDeviceSynchronize();
}

