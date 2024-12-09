#include "gemm_test.h"

int main()
{
    {
        TestParams params{};
        params.M = 768;
        params.N = 768;
        params.K = 192;
        params.test_data_code_A = TestDataCode::tiled_numbers;
        params.test_data_code_B = TestDataCode::identity;
        gemm_test(params, {});
    }

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

    sized_tests(768, 768, 3 * 128, false);
    sized_tests(768, 7680, 3 * 128, false);
    sized_tests(7680, 768, 3 * 128, false);
    sized_tests(7680, 7680, 3 * 128, false);
    sized_tests(3 * 4096, 2560, 3 * 2048, true);
    sized_tests(3 * 4096, 25600, 3 * 512, true);
    sized_tests(1536, 2048, 3 * 40960, true);
    sized_tests(1536, 1024, 3 * 40960, true);
    sized_tests(768, 512, 3 * 40960, true);

    cudaDeviceSynchronize();
}
