#pragma once

#include <cuda_runtime.h>

#include "cute/numeric/numeric_types.hpp"

namespace cute_gemm {

void f16(char transA, char transB, int m, int n, int k,
         cute::half_t const* A, int ldA,
         cute::half_t const* B, int ldB,
         cute::half_t* C, int ldC,
         cudaStream_t stream);

}
