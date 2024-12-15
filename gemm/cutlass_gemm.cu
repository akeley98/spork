#include "cutlass_gemm.h"

#include <cutlass/gemm/device/gemm.h>
#include <stdio.h>

using gemm_t = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor,
                                           float, cutlass::layout::ColumnMajor,
                                           float, cutlass::layout::RowMajor,
                                           float, cutlass::arch::OpClassSimt, cutlass::arch::Sm70>;


void matmul_cutlass(GPU_Tensors t, cudaStream_t stream)
{
    assert(!t.a_col_major);
    assert(t.b_col_major);
    assert(!t.c_col_major);

    gemm_t gemm_op;
    cutlass::Status status = gemm_op({
        {int(t.M), int(t.N), int(t.K)},
        {t.a, int(t.K)},
        {t.b, int(t.K)},
        {t.c, int(t.N)},
        {t.c, int(t.N)},
        {1.0f, 0.0f}
    }, nullptr, stream);

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "%s:%i: %i\n", __FILE__, __LINE__, (int)status);
    }
}
