#include "cutlass_gemm.h"

#include <stdio.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/packed_stride.hpp"

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#error CUTLASS_ARCH_MMA_SM90_SUPPORTED
#endif

namespace cutlass_h100_gemm {

using namespace cute;

// cutlass/examples/48_hopper_warp_specialized_gemm.cu

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

using element_t = float;

// A matrix configuration
using         ElementA    = element_t;                                      // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = element_t;                                      // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = element_t;                                      // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = element_t;                                      // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_256,_128,_32>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

void matmul(GPU_Tensors t, StreamWorkspace& stream_ws)
{
    assert(!t.a_col_major);
    assert(t.b_col_major);
    assert(!t.c_col_major);

    // Swap a/b, m/n to deal with BLAS-style column major output
    StrideB stride_original_a = cutlass::make_cute_packed_stride(StrideB{}, {int(t.M), int(t.K), 1});
    StrideA stride_original_b = cutlass::make_cute_packed_stride(StrideA{}, {int(t.N), int(t.K), 1});
    StrideC stride_original_c = cutlass::make_cute_packed_stride(StrideC{}, {int(t.N), int(t.M), 1});

    Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {int(t.N), int(t.M), int(t.K)},
      {t.b, stride_original_b, t.a, stride_original_a},
      {{1.0f, 0.0f}, t.c, stride_original_c, t.c, stride_original_c}};
    args.scheduler.raster_order = RasterOrderOptions::Heuristic;
    args.scheduler.max_swizzle_size = 1;  // XXX

    Gemm gemm;
    const auto ws_size = gemm.get_workspace_size(args);
    gemm.initialize(args, stream_ws.alloc_at_least(ws_size));
    const auto status = gemm.run(stream_ws);

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "%s:%i: %i\n", __FILE__, __LINE__, (int)status);
    }
}

}  // end namespace

void matmul_cutlass(GPU_Tensors t, StreamWorkspace& stream_ws)
{
    cutlass_h100_gemm::matmul(t, stream_ws);
}
