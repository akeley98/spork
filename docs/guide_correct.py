from __future__ import annotations
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.x86 import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *

xyzzy = 1

# TeX: version async_block 1
# TeX: begin async_block[0]
@proc
def async_blocks():
    # TeX: remark! *
    # CPU code here
    gmem: f32[1024] @ CudaGmemLinear  # CPU calls cudaMallocAsync
    grid_const: f32 @ CudaGridConstant
    grid_const = 42
    # Launch CUDA kernel with blockDim=256; gridDim={device-specific}
    # Exo implicitly transfers CPU-defined values (gmem, grid_const) as kernel launch parameters.
    # TeX: color line *
    #    gggggggggggggggggg
    with CudaDeviceFunction(blockDim=256):
        # TeX: remark! *
        # In-order CUDA code here
        for t in cuda_tasks(0, xyzzy):  # See ``Loop Mode''
            smem: f32[64] @ CudaSmemLinear  # Shared memory allocation
            for tid in cuda_threads(0, 256):  # See ``Loop Mode''
                smem[0] = grid_const
                # TeX: color line *
                #    ggggggggg
                with CudaAsync(Sm80_cp_async):
                    # TeX: remark! *
                    # Async CUDA code here
                    Sm80_cp_async_f32(smem[4:8], gmem[4:8], size=4)  # smem[4:8] = gmem[4:8]
                    # TeX: end async_block[0]

# TeX: version intro_tasks_threads 1
# TeX: begin intro_tasks_threads[0]
@proc
def intro_tasks_threads(N: size, X: f32[N] @ CudaGmemLinear, Y: f32[N] @ CudaGmemLinear):
    assert N % 128 == 0
    # TeX: color line *
    #                                yyy
    with CudaDeviceFunction(blockDim=128):
        # Hello world: parallel X += Y vec add.
        # TeX: color line *
        #           gggggggggg
        for task in cuda_tasks(0, N / 128):
            # TeX: color remark! *
            #                              yyy
            # Collective unit here: CTA of 128 threads
            # TeX: color line *
            #          bbbbbbbbbbbb         bbbbbbbbbbbbbbbb
            for tid in cuda_threads(0, 128, unit=cuda_thread):
                # TeX: color remark! *
                #                       bbbbbbbbbbbbb
                # Collective unit here: 1 CUDA thread
                X[128 * task + tid] += Y[128 * task + tid]
            # TeX: color line *
            #                               ggggggggggg  bbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            # Underlying loop mode objects: CudaTasks(), CudaThreads(unit=cuda_thread)
# TeX: end intro_tasks_threads[0]

# TeX: version cuda_threads_cxx 1
@proc
def cuda_threads_cxx():
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            # TeX: begin cuda_threads_cxx[0]
            # Exo
            # TeX: color line *
            #                       g          vv
            # Break threads into an 8 $\times$ 16 iteration space
            # TeX: color line *
            #   g                    g       bbbbbbbbbbbbbbbb
            for m in cuda_threads(0, 8, unit=16 * cuda_thread):
                # TeX: color line *
                #   v                    vv
                for n in cuda_threads(0, 16, unit=cuda_thread):
                    # TeX: end cuda_threads_cxx[0]
                    pass
"""
            # TeX: begin cuda_threads_cxx[0]
            # C++
            # TeX: color line *
            #       ggggggggggg                  bb                 g
            if (int exo_16thr_m = (threadIdx.x / 16); exo_16thr_m < 8) {
              # TeX: color line *
              #       vvvvvvvvvv
              if (int exo_1thr_n = (threadIdx.x % 16); 1) {
            # TeX: end cuda_threads_cxx[0]
"""

# TeX: version my_warp_config 1
# TeX: begin my_warp_config[0]
my_warp_config = [
    CudaWarpConfig("producer", 1, setmaxnreg_dec=40),  # 1 warp; reduce to 40 regs
    CudaWarpConfig("unused", 3, setmaxnreg_dec=40),  # 3 warps; reduce to 40 regs
    CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),  # 8 warps; increase to 232 regs
]  # Total: 12 warps: blockDim=384

@proc
def warp_config_example():
    with CudaDeviceFunction(warp_config=my_warp_config):
        for task in cuda_tasks(0, xyzzy):
            smem: f32[3, 128, 256] @ CudaSmemLinear  # Common code executed by all warps
            with CudaWarps(name="producer"):
                # The string ``producer'' has no meaning to Exo,
                # but it's less confusing if you put producer code here.
                # TeX: end my_warp_config[0]
                for tid in cuda_threads(0, 1):
                    smem[0,0,0] = 0  # compiler warning fix
            # TeX: begin my_warp_config[0]
            with CudaWarps(name="consumer"):
                # The string ``consumer'' has no meaning to Exo,
                # but it's less confusing if you put consumer code here.
                # TeX: end my_warp_config[0]
                pass


# TeX: version warpgroup_CudaWarps 1
# TeX: begin warpgroup_CudaWarps[0]
@proc
def warpgroup_CudaWarps():
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                # Collective unit here is 1 warpgroup (4 warps)
                with CudaWarps(3, 4):
                    # Collective unit here is 1 warp
                    # TeX: end warpgroup_CudaWarps[0]
                    pass


@proc
def simple_dist():
    # TeX: version simple_dist 2
    # TeX: begin simple_dist
    with CudaDeviceFunction(blockDim=512):
        for task in cuda_tasks(0, xyzzy):
            # TeX: color remark simple_dist[0]
            # rrrrrrrrrrrrrrrrrrrrr  yyyyyyyyyyyyyyyyyyyyyyy
            # Distributed (16 x 32); non-distributed (8 x 4)
            # TeX: color line *
            #         rrrrrr  yyyy                                      yyyyy
            vals: f32[16, 32, 8, 4] @ CudaRmem  # Each thread allocates 8 x 4 registers
            # TeX: remark simple_dist[1]
            # Tile here: (512,); $t_a = 512$; $t_n = 1$ (native unit cuda_thread for CudaRmem).
            # TeX: color line *
            #   g                                                   gggggggggggggggggg
            for m in cuda_threads(0, 16, unit=32 * cuda_thread):  # m: $512\mapsto 32$
                # TeX: remark simple_dist[1]
                # Tile here: (32,)
                # TeX: color line *
                #   v                                               vvvvvvvvvvvvvvvv
                for n in cuda_threads(0, 32, unit=cuda_thread):   # n: $32\mapsto 1$
                    # TeX: remark simple_dist[1]
                    # Tile here: (1,)
                    # TeX: color line simple_dist
                    #    g  v  yyyy
                    vals[m, n, 0, 0] = 0
                    # TeX: color line simple_dist
                    #    g  v  yyyy
                    vals[m, n, 0, 1] = 0
                    # TeX: remark simple_dist[0]
                    # ...
                    # TeX: color remark simple_dist[1]
                    #               ggggggggggggggggggg  vvvvvvvvvvvvvvvvv
                    # Tiling chain: m: $512 \mapsto 32$, n: $32 \mapsto 1$
                    # TeX: color remark simple_dist[1]
                    #                       yyyyyyyyyyyyyyy
                    # Remaining indices are non-distributed
    # TeX: end simple_dist

"""
# TeX: version simple_dist_cxx 1
# TeX: begin simple_dist_cxx[0]
# TeX: color line *
#          yyyyy                                rrrrrrr
float vals[8 * 4];  # Distributed across CTA of 16 x 32 threads
# TeX: color line *
#                        ggggggggggg                               ggggggggggggggggggg
if ([[maybe_unused]] int exo_32thr_m = (threadIdx.x / 32); 1) {  # m: $512 \mapsto 32$
# TeX: color line *
#                          vvvvvvvvvv                               vvvvvvvvvvvvvvvvv
  if ([[maybe_unused]] int exo_1thr_n = (threadIdx.x % 32); 1) {  # n: $32 \mapsto 1$
    vals[0] = 0.0f;
    vals[1] = 0.0f;  # [m, n] (distributed indices) removed
    # TeX: end simple_dist_cxx[0]
  }
}
"""

# TeX: version sync_warp_cta 1
@proc
def sync_warp_cta():
    with CudaDeviceFunction(blockDim=128):
        # TeX: begin sync_warp_cta[0]
        for task in cuda_tasks(0, xyzzy):
            smem: f32[4] @ CudaSmemLinear
            # TeX: color line *
            #                                bbbbbbbbb
            for w in cuda_threads(0, 4, unit=cuda_warp):
                for tid in cuda_threads(0, 1, unit=cuda_thread):
                    smem[w] = 42
                # TeX: color remark! *
                #                       bbbb
                # Collective unit here: warp (Fence lowers to __syncwarp() equivalent)
                # TeX: color line *
              #       ggggggggggggg  ggggggggggggg
                Fence(cuda_in_order, cuda_in_order)
                # Only threads in the same warp can see smem[w] = 42
            # TeX: remark! *
            # Collective unit here: CTA  (Fence lowers to __syncthreads() equivalent)
            # TeX: color line *
          #       ggggggggggggg  ggggggggggggg
            Fence(cuda_in_order, cuda_in_order)
            # Now all threads in the CTA can see smem[0:4] = [42, 42, 42, 42]
            # TeX: color line *
            #       ggggggggggggg
            # NOTE: cuda_in_order means non-async CUDA instructions
            # TeX: end sync_warp_cta[0]


# TeX: version Sm80_cp_async_simple 1
@proc
def Sm80_cp_async_simple(src: f32[128] @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=128):
        for task in cuda_tasks(0, xyzzy):
            # TeX: begin Sm80_cp_async_simple[0]
            sync_dst: f32[128] @ CudaSmemLinear
            async_dst: f32[128] @ CudaSmemLinear
            for tid in cuda_threads(0, 128, unit=cuda_thread):
                # TeX: color line *
                #                           gggggggg
                sync_dst[tid] = src[tid]  # in-order CUDA instruction (outside CudaAsync block)
                with CudaAsync(Sm80_cp_async):
                    # TeX: color line *
                    #                            yyyyyyyyyyyyyy
                    # async_dst[tid] = src[tid], asynchronously
                    Sm80_cp_async_f32(async_dst[tid:tid+1], src[tid:tid+1], size=1)
            # Here: only thread #tid can see sync_dst[tid] = src[tid]
            # TeX: color line *
            #     ggggggggggggg  ggggggggggggg
            Fence(cuda_in_order, cuda_in_order)
            # Here: sync_dst[tid] = src[tid] visible to all threads in CTA
            # The write to async_dst[tid] is still pending
            # TeX: color line *
            #     yyyyyyyyyyyyy  ggggggggggggg
            Fence(Sm80_cp_async, cuda_in_order)
            # Here: all threads in the CTA can see sync_dst[tid] = src[tid] and async_dst[tid] = src[tid]
            # TeX: end Sm80_cp_async_simple[0]

# TeX: version xgemm_Sm80_fence 4
# TeX: begin xgemm_Sm80_fence[0]
Mw, Nw = 96, 64
M1, N1 = 192, 256
K0, MMA_K = 16, 4
@proc
def xgemm_Sm80_fence(M: size, N: size, K: size,
                     A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear,
                     C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0
    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # TeX: end xgemm_Sm80_fence[0]
                # TeX: begin xgemm_Sm80_fence
                # Per CTA code: each CTA handles (M1 x N1) tile
                A_smem : f32[2, M1, K0] @ CudaSmemLinear  # Input tiles (double buffered)
                B_smem : f32[2, K0, N1] @ CudaSmemLinear
                # Accumulator tiles
                # TeX: color remark xgemm_Sm80_fence[1] xgemm_Sm80_fence[3]
                #                                       rrrrrrrrrrrrrrrrrrrrrr
                # Distributed memory: suballocated into (M1/Mw = 2, N1/Nw = 4) grid of warps
                # TeX: color remark xgemm_Sm80_fence[1]
                #                   yyyyyyyyyyyyyyyyyyyyy
                # Each warp holds a (Mw/16 = 6, Nw/8 = 4) grid of MMA D tiles, themselves
                # TeX: color remark xgemm_Sm80_fence[1]
                #         bbbbbbbbbbbbb
                # holding $16 \times 8$ values in a CUDA-defined packed format. $t_a = 256$, $t_n = 32$
                # TeX: color line xgemm_Sm80_fence[1]
                #            rrrrrrrrrrrr  yyyyyyyyyyy  bbbbb
                # TeX: color line xgemm_Sm80_fence[3]
                #            rrrrrrrrrrrr
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD(16, 8)
                # TeX: end xgemm_Sm80_fence
                # TeX: begin xgemm_Sm80_fence[0]
                # TeX: begin xgemm_Sm80_fence[1]
                # TeX: color line xgemm_Sm80_fence[1]
                #   gg
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    # TeX: color line xgemm_Sm80_fence[1]
                    #   vv
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        # TeX: end xgemm_Sm80_fence[0]
                        # TeX: summary
                        # Each warp zeros out its accumulators
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                # TeX: color line xgemm_Sm80_fence[1]
                                #                           gg  vv
                                Sm80_mma_zero_d_tf32(D_rmem[mw, nw, m_seq, n_seq,:,:])
                                # TeX: color remark xgemm_Sm80_fence[1]
                                # ggggggggggggggggggggg  vvvvvvvvvvvvvvvvvvvv
                                # mw: $256 \mapsto 128$, nw: $128 \mapsto 32$
                # TeX: end xgemm_Sm80_fence[1]

                # TeX: begin xgemm_Sm80_fence
                # K tiles loop, double buffered. 1 iteration delay between load and use.
                # Don't accum tile in first iteration; don't load tile in last iteration.
                for k1 in seq(0, K / K0 + 1):
                    if k1 < K / K0:
                        # TeX: end xgemm_Sm80_fence
                        # TeX: begin xgemm_Sm80_fence[2]
                        # TeX: color line *
                        #    yyyyyyyyy
                        with CudaAsync(Sm80_cp_async):  # We must wrap cp.async usage with CudaAsync
                            # TeX: summary
                            # Load A tile using cp.async
                            # TeX: color line *
                            #         yyy
                            for m1 in seq(0, M1 / 64):
                                # Split CTA into (m0, k0) grid of (64, 4) threads
                                for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                    for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                        Sm80_cp_async_f32(  # Exo cp.async instr; double buffer with k1 % 2
                                            # TeX: color line *
                                            #      yyyyyy
                                            A_smem[k1 % 2, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                            A[m2 * M1 + m1 * 64 + m0, k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4],
                                            size=4)
                            # TeX: summary
                            # Load B tile using cp.async
                            # TeX: color line *
                            #             yyy
                            for k0_seq in seq(0, 4):
                                # Split CTA into (k0_par, n0) grid of (4, 64) threads
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        Sm80_cp_async_f32(  # Exo cp.async instr; double buffer with k1 % 2
                                            # TeX: color line *
                                            #      yyyyyy
                                            B_smem[k1 % 2, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                            B[k1 * K0 + k0_seq * 4 + k0_par,
                                              n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4],
                                            size=4)
                    # TeX: end xgemm_Sm80_fence[2]
                    # TeX: begin xgemm_Sm80_fence
                    if k1 > 0:
                        # TeX: remark! xgemm_Sm80_fence[3]
                        # Split CTA into (mw, nw) grid of (2, 4) warps
                        # TeX: color line *
                        #   rr
                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            # TeX: color line *
                            #   rr
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # TeX: end xgemm_Sm80_fence
                                # TeX: begin xgemm_Sm80_fence[3]
                                # Load all B matrix tiles ahead of time
                                # Note double buffer index 1 - k1 % 2 (opposite buffer as used by cp.async)
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB(8, MMA_K)
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        # TeX: color line *
                                        #                                                    yyyyyyyyyy
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:], B_smem[1 - k1 % 2,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)
                                # TeX: color line *
                                #                   ggggggg
                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA(16, MMA_K)
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        # TeX: color line *
                                        #                                              yyyyyyyyyy
                                        Sm80_mma_load_a_tf32(A_rmem[k_seq,:,:], A_smem[1 - k1 % 2,
                                                             mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # TeX: summary
                                    # Accumulate to tile owned by warp.
                                    # TeX: color line *
                                    #                      ggggggg  vvvvvv
                                    # Each warp handles a (Mw / 16, Nw / 8) grid of $16 \times 8$ packed tiles
                                    # TeX: color line *
                                    #                   vvvvvv
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            # TeX: color line *
                                            #                    rr  rr
                                            Sm80_mma_tf32(D_rmem[mw, nw, m_seq, n_seq, :,:],
                                                          A_rmem[k_seq,:,:], B_rmem[k_seq,n_seq,:,:], K=MMA_K)
                                            # TeX: color remark xgemm_Sm80_fence[3]
                                            # rrrrrrrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrrrrrr
                                            # mw: $256 \mapsto 128$, nw: $128 \mapsto 32$
                                # TeX: end xgemm_Sm80_fence[3]

                    # TeX: begin xgemm_Sm80_fence
                    # Sm80_generic = (cuda_in_order | Sm80_cp_async)
                    # TeX: color line xgemm_Sm80_fence[0]
                  # yyyyy
                    Fence(Sm80_generic, Sm80_generic)
                # for-k1 (K tiles) loop ends
                # TeX: end xgemm_Sm80_fence

                # TeX: begin xgemm_Sm80_fence[0]
                # TeX: color line xgemm_Sm80_fence[1]
                #   gg
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    # TeX: color line xgemm_Sm80_fence[1]
                    #   vv
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        # TeX: end xgemm_Sm80_fence[0]
                        # TeX: begin xgemm_Sm80_fence[1]
                        # TeX: summary
                        # Write out per-warp accumulators D to GMEM C
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                Sm80_mma_store_d_tf32(
                                    C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                    n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                    # TeX: color line xgemm_Sm80_fence[1]
                                    #      gg  vv
                                    D_rmem[mw, nw, m_seq,n_seq,:,:])
                                # TeX: color remark xgemm_Sm80_fence[1]
                                # ggggggggggggggggggggg  vvvvvvvvvvvvvvvvvvvv
                                # mw: $256 \mapsto 128$, nw: $128 \mapsto 32$
                        # TeX: end xgemm_Sm80_fence[1]
                # TeX: begin xgemm_Sm80_fence
                # End per-CTA code
                # TeX: end xgemm_Sm80_fence
# TeX: begin xgemm_Sm80_fence[0]
xgemm_Sm80_fence = simplify(xgemm_Sm80_fence)
# TeX: end xgemm_Sm80_fence[0]


# TeX: version xgemm_Sm80_mbarrier 3
# TeX: begin xgemm_Sm80_mbarrier[0]
RING = 3
LAG = 1
@proc
def xgemm_Sm80_mbarrier(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear,
                        B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0
    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # TeX: begin xgemm_Sm80_mbarrier[1]
                # Per CTA code
                # TeX: color line xgemm_Sm80_mbarrier
                #                                  bbbbbbbb             rrrrrrrr
                # Defines two queue barrier arrays +ringbar [front] and -ringbar [back]
                # TeX: remark xgemm_Sm80_mbarrier[1:]
                # We will use the front to guard against RAW hazards, and back to guard against WAR hazards.
                ringbar: barrier @ CudaMbarrier
                # TeX: color line *
                #            yyyy
                A_smem : f32[RING, M1, K0] @ CudaSmemLinear # Input tiles (double buffered)
                # TeX: color line *
                #            yyyy
                B_smem : f32[RING, K0, N1] @ CudaSmemLinear
                # Accumulator tiles
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD(16, 8)
                # TeX: end xgemm_Sm80_mbarrier[0]
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        # TeX: summary
                        # Each warp zeros out its accumulators
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                Sm80_mma_zero_d_tf32(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # TeX: begin xgemm_Sm80_mbarrier[0]
                # K tiles loop, ring buffered. LAG iteration delay between load and use.
                # Don't accum tile in first LAG-many iterations.
                # Don't load tile in last LAG-many iterations.
                for k1 in seq(0, K / K0 + LAG):
                    if k1 < K / K0:
                        with CudaAsync(Sm80_cp_async):
                            # Wait for ring buffer to be consumed;
                            # don't wait for first RING-many iterations
                            # TeX: color line xgemm_Sm80_mbarrier
                            #     rrrrrrrr
                            Await(-ringbar, Sm80_cp_async, ~RING)
                            # TeX: end xgemm_Sm80_mbarrier[0]
                            # TeX: summary
                            # Load A tile
                            for m1 in seq(0, M1 / 64):
                                for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                    for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                        Sm80_cp_async_f32(
                                        # TeX: color line xgemm_Sm80_mbarrier[1]
                                        #          yyyyyyyyy
                                            A_smem[k1 % RING, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                            A[m2 * M1 + m1 * 64 + m0,
                                              k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4],
                                            size=4)
                            # TeX: summary
                            # Load B tile
                            for k0_seq in seq(0, 4):
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        Sm80_cp_async_f32(
                                        # TeX: color line xgemm_Sm80_mbarrier[1]
                                        #          yyyyyyyyy
                                            B_smem[k1 % RING, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                            B[k1 * K0 + k0_seq * 4 + k0_par,
                                              n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4],
                                            size=4)
                            # TeX: begin xgemm_Sm80_mbarrier[0]
                            # TeX: color line xgemm_Sm80_mbarrier
                            #                           bbbbbbbb
                            Arrive(Sm80_cp_async, 1) >> +ringbar
                # TeX: end xgemm_Sm80_mbarrier[1]
                # TeX: begin xgemm_Sm80_mbarrier[2]
                # for-k1 (K tiles) loop continues
                    if k1 >= LAG:
                        # Wait for ring buffer to be filled
                        # TeX: color line xgemm_Sm80_mbarrier
                        #     bbbbbbbb
                        Await(+ringbar, cuda_in_order, ~0)
                        # TeX: end xgemm_Sm80_mbarrier[0]

                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB(8, MMA_K)
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                        # TeX: color line xgemm_Sm80_mbarrier[2]
                                        #                           yyyyyyyyyyyyyyyyy
                                                             B_smem[(k1 - LAG) % RING,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA(16, MMA_K)
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_a_tf32(A_rmem[k_seq,:,:],
                                        # TeX: color line xgemm_Sm80_mbarrier[2]
                                        #                           yyyyyyyyyyyyyyyyy
                                                             A_smem[(k1 - LAG) % RING,
                                                             mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # TeX: summary
                                    # Accumulate to tile owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                          A_rmem[k_seq,:,:],
                                                          B_rmem[k_seq,n_seq,:,:], K=MMA_K)
                        # TeX: begin xgemm_Sm80_mbarrier[0]
                        # Signal that it's safe to overwrite ring buffer entry
                        # TeX: color line xgemm_Sm80_mbarrier
                        #                           rrrrrrrr
                        Arrive(cuda_in_order, 1) >> -ringbar
                # for-k1 (K tiles) loop ends
                # TeX: end xgemm_Sm80_mbarrier[0]

                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        # TeX: summary
                        # Write out per-warp accumulators D to GMEM C
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                Sm80_mma_store_d_tf32(
                                    C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                    n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                    D_rmem[mw,nw,m_seq,n_seq,:,:])
                # TeX: begin xgemm_Sm80_mbarrier[0]
                # End per-CTA code
                # TeX: end xgemm_Sm80_mbarrier[0]
                # TeX: end xgemm_Sm80_mbarrier[2]
# TeX: begin xgemm_Sm80_mbarrier[0]
xgemm_Sm80_mbarrier = simplify(xgemm_Sm80_mbarrier)
# TeX: end xgemm_Sm80_mbarrier[0]
