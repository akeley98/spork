from __future__ import annotations

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm90 import *

# TeX: version intro 3
# TeX: version par 1

# TeX: begin intro
@proc
def gemm_one_per_thread(M: size, N: size, K: size,
                        # TeX: color line intro[1]
                        #             rrrrrrrrrrrrrr                   rrrrrr
                        A: f32[M,K] @ CudaGmemLinear,   # Row-major in global memory
                        # TeX: color line intro[1]
                        #             rrrrrrrrrrrrrr                      rrrrrr
                        B: f32[N,K] @ CudaGmemLinear,   # Column-major in global memory
                        # TeX: color line intro[1]
                        #             rrrrrrrrrrrrrr                      rrrrrr
                        C: f32[N,M] @ CudaGmemLinear):  # Column-major in global memory
    # TeX: remark! intro[0]
    # instr-tl = CPU here
    # TeX: remark! intro[0]
    # CUDA device function (``kernel'') opened with CudaDeviceFunction
    # TeX: color line intro[0] par
    #                                yyy
    with CudaDeviceFunction(blockDim=256):
        # TeX: remark! intro[0]
        # instr-tl = cuda_in_order_tl here
        # TeX: color line intro[0]
        #                           gg
        for m2 in cuda_tasks(0, M / 16):  # Distribute work (``tasks'') to CTAs
            # TeX: color line intro[0]
            #                           vv
            for n2 in cuda_tasks(0, N / 16):
                # TeX: begin par
                # TeX: remark! intro[2]
                # Collective unit = 256 threads here
                # TeX: color line intro[0] par
                #                                yyy                  gg   vv
                # Per-CTA code here; each CTA of 256 threads computes 16 x 16 output tile
                # TeX: color line intro[2] par
                #   rr                        ggggggggggggggggggggg
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color remark intro[2] par
                    #        yyy                              gg
                    # CTA of 256 threads splits into teams of 16 threads
                    # TeX: remark! intro[2]
                    # Collective unit = 16 threads here
                    # TeX: color line intro[2] par
                    #   bb                        vvvvvvvvvvvvvvvv
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: color remark intro[2] par
                        #                                vvvvvv
                        # Teams of 16 threads split into single threads
                        # TeX: remark! intro[2]
                        # Collective unit = 1 thread here
                        # Per-thread code: calculate one value of the output matrix
                        # TeX: color remark intro[1]
                        #                                                rrrrrrrr
                        # Accumulate one value of the output matrix in a register
                        # TeX: color line intro[1]
                        #            rrrrrrrr
                        accum: f32 @ CudaRmem
                        # TeX: end par
                        accum = 0
                        # TeX: remark intro[1]
                        # Sequentially accumulate dot(A[m,0:K], B[n,0:K])
                        for k in seq(0, K):
                            accum += A[m2*16 + m1, k] * B[n2*16 + n1, k]
                        # TeX: remark intro[1]
                        # Write accumulated value to output matrix, column major
                        C[n2*16 + n1, m2*16 + m1] = accum
# TeX: end intro

"""
# TeX: begin par
                # Lowered C++
                # TeX: color line *
                #       rrrrrrrrrrrr   gggggggggggggggggg
                if (int exo_16thr_m1 = (threadIdx.x / 16); 1) {
                  # TeX: color line *
                  #       bbbbbbbbbbb   vvvvvvvvvvvvvvvvvv
                  if (int exo_1thr_n1 = (threadIdx.x % 16); 1) {
                    float accum;
# TeX: end par
"""

# TeX: version tile 3

@proc
def gemm_tile_per_thread(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    # TeX: begin tile
    # TeX: color line tile[0]
    #                                yyy
    with CudaDeviceFunction(blockDim=256):
        # TeX: color line tile[0:2]
        #   gg                      ggg
        for m2 in cuda_tasks(0, M / 128):
            # TeX: color line tile[0:2]
            #   vv                      vvv
            for n2 in cuda_tasks(0, N / 256):
                # TeX: remark! tile[0]
                # Expanded tile size
                # TeX: color line tile[0]
                #             yyy                  ggg   vvv
                # Each CTA of 256 threads computes 128 x 256 output tile
                # TeX: remark! tile[2]
            # Parallel loops: splits thread collectives: 1 sub-collective assigned per ``iteration''
                # TeX: color line tile[1]
                #   gg
                # TeX: color line tile[2]
                #         rrrrrrrrrrrrr                            r
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color line tile[1]
                    #   vv
                    # TeX: color line tile[2]
                    #         rrrrrrrrrrrrr                       r
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: color remark! tile[0]
                        #                            bbbbbb                       rrrrrrrrr
                        # Each thread accumulates an 8 x 16 output matrix tile in registers
                        # TeX: color remark tile[0]
                        #           bbbbbbbbbbbbbbbbbbbb
                        # Lowers to float accum[8 * 16];
                        # TeX: color line tile[0]
                        #          bbbbb    rrrrrrrr
                        accum: f32[8, 16] @ CudaRmem
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m0, n0] = 0       # Clear accumulators to 0
                        # TeX: remark! tile[2]
                        # Sequential loops: same thread(s) performs all work
                        # TeX: color line tile[2]
                        #        bbbb    b
                        for k in seq(0, K):
                            # TeX: color line tile[1]
                            #                g
                            # TeX: color line tile[2]
                            #         bbbb    b
                            for m0 in seq(0, 8):
                                # TeX: color line tile[1]
                                #                vv
                                # TeX: color line tile[2]
                                #         bbbb     b
                                for n0 in seq(0, 16):
                                    # TeX: remark! tile[1]
                                    # Index expressions got more complicated due to work tiling
                                    # TeX: color line tile[1]
                                    #                      ggg      g
                                    accum[m0, n0] += (A[m2*128 + m1*8 + m0, k]
                                    # TeX: color line tile[1]
                                    #                       vvv      vv
                                                     * B[n2*256 + n1*16 + n0, k])
                        # Write 8 x 16 output matrix tile to main memory
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m0, n0]
    # TeX: end tile

# TeX: version gmem_remark 1

# TeX: begin gmem_remark
@proc
def gemm_tile_per_thread(M: size, N: size, K: size,
                         # TeX: color line *
                         #             rrrrrrrrrrrrrr                   rrrrrr
                         A: f32[M,K] @ CudaGmemLinear,   # Row-major in global memory
                         # TeX: color line *
                         #             rrrrrrrrrrrrrr                      rrrrrr
                         B: f32[N,K] @ CudaGmemLinear,   # Column-major in global memory
                         C: f32[N,M] @ CudaGmemLinear):  # Column-major in global memory
    with CudaDeviceFunction(blockDim=256):
# TeX: end gmem_remark
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                # TeX: summary
                # Distribute work over threads
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        accum: f32[8, 16] @ CudaRmem
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m0, n0] = 0
                        # TeX: begin gmem_remark
                        for k in seq(0, K):
                            for m0 in seq(0, 8):
                                for n0 in seq(0, 16):
                                    # TeX: remark *
                                    # Reading directly from global memory
                                    # TeX: remark! *
                                    # Inefficient
                                    # TeX: color line *
                                    #                 rr                     r
                                    accum[m0, n0] += (A[m2*128 + m1*8 + m0, k]
                                    # TeX: color line *
                                    #                  rr                      r
                                                     * B[n2*256 + n1*16 + n0, k])
                        # TeX: end gmem_remark
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m0, n0]



# TeX: version dmem 5

@proc
def gemm_tile_per_cta(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
# TeX: begin dmem[0]
    with CudaDeviceFunction(blockDim=256):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
# TeX: end dmem[0]
# TeX: begin dmem
                # TeX: remark! dmem[0] dmem[4]
                # Tensor allocated by 256 threads (``alloc unit'')
                # TeX: remark! dmem[0] dmem[4]
                # CudaRmem requires allocation by 1 thread (``native unit'')
                # TeX: remark! dmem[1]
                # Distributed memory: distributed ``slices'' allocated per-thread
                # TeX: color line dmem[0]
               #rrrrr                       rrrrrrrr
                # TeX: color line dmem[1] dmem[4]
               #rrrrr              bbbbb    rrrrrrrr
                # TeX: remark! dmem[2:4]
                # Distribution deduced from later usage pattern
                # TeX: color remark dmem[2:]
                # ggggggggggggggggggg
                # m1: 256->16 threads
                # TeX: color remark dmem[3:]
                # vvvvvvvvvvvvvvvv
                # n1: 16->1 thread
                # TeX: color line dmem[2]
               #rrrrr      gg               rrrrrrrr
                # TeX: color line dmem[3]
               #rrrrr      gg  vv           rrrrrrrr
                # TeX: remark! dmem[4]
                # Goal met: remaining dimensions are not distributed
                accum: f32[16, 16, 8, 16] @ CudaRmem
                # TeX: remark dmem[1:]
                # collective unit = 256 threads here
                # TeX: color line dmem[2:4]
                #   gg                             gggggggggggggggg
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: remark dmem[1:]
                    # collective unit = 16 threads here
                    # TeX: color line dmem[3]
                    #   vv                             vvvvvvvvvvv
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: remark dmem[1:]
                        # collective unit = 1 thread here
# TeX: end dmem
# TeX: begin dmem[:4]
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                # TeX: color line dmem[2]
                                #     gg
                                # TeX: color line dmem[3]
                                #     gg  vv
                                accum[m1, n1, m0, n0] = 0
                        for k in seq(0, K):
                            # TeX: summary
                            # Each thread accumulates to accum[m1, n1, :, :]
                            for m0 in seq(0, 8):
                                for n0 in seq(0, 16):
                                    # TeX: color line dmem[2]
                                    #     gg
                                    # TeX: color line dmem[3]
                                    #     gg  vv
                            # TeX: begin dmem
                                    accum[m1, n1, m0, n0] += A[m2*128 + m1*8 + m0, k] \
                                                             * B[n2*256 + n1*16 + n0, k]
                            # TeX: end dmem
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = (
                                    # TeX: color line dmem[2]
                                    #     gg
                                    # TeX: color line dmem[3]
                                    #     gg  vv
                                    accum[m1, n1, m0, n0])
# TeX: end dmem[:4]
"""
# TeX: begin dmem[4]
                # Deduced assignments
                # TeX: color line *
                #     g  v  bbbb
                accum[0, 0, :, :] -> threadIdx.x = 0
                # TeX: color line *
                #     g  v  bbbb
                accum[0, 1, :, :] -> threadIdx.x = 1
                # TeX: color line *
                #     g  v  bbbb
                accum[0, 2, :, :] -> threadIdx.x = 2
                ...
                # TeX: color line *
                #     g  v  bbbb
                accum[1, 0, :, :] -> threadIdx.x = 16
                ...
                # TeX: color line *
                #     gg  vv  bbbb
                accum[15, 15, :, :] -> threadIdx.x = 255
# TeX: end dmem[4]
"""


# TeX: version working_smem 3

@proc
def gemm_simple_smem(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=256):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                # TeX: begin working_smem
                # TeX: remark! working_smem[0]
                # Per-CTA allocations
                # TeX: color remark working_smem[0]
                # yyyy
                # SMEM allocations: not distributed, as SMEM expects to be allocated by a CTA
                # TeX: color line *
                #                      yyyyyyyyyyyyyy
                A_smem: f32[128, 32] @ CudaSmemLinear
                # TeX: color line *
                #                      yyyyyyyyyyyyyy
                B_smem: f32[256, 32] @ CudaSmemLinear
                # TeX: color remark working_smem[0]
                # rrrr
                # RMEM allocation: distributed, as registers are allocated per-thread
                # TeX: color line *
                #          gg  vv           rrrrrrrr
                accum: f32[16, 16, 8, 16] @ CudaRmem
                # TeX: end working_smem

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                # TeX: summary
                                # Zero per-thread accumulators
                                accum[m1, n1, m0, n0] = 0

                # TeX: begin working_smem
                # TeX: color line *
                #   bb
                for k1 in seq(0, K / 32):
                    # TeX: end working_smem
                    # TeX: begin working_smem[1]
                    for m1 in seq(0, 16):
                        for m0 in cuda_threads(0, 8, unit=32*cuda_thread):
                            for k0 in cuda_threads(0, 32):
                                # TeX: summary!
                                # Load A_smem
                                # TeX: color line *
                               #yyyyyyy             y
                                A_smem[m1*8 + m0, k0] = A[m2*128 + m1*8 + m0, k1*32 + k0]
                    for n1 in seq(0, 32):
                        for n0 in cuda_threads(0, 8, unit=32*cuda_thread):
                            for k0 in cuda_threads(0, 32):
                                # TeX: summary!
                                # Load B_smem
                                # TeX: color line *
                               #yyyyyyy             y
                                B_smem[n1*8 + n0, k0] = B[n2*256 + n1*8 + n0, k1*32 + k0]
                    # TeX: end working_smem[1]

                    # TeX: begin working_smem
                    Fence(cuda_in_order, cuda_in_order)  # __syncthreads()
                    # TeX: end working_smem

                    # TeX: begin working_smem
                    # TeX: color line *
                    #   gg
                    for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                        # TeX: color line *
                        #   vv
                        for n1 in cuda_threads(0, 16, unit=cuda_thread):
                            # TeX: end working_smem
                            # TeX: begin working_smem[2]
                            for k0 in seq(0, 32):
                                # TeX: summary!
                                # accum += A_smem @ B_smem
                                for m0 in seq(0, 8):
                                    for n0 in seq(0, 16):
                                        # TeX: color line *
                                       #rrrrrrgg  vv        r     yyyyyyy             y
                                        accum[m1, n1, m0, n0] += (A_smem[m1*8 + m0, k0] 
                                        # TeX: color line *
                                        #                              yyyyyyy              y
                                                                     * B_smem[n1*16 + n0, k0])
                            # TeX: end working_smem[2]

                    # TeX: begin working_smem
                    Fence(cuda_in_order, cuda_in_order)  # __syncthreads()
                # TeX: color line *
                #     bb
                # End k1 loop
                # TeX: end working_smem
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                # TeX: summary
                                # Each thread writes out accumulators
# TeX: begin working_smem
# TeX: color line *
#                                                                            rrrrrrgg  vv        r
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m1, n1, m0, n0]
# TeX: end working_smem

# TeX: version ring 4
# TeX: version sync_copy 1

@proc
def gemm_ring(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    # TeX: begin ring[0]
    # TeX: color line ring[0]
    #                                yyy                   yyyyyyyy
    with CudaDeviceFunction(blockDim=384):  # We added 128 producer threads (4 warps)
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                # TeX: end ring[0]
                # TeX: begin ring sync_copy
                # TeX: color line *
                #           r
                A_smem: f32[4, 128, 32] @ CudaSmemLinear  # Added SMEM dimension:
                # TeX: color line *
                #           r                               rrrrrr
                B_smem: f32[4, 256, 32] @ CudaSmemLinear  # 4-deep ring buffer of 2D tiles
                accum: f32[16, 16, 8, 16] @ CudaRmem
                # TeX: color line ring[:3]
               #vvvvvvv            vvvvvvvvvvvv
                ringbar: barrier @ CudaMbarrier
                # TeX: end ring sync_copy

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                # TeX: summary
                                # Zero per-thread accumulators
                                accum[m1, n1, m0, n0] = 0

                # TeX: begin ring sync_copy
                # TeX: color line *
                #   bb
                for k1 in seq(0, K / 32):
                    # TeX: color line ring[:4]
                    #    yyyyyyyyyyyyyyyy     yyyyyyyyy
                    with CudaWarps(8, 12):  # Producer: threads [256, 383]
                # TeX: end ring sync_copy
                # TeX: begin ring[:2]
                        # TeX: color line *
                        #                         r
                        # Wait for ReverseArrive, 4 iteration delay
                # TeX: end ring[:2]
                # TeX: begin ring
                        # TeX: color line ring[:3]
                        #            vvvvvvv                 rr
                        ReverseAwait(ringbar, cuda_temporal, ~4)
                # TeX: end ring
                # TeX: begin ring[1]
                        for m1 in seq(0, 32):
                            # TeX: begin ring[3]
                            # TeX: remark! ring[3]
                            # We want to use TMA here
                            # TeX: end ring[3]
                            # TeX: summary!
                            # Fill A_smem[k1 % 4]
                            # TeX: begin sync_copy
                            for m0 in cuda_threads(0, 4, unit=32*cuda_thread):
                                for k0 in cuda_threads(0, 32):
                                    # TeX: remark! sync_copy
                                    # Problem: this is not really hiding copy latency
                                    # TeX: color line *
                                    #      rrrr
                                    A_smem[k1%4, m1*4 + m0, k0] = A[m2*128 + m1*4 + m0, k1*32 + k0]
                                    # TeX: end ring[1]
                                    # TeX: color remark! sync_copy
                                    #                                 rrrrrrrr
                                    # Reality: copy goes from GMEM to register to SMEM
                                    # TeX: color line sync_copy
                                    #          rrrrrrrr
                                    tmp: f32 @ CudaRmem
                                    tmp = A[m2*128 + m1*4 + m0, k1*32 + k0]
                                    A_smem[k1%4, m1*4 + m0, k0] = tmp
                                    # TeX: remark! sync_copy
                                    # We need to use TMA to copy from GMEM to SMEM directly
                                    # TeX: begin ring[1]
                            # TeX: end sync_copy
                        for n1 in seq(0, 64):
                            # TeX: summary!
                            # Fill B_smem[k1 % 4]
                            for n0 in cuda_threads(0, 4, unit=32*cuda_thread):
                                for k0 in cuda_threads(0, 32):
                                    # TeX: color line *
                                    #      rrrr
                                    B_smem[k1%4, n1*4 + n0, k0] = B[n2*256 + n1*4 + n0, k1*32 + k0]
                # TeX: end ring[1]
                        # TeX: begin ring
                        # TeX: color line ring[:3]
                        #                     vvvvvvv
                        Arrive(cuda_in_order, ringbar, 1)
                        # TeX: end ring

                    # TeX: begin ring sync_copy
                    # TeX: color line ring[:4]
                    #    ggggggggggggggg     ggggggggg
                    with CudaWarps(0, 8):  # Consumer: threads [0, 255]
                    # TeX: end ring sync_copy
                    # TeX: begin ring[0] ring[2] ring[3]
                        # TeX: color line ring[:3]
                        #     vvvvvvv
                        Await(ringbar, cuda_in_order, ~0)  # Wait for Arrive, 0 iteration delay
                    # TeX: end ring[0] ring[3]
                        for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                            for n1 in cuda_threads(0, 16, unit=cuda_thread):
                                # TeX: summary!
                                # accum += A_smem[k1 % 4] @ B_smem[k1 % 4]
                                for k0 in seq(0, 32):
                                    for m0 in seq(0, 8):
                                        for n0 in seq(0, 16):
                                            # TeX: color line *
                                            #                                rrrr
                                            accum[m1, n1, m0, n0] += (A_smem[k1%4, m1*8 + m0, k0]
                                            # TeX: color line *
                                            #                                 rrrr
                                                                     * B_smem[k1%4, n1*16 + n0, k0])
                        # TeX: begin ring[0] ring[3]
                        # TeX: color line ring[:3]
                        #                            vvvvvvv
                        ReverseArrive(cuda_in_order, ringbar, 1)
                        # TeX: end ring[0] ring[2] ring[3]
                # TeX: begin ring sync_copy
                # TeX: color line *
                #     bb
                # End k1 loop
                # TeX: end ring sync_copy

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                # TeX: summary
                                # Threads write out accumulators
                                # TeX: begin ring[0] ring[3]
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m1, n1, m0, n0]
                                # TeX: end ring[0] ring[3]


# TeX: version wgmma 1
# TeX: version prep_tma 4
# TeX: version tma 4

# TeX: begin tma[2:]
@proc
def gemm_tma(M: size, N: size, K: size,
             A: f32[M,K] @ CudaGmemLinear,
             B: f32[N,K] @ CudaGmemLinear,
             C: f32[N,M] @ CudaGmemLinear):
    # TeX: color remark! tma[2]
    #                                            vvvvvvvvvvvvvvvvvv
    # tensorMap constructed in CPU code (outside CudaDeviceFunction)
    # TeX: remark tma[2]
    # ``window'' (alias) to global memory A and B
    # TeX: color remark! tma[3]
    #                                     vvvvvvvvvvvvv
    # Window annotated with parameterized SpecialWindow type
    # TeX: color line tma[2]
   #yyyyyyyyyyy
    # TeX: color line tma[3]
   #yyyyyyyyyyy                              vvvvvvv
    A_tensorMap = A[:,:] @ Sm90_tensorMap(0, 128, 32)  # (swizzle, tile M/N, tile K)
    # TeX: color line tma[2]
   #yyyyyyyyyyy
    # TeX: color line tma[3]
   #yyyyyyyyyyy                              vvvvvvv
    B_tensorMap = B[:,:] @ Sm90_tensorMap(0, 256, 32)  # 0 means no swizzling (ignore this)
    # TeX: color line tma[2]
    #    vvvvvvvvvvvvvvvvvv
    with CudaDeviceFunction(blockDim=384):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
# TeX: end tma[2:]
                # TeX: begin tma[:2] prep_tma wgmma
                # TeX: color line *
                #           r
                A_smem: f32[4, 128, 32] @ CudaSmemLinear
                # TeX: color line *
                #           r                               rrrrrr
                B_smem: f32[4, 256, 32] @ CudaSmemLinear  # 4-deep ring buffer of 2D tiles
                accum: f32[16, 16, 8, 16] @ CudaRmem
                ringbar: barrier @ CudaMbarrier
                # TeX: end tma[:2] prep_tma wgmma

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                # TeX: summary
                                # Zero per-thread accumulators
                                accum[m1, n1, m0, n0] = 0

                # TeX: begin tma prep_tma wgmma
                # TeX: color line *
                #   bb
                for k1 in seq(0, K / 32):
                    with CudaWarps(8, 9):  # Producer
                        # TeX: remark! prep_tma[0]
                        # Wrap producer code in CudaAsync block. Informs the compiler that the
                        # TeX: color remark! prep_tma[0]
                        #                                               yyyyyyyyyyyyyyyyyyyyyyy
                        # body will contain only instrs with instr-tl = tma_to_smem_async_instr
                        # TeX: color line prep_tma[:2]
                        #    yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
                        with CudaAsync(tma_to_smem_async_instr):
                            # TeX: remark! prep_tma[3]
                            # Write-after-read (WAR) hazard: Producer will not issue instructions
                            # TeX: color remark! prep_tma[3]
                            #                vvvvvvvvvvvv                    b bb
                            # until consumer cuda_in_order instructions from 4 k1 iterations ago finish.
                            # TeX: color remark prep_tma[3]
                            #  rrrrrrrrrrrrr
                            # (cuda_temporal sync-tl avoids memory fences; redundant for WAR hazard)
                            # TeX: color line *
                            #                                    rr
                            # TeX: color line prep_tma[3]
                            #                     rrrrrrrrrrrrr  bb
                            ReverseAwait(ringbar, cuda_temporal, ~4)
                            # TeX: end tma prep_tma wgmma
                            # TeX: begin tma[0]
                            # TeX: summary!
                            # TMA instrs: fill A_smem[k1 % 4], B_smem[k1 % 4]
                            # TeX: begin tma[1:4]
                            # TeX: remark! tma[1]
                            # What are these tensorMaps?
                            # TeX: color remark! tma[3]
                            #     yyyyyyyyy                vvvvv
# TMA instructions require opaque tensorMap blob to encode sizes, etc., of the copied tile
                            # TeX: color line *
                            #                                           rrrr
                            Sm90_copy_tensor_to_smem_linear_2f32(A_smem[k1%4,:,:],
                            # TeX: color line tma[1:4]
            #       yyyyyyyyyyy
                    A_tensorMap[m2*128:(m2+1)*128, k1*32:(k1+1)*32], box0=128, box1=32)
                            # TeX: color line *
                            #                                           rrrr
                            Sm90_copy_tensor_to_smem_linear_2f32(B_smem[k1%4,:,:],
                            # TeX: color line tma[1:4]
            #       yyyyyyyyyyy
                    B_tensorMap[n2*256:(n2+1)*256, k1*32:(k1+1)*32], box0=256, box1=32)
                            # TeX: end tma[0:4]
                            # TeX: begin tma prep_tma wgmma
                            # TeX: remark! prep_tma[1]
                            # Change Arrive sync-tl
                            # TeX: color line prep_tma[1:4]
                            #      yyyyyyyyyyyyyyyyy
                            Arrive(tma_to_smem_async, ringbar, 1)
                            # TeX: end tma prep_tma wgmma

                    # TeX: begin tma[:2] prep_tma wgmma
                    with CudaWarps(0, 8):  # Consumer
                    # TeX: end wgmma
                        # TeX: color remark! prep_tma[2]
                        #          vvvvvvvvvvvvv                                yyyyyyyyyyyyyyyyy
                        # Consumer cuda_in_order instructions wait for producer tma_to_smem_async
                        # TeX: color remark! prep_tma[2]
                        #                            bbbb bb
                        # instructions issued in the same k1 iteration.
                        # TeX: color line prep_tma[2]
                        #              vvvvvvvvvvvvv  bb
                        Await(ringbar, cuda_in_order, ~0)
                    # TeX: end tma[:2] prep_tma
                        if False:  # Don't compile missing wgmma code
                         # TeX: begin wgmma
                         # TeX: color line wgmma
                         #              ggggggggggg
                         Await(ringbar, wgmma_async, ~0)
                         # TeX: color line wgmma
                         #              ggggggggggggggggg
                         with CudaAsync(wgmma_async_instr):
                                 # TeX: remark! wgmma
                                 # Next step: fill in wgmma (async tensor core) instructions here
                                 # TeX: remark! wgmma
                                 # This is way too much to fit on these slides, so this is the end of the talk.
                         # TeX: end wgmma
                             pass

                        for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                            for n1 in cuda_threads(0, 16, unit=cuda_thread):
                                # TeX: summary!
                                # accum += A_smem[k1 % 4] @ B_smem[k1 % 4]
                                for k0 in seq(0, 32):
                                    for m0 in seq(0, 8):
                                        for n0 in seq(0, 16):
                                            accum[m1, n1, m0, n0] += A_smem[k1%4, m1*8 + m0, k0] * B_smem[k1%4, n1*16 + n0, k0]
                                # TeX: end prep_tma[4]
                        # TeX: begin tma[:2] prep_tma wgmma
                        # TeX: color line prep_tma[3]
                        #             vvvvvvvvvvvvv
                        ReverseArrive(cuda_in_order, ringbar, 1)
                # TeX: color line *
                #     bb
                # End k1 loop
                # TeX: end tma[:2] prep_tma wgmma

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                # TeX: summary
                                # Threads write out accumulators
                                # TeX: begin tma[0] prep_tma
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m1, n1, m0, n0]
                                # TeX: end tma[0] prep_tma


gemm_tma = simplify(gemm_tma)
