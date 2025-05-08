from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

smem_m = 256
smem_n = 128
smem_k = 32
wg_m = 128
wg_n = 128
wg_k = 8
ring = 4

my_warp_config = [
    CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),
    CudaWarpConfig("producer", 1, setmaxnreg_dec=40),
    CudaWarpConfig("unused", 3, setmaxnreg_dec=40),
]

@proc
def xgemm_Sm90_wgmma(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    assert M % smem_m == 0
    assert N % smem_n == 0
    assert K % smem_k == 0

    A_tensorMap = A[:,:] @ Sm90_tensorMap(128, smem_m, smem_k)
    B_tensorMap = B[:,:] @ Sm90_tensorMap(128, smem_n, smem_k)

    with CudaDeviceFunction(warp_config = my_warp_config, blocks_per_sm = 1):
        for m_task in cuda_tasks(0, M / smem_m):
            for n_task in cuda_tasks(0, N / smem_n):
                ringbar : barrier @ CudaMbarrier
                cg : barrier @ CudaCommitGroup
                D_rmem : f32[2, wg_m / 64, 64, wg_n] @ Sm90_RmemMatrixD
                A_smem : f32[ring, smem_m / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)
                B_smem : f32[ring, smem_n / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)

                with CudaWarps(name="consumer"):
                    for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                        for m_mma in seq(0, wg_m / 64):
                            Sm90_zero_scale_d_tf32(D_rmem[wg,m_mma,:,:], n=wg_n)

                for k_iter in seq(0, K/smem_k):  # This loop should be cut at 1
                    with CudaWarps(name="producer"):
                        # TMA producer warp
                        with CudaAsync(tma_to_smem_async):
                            ReverseAwait(ringbar, cuda_temporal, ~ring)
                            Sm90_copy_tensor_to_smem_swizzled_2f32(
                                A_smem[k_iter % ring,:,:,:],
                                A_tensorMap[m_task * smem_m:(m_task+1)* smem_m, k_iter * smem_k:(k_iter+1) * smem_k],
                                box0=smem_m, box1=smem_k)
                            Sm90_copy_tensor_to_smem_swizzled_2f32(
                                B_smem[k_iter % ring,:,:,:],
                                B_tensorMap[n_task * smem_n:(n_task+1)* smem_n, k_iter * smem_k:(k_iter+1) * smem_k],
                                box0=smem_n, box1=smem_k)
                            Arrive(tma_to_smem_async, ringbar, 1)

                    with CudaWarps(name="consumer"):
                        # Producer warpgroups
                        Await(ringbar, wgmma_async, ~0)
                        for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                            with CudaAsync(wgmma_async):
                                Fence(wgmma_fence_1, wgmma_fence_2)
                                for k_mma in seq(0, smem_k / wg_k):
                                    for m_mma in seq(0, wg_m / 64):
                                        Sm90_mma_async_tf32(D_rmem[wg,m_mma,:,:],
                                            A_smem[k_iter % ring,wg*16+m_mma*8:wg*16+m_mma*8+8,:,k_mma*8:k_mma*8+8],
                                            B_smem[k_iter % ring,:,:,k_mma*8:k_mma*8+8], n=wg_n)
                                Arrive(wgmma_async, cg[wg], 1)
                            if k_iter >= 1:
                                Await(cg[wg], cuda_classic, 1)
                        if k_iter >= 1:
                            ReverseArrive(cuda_classic, ringbar, 1)

                with CudaWarps(name="consumer"):
                    for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                        Await(cg[wg], cuda_classic, 0)
                    ReverseArrive(cuda_classic, ringbar, ~0)

                with CudaWarps(name="consumer"):
                    for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                        for m_mma in seq(0, wg_m / 64):
                            Sm90_mma_write_d_col_major_tf32(
                                C[n_task * smem_n:(n_task+1) * smem_n,
                                  m_task * smem_m + wg * wg_m + m_mma * 64 : m_task * smem_m + wg * wg_m + m_mma * 64 + 64],
                                D_rmem[wg,m_mma,:,:],
                                n=wg_n)

                Fence(cuda_classic, cuda_classic)


xgemm_Sm90_wgmma = simplify(xgemm_Sm90_wgmma)
xgemm_Sm90_wgmma = cut_loop(xgemm_Sm90_wgmma, xgemm_Sm90_wgmma.find_loop("k_iter"), 1)
# print(xgemm_Sm90_wgmma)

try:
    print(xgemm_Sm90_wgmma.find("Await(cg[wg], cuda_classic, 0)"))
except Exception as e:
    print(e)


# Gemms for presentation

RING = 4

@proc
def gemm_ring(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=384):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                A_smem: f32[RING, 128, 32] @ CudaSmemLinear
                B_smem: f32[RING, 256, 32] @ CudaSmemLinear
                accum: f32[16, 16, 8, 16] @ CudaRmem
                ringbar: barrier @ CudaMbarrier

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m1, n1, m0, n0] = 0

                for k1 in seq(0, K / 32):
                    with CudaWarps(8, 12):
                        ReverseAwait(ringbar, cuda_temporal, ~RING)
                        for m1 in seq(0, 32):
                            for m0 in cuda_threads(0, 4, unit=32*cuda_thread):
                                for k0 in cuda_threads(0, 32):
                                    A_smem[k1%RING, m1*4 + m0, k0] = A[m2*128 + m1*4 + m0, k1*32 + k0]
                        for n1 in seq(0, 64):
                            for n0 in cuda_threads(0, 4, unit=32*cuda_thread):
                                for k0 in cuda_threads(0, 32):
                                    B_smem[k1%RING, n1*4 + n0, k0] = B[n2*256 + n1*4 + n0, k1*32 + k0]
                        Arrive(cuda_classic, ringbar, 1)

                    with CudaWarps(0, 8):
                        Await(ringbar, cuda_classic, ~0)
                        for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                            for n1 in cuda_threads(0, 16, unit=cuda_thread):
                                for k0 in seq(0, 32):
                                    for m0 in seq(0, 8):
                                        for n0 in seq(0, 16):
                                            accum[m1, n1, m0, n0] += A_smem[k1%RING, m1*8 + m0, k0] * B_smem[k1%RING, n1*16 + n0, k0]
                        ReverseArrive(cuda_classic, ringbar, 1)

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m1, n1, m0, n0]


@proc
def gemm_tma(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):

    A_tensorMap = A[:,:] @ Sm90_tensorMap(0, 128, 32)
    B_tensorMap = B[:,:] @ Sm90_tensorMap(0, 256, 32)

    with CudaDeviceFunction(blockDim=384):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                A_smem: f32[RING, 128, 32] @ CudaSmemLinear
                B_smem: f32[RING, 256, 32] @ CudaSmemLinear
                accum: f32[16, 16, 8, 16] @ CudaRmem
                ringbar: barrier @ CudaMbarrier

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m1, n1, m0, n0] = 0

                for k1 in seq(0, K / 32):
                    with CudaWarps(8, 9):
                        with CudaAsync(tma_to_smem_async):
                            ReverseAwait(ringbar, cuda_temporal, ~RING)
                            Sm90_copy_tensor_to_smem_linear_2f32(A_smem[k1%RING,:,:], A_tensorMap[m2*128:(m2+1)*128, k1*32:(k1+1)*32], box0=128, box1=32)
                            Sm90_copy_tensor_to_smem_linear_2f32(B_smem[k1%RING,:,:], B_tensorMap[n2*256:(n2+1)*256, k1*32:(k1+1)*32], box0=256, box1=32)
                            Arrive(tma_to_smem_async, ringbar, 1)

                    with CudaWarps(0, 8):
                        Await(ringbar, cuda_classic, ~0)
                        for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                            for n1 in cuda_threads(0, 16, unit=cuda_thread):
                                for k0 in seq(0, 32):
                                    for m0 in seq(0, 8):
                                        for n0 in seq(0, 16):
                                            accum[m1, n1, m0, n0] += A_smem[k1%RING, m1*8 + m0, k0] * B_smem[k1%RING, n1*16 + n0, k0]
                        ReverseArrive(cuda_classic, ringbar, 1)

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m1, n1, m0, n0]

