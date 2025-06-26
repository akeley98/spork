# exo-GPU commit bf432f182c9fc4bf37276fac305951589b2dcaca

from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

def make_Sm90_gemm(N):
    M1 = 4
    smem_m = 256
    smem_n = N
    smem_k = 32
    wg_m = 128
    wg_n = N
    wg_k = 8
    ring = 4

    my_warp_config = [
        CudaWarpConfig("producer", 1, setmaxnreg_dec=40),
        CudaWarpConfig("unused", 3, setmaxnreg_dec=40),
        CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),
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
                    D_rmem : f32[2, wg_m, wg_n] @ Sm90_RmemMatrixD
                    A_smem : f32[ring, smem_m / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)
                    B_smem : f32[ring, smem_n / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)

                    with CudaWarps(name="consumer"):
                        for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                            Sm90_zero_scale_d_f32(wg_m, wg_n, D_rmem[wg,:,:])

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
                            with CudaWarps(1, 3, name="consumer"):
                                pass
                            # Producer warpgroups
                            Await(ringbar, wgmma_async, ~0)
                            for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                with CudaAsync(wgmma_async):
                                    Fence(wgmma_fence_1, wgmma_fence_2)
                                    for k_mma in seq(0, smem_k / wg_k):
                                        Sm90_mma_async_tf32(D_rmem[wg,:,:],
                                            A_smem[k_iter % ring,wg*16:wg*16+16,:,k_mma*8:k_mma*8+8],
                                            B_smem[k_iter % ring,:,:,k_mma*8:k_mma*8+8], M=wg_m, N=wg_n)
                                    Arrive(wgmma_async, cg[wg], 1)
                                if k_iter >= 1:
                                    Await(cg[wg], cuda_in_order, 1)
                            if k_iter >= 1:
                                ReverseArrive(cuda_in_order, ringbar, 1)

                    with CudaWarps(name="consumer"):
                        for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                            Await(cg[wg], cuda_in_order, 0)
                        ReverseArrive(cuda_in_order, ringbar, ~0)

                    with CudaWarps(name="consumer"):
                        for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                            Sm90_mma_write_d_col_major_tf32(
                                C[n_task * smem_n:(n_task+1) * smem_n,
                                  m_task * smem_m + wg * wg_m : m_task * smem_m + wg * wg_m + wg_m],
                                D_rmem[wg,:,:], M=wg_m, N=wg_n)

                    Fence(cuda_in_order, cuda_in_order)


    xgemm_Sm90_wgmma = simplify(xgemm_Sm90_wgmma)
    xgemm_Sm90_wgmma = cut_loop(xgemm_Sm90_wgmma, xgemm_Sm90_wgmma.find_loop("k_iter"), 1)
    xgemm_Sm90_wgmma = divide_loop(xgemm_Sm90_wgmma, "m_task", 4, ("m1_task", "m0_task"))
    c_n_task = xgemm_Sm90_wgmma.find_loop("n_task")
    xgemm_Sm90_wgmma = lift_scope(xgemm_Sm90_wgmma, c_n_task)
    xgemm_Sm90_wgmma = lift_scope(xgemm_Sm90_wgmma, c_n_task)
    print(xgemm_Sm90_wgmma)
    return rename(xgemm_Sm90_wgmma, f"xgemm_Sm90_wgmma_n{N}")
