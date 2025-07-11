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
    smem_m = 128
    cluster_m = smem_m * 2
    smem_n = N
    smem_k = 32
    wg_m = smem_m // 2
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
        assert M % cluster_m == 0
        assert N % smem_n == 0
        assert K % smem_k == 0

        A_tensorMap = A[:,:] @ Sm90_tensorMap(128, smem_m, smem_k)
        B_tensorMap = B[:,:] @ Sm90_tensorMap(128, smem_n // 2, smem_k)

        with CudaDeviceFunction(clusterDim=2, warp_config=my_warp_config, blocks_per_sm=1):
            for m_task in cuda_tasks(0, M / cluster_m):
                for n_task in cuda_tasks(0, N / smem_n):
                    ringbar : barrier[2] @ CudaMbarrier
                    cg : barrier[2, 2] @ CudaCommitGroup
                    D_rmem : f32[2, 2, wg_m, wg_n] @ Sm90_RmemMatrixD(wg_m, wg_n)
                    A_smem : f32[2, ring, smem_m / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)
                    B_smem : f32[2, ring, smem_n / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)

                    for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                        with CudaWarps(name="consumer"):
                            for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                Sm90_zero_scale_d_f32(D_rmem[cta,wg,:,:], M=wg_m, N=wg_n)

                    for k_iter in seq(0, K/smem_k):  # This loop should be cut at 1
                        with CudaWarps(name="producer"):
                            # TMA producer warp
                            with CudaAsync(tma_to_smem_async):
                                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                                    Await(-ringbar[cta], cuda_temporal, ~(ring-1))
                                    Sm90_copy_tensor_to_smem_swizzled_2f32(
                                        A_smem[cta,k_iter % ring,:,:,:],
                                        A_tensorMap[(2*m_task + cta) * smem_m:((2*m_task + cta)+1)* smem_m, k_iter * smem_k:(k_iter+1) * smem_k],
                                        box0=smem_m, box1=smem_k) >> +ringbar[cta]
                                Sm90_tmp_cta_pair_copy_tensor_to_smem_swizzled_2f32(
                                    B_smem[:,k_iter % ring,:,:,:],
                                    B_tensorMap[n_task * smem_n:(n_task+1)* smem_n, k_iter * smem_k:(k_iter+1) * smem_k],
                                    size0=smem_n, size1=smem_k) >> +ringbar[:]
                                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                                    Arrive(cuda_temporal, 1) >> +ringbar[:] >> +ringbar[cta]

                        with CudaWarps(name="consumer"):
                            for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                                # Producer warpgroups
                                Await(ringbar[cta], wgmma_async, ~0)
                                for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    with CudaAsync(wgmma_async):
                                        Fence(wgmma_fence_1, wgmma_fence_2)
                                        for k_mma in seq(0, smem_k / wg_k):
                                            Sm90_mma_async_tf32(D_rmem[cta,wg,:,:],
                                                A_smem[cta,k_iter % ring,wg*8:wg*8+8,:,k_mma*8:k_mma*8+8],
                                                B_smem[cta,k_iter % ring,:,:,k_mma*8:k_mma*8+8], M=wg_m, N=wg_n)
                                        Arrive(wgmma_async, 1) >> cg[cta,wg]
                                    if k_iter >= 1:
                                        Await(cg[cta,wg], cuda_in_order, 1)
                                Arrive(cuda_in_order, 1) >> -ringbar[:] >> -ringbar[cta]

                    for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                        with CudaWarps(name="consumer"):
                            for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                Await(cg[cta,wg], cuda_in_order, 0)

                        with CudaWarps(name="consumer"):
                            for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                Sm90_mma_write_d_col_major_tf32(
                                    C[n_task * smem_n:(n_task+1) * smem_n,
                                      (2*m_task + cta) * smem_m + wg * wg_m : (2*m_task + cta) * smem_m + wg * wg_m + wg_m],
                                    D_rmem[cta,wg,:,:], M=wg_m, N=wg_n)

                    Fence(cuda_in_order, cuda_in_order)


    xgemm_Sm90_wgmma = simplify(xgemm_Sm90_wgmma)
    xgemm_Sm90_wgmma = cut_loop(xgemm_Sm90_wgmma, xgemm_Sm90_wgmma.find_loop("k_iter"), 1)
    xgemm_Sm90_wgmma = divide_loop(xgemm_Sm90_wgmma, "m_task", 4, ("m1_task", "m0_task"))
    c_n_task = xgemm_Sm90_wgmma.find_loop("n_task")
    xgemm_Sm90_wgmma = lift_scope(xgemm_Sm90_wgmma, c_n_task)
    xgemm_Sm90_wgmma = lift_scope(xgemm_Sm90_wgmma, c_n_task)
    print(xgemm_Sm90_wgmma)
    return rename(xgemm_Sm90_wgmma, f"xgemm_Sm90_wgmma_n{N}")
