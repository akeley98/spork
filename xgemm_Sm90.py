# exo-GPU commit bf432f182c9fc4bf37276fac305951589b2dcaca

from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

def make_Sm90_gemm(N, M_CTA, N_CTA):
    M1 = 4
    smem_m = 128
    smem_n = N
    smem_k = 32
    cluster_m = smem_m * M_CTA
    cluster_n = smem_n * N_CTA
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
        assert N % cluster_n == 0
        assert K % smem_k == 0

        A_tensorMap = A[:,:] @ Sm90_tensorMap(128, smem_m // N_CTA, smem_k)
        B_tensorMap = B[:,:] @ Sm90_tensorMap(128, smem_n // M_CTA, smem_k)  # M_CTA is not a typo

        with CudaDeviceFunction(clusterDim=M_CTA * N_CTA, warp_config=my_warp_config, blocks_per_sm=1):
            for m_task in cuda_tasks(0, M / cluster_m):
                for n_task in cuda_tasks(0, N / cluster_n):
                    ringbar : barrier[M_CTA, N_CTA] @ CudaMbarrier
                    cg : barrier[M_CTA, N_CTA, 2] @ CudaCommitGroup
                    D_rmem : f32[M_CTA, N_CTA, 2, wg_m, wg_n] @ Sm90_RmemMatrixD(wg_m, wg_n)
                    A_smem : f32[M_CTA, N_CTA, ring, smem_m / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)
                    B_smem : f32[M_CTA, N_CTA, ring, smem_n / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)

                    for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                        for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Sm90_zero_scale_d_f32(D_rmem[m_cta,n_cta,wg,:,:], M=wg_m, N=wg_n)

                    for k_iter in seq(0, K/smem_k):  # This loop should be cut at 1
                        with CudaWarps(name="producer"):
                            # TMA producer warp
                            with CudaAsync(tma_to_smem_async):
                                for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                                    for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                                        Await(-ringbar[m_cta,n_cta], cuda_temporal, ~(ring-1))
                                    Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
                                        A_smem[m_cta,:,k_iter % ring,:,:,:],
                                        A_tensorMap[(M_CTA*m_task + m_cta) * smem_m:((M_CTA*m_task + m_cta)+1) * smem_m, k_iter * smem_k:(k_iter+1) * smem_k],
                                        n_cta=N_CTA, cta_stride=1, size0=smem_m, size1=smem_k
                                    ) >> +ringbar[m_cta,:]
                                for n_cta in cuda_threads(0, N_CTA, unit=M_CTA * cuda_warp_in_cluster_strided(N_CTA)):
                                    Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
                                        B_smem[:,n_cta,k_iter % ring,:,:,:],
                                        B_tensorMap[(N_CTA*n_task+n_cta) * smem_n:(N_CTA*n_task+n_cta+1) * smem_n, k_iter * smem_k:(k_iter+1) * smem_k],
                                        n_cta=M_CTA, cta_stride=N_CTA, size0=smem_n, size1=smem_k
                                    ) >> +ringbar[:,n_cta]
                                    for m_cta in cuda_threads(0, M_CTA, unit=cuda_cta_in_cluster):
                                        Arrive(cuda_temporal, 1) >> +ringbar[m_cta,:] >> +ringbar[:,n_cta]

                        with CudaWarps(name="consumer"):
                            for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                                for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                                    # Producer warpgroups
                                    Await(ringbar[m_cta,n_cta], wgmma_async, ~0)
                                    for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                        with CudaAsync(wgmma_async):
                                            Fence(wgmma_fence_1, wgmma_fence_2)
                                            for k_mma in seq(0, smem_k / wg_k):
                                                Sm90_mma_async_tf32(D_rmem[m_cta,n_cta,wg,:,:],
                                                    A_smem[m_cta,n_cta,k_iter % ring,wg*8:wg*8+8,:,k_mma*8:k_mma*8+8],
                                                    B_smem[m_cta,n_cta,k_iter % ring,:,:,k_mma*8:k_mma*8+8], M=wg_m, N=wg_n)
                                            Arrive(wgmma_async, 1) >> cg[m_cta,n_cta,wg]
                                        if k_iter >= 1:
                                            Await(cg[m_cta,n_cta,wg], cuda_in_order, 1)
                                    Arrive(cuda_in_order, 1) >> -ringbar[m_cta,:] >> -ringbar[:,n_cta]
                    
                    cluster_sync: barrier @ CudaClusterSync
                    Arrive(cuda_in_order, 1) >> cluster_sync

                    for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                        for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Await(cg[m_cta,n_cta,wg], cuda_in_order, 0)
                                    Sm90_mma_write_d_col_major_tf32(
                                        C [(N_CTA*n_task + n_cta) * smem_n
                                        : ((N_CTA*n_task + n_cta)+1) * smem_n,
                                          (M_CTA*m_task + m_cta) * smem_m + wg * wg_m
                                        : (M_CTA*m_task + m_cta) * smem_m + wg * wg_m + wg_m],
                                        D_rmem[m_cta,n_cta,wg,:,:], M=wg_m, N=wg_n)

                    Await(cluster_sync, cuda_in_order, 0)


    xgemm_Sm90_wgmma = simplify(xgemm_Sm90_wgmma)
    xgemm_Sm90_wgmma = cut_loop(xgemm_Sm90_wgmma, xgemm_Sm90_wgmma.find_loop("k_iter"), 1)
    xgemm_Sm90_wgmma = divide_loop(xgemm_Sm90_wgmma, "m_task", 4, ("m1_task", "m0_task"))
    c_n_task = xgemm_Sm90_wgmma.find_loop("n_task")
    xgemm_Sm90_wgmma = lift_scope(xgemm_Sm90_wgmma, c_n_task)
    xgemm_Sm90_wgmma = lift_scope(xgemm_Sm90_wgmma, c_n_task)
    return rename(xgemm_Sm90_wgmma, f"xgemm_Sm90_wgmma_n{N}")
