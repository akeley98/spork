# exo-GPU commit bf432f182c9fc4bf37276fac305951589b2dcaca

from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

def make_Sm90_gemm(N, M_CTA, N_CTA, tma_to_gmem=False, K_tasks=1):
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

    assert K_tasks > 0
    if K_tasks > 1:
        assert tma_to_gmem
    enable_split_k = K_tasks != 1

    smem_box_A = (1, smem_m // N_CTA, smem_k)
    smem_box_B = (1, smem_n // M_CTA, smem_k)  # M_CTA is not a typo
    smem_box_C = (1, smem_n, smem_m)

    @proc
    def xgemm_Sm90_wgmma(
        L: size, M: size, N: size, K: size,
        A: f32[L,M,K] @ CudaGmemLinear,
        B: f32[L,N,K] @ CudaGmemLinear,
        C: f32[L,N,M] @ CudaGmemLinear
    ):
        # assert M % cluster_m == 0
        # assert N % cluster_n == 0
        assert L > 0
        assert M > 0
        assert N > 0
        assert K > 0

        A_tensorMap = A[:,:,:] @ Sm90_tensorMap(128, *smem_box_A)
        B_tensorMap = B[:,:,:] @ Sm90_tensorMap(128, *smem_box_B)
        C_tensorMap = C[:,:,:] @ Sm90_tensorMap(0, *smem_box_C)

        if enable_split_k:
            cudaMemsetAsync0_3f32(L, N, M, C[:,:,:])

        with CudaDeviceFunction(clusterDim=M_CTA * N_CTA, warp_config=my_warp_config, blocks_per_sm=1):
          for batch in cuda_tasks(0, L):
            for k_task in cuda_tasks(0, K_tasks):
              for m_task in cuda_tasks(0, (M + cluster_m - 1) / cluster_m):
                for n_task in cuda_tasks(0, (N + cluster_n - 1) / cluster_n):
                    raw : barrier[M_CTA, N_CTA] @ CudaMbarrier
                    war : barrier(raw)[M_CTA, N_CTA] @ CudaMbarrier
                    cg : barrier[M_CTA, N_CTA, 2] @ CudaCommitGroup
                    D_rmem : f32[M_CTA, N_CTA, 2, wg_m, wg_n] @ Sm90_RmemMatrixD(wg_m, wg_n)
                    A_smem : f32[M_CTA, N_CTA, ring, smem_m / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)
                    B_smem : f32[M_CTA, N_CTA, ring, smem_n / 8, 8, smem_k] @ Sm90_SmemSwizzled(128)

                    for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                        for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Sm90_zero_scale_d_f32(D_rmem[m_cta,n_cta,wg,:,:], M=wg_m, N=wg_n)

                    # This loop should be cut at 1
                    for k_iter in seq(0, (K + smem_k * K_tasks - 1) / smem_k / K_tasks):
                        with CudaWarps(name="producer"):
                            # TMA producer warp
                            for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                                for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                                    Await(war[m_cta,n_cta], cuda_temporal, ~(ring-1))
                                Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
                                    A_smem[m_cta,:,k_iter % ring,:,:,:],
                                    A_tensorMap[
                                        batch,
                                        (M_CTA*m_task + m_cta) * smem_m:
                                        ((M_CTA*m_task + m_cta)+1) * smem_m,
                                        # XXX due to affine restriction for Exo, the split_K is backwards!!!
                                        # The k_task strides fast and k_iter strides slowly.
                                        # This may be suboptimal L2 cache usage!
                                        (k_iter * K_tasks + k_task) * smem_k:
                                        (k_iter * K_tasks + k_task) * smem_k + smem_k],
                                    n_cta=N_CTA, cta_stride=1, size0=smem_m, size1=smem_k, smem_box=smem_box_A
                                ) >> raw[m_cta,:]
                            for n_cta in cuda_threads(0, N_CTA, unit=M_CTA * cuda_warp_in_cluster_strided(N_CTA)):
                                Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
                                    B_smem[:,n_cta,k_iter % ring,:,:,:],
                                    B_tensorMap[
                                        batch,
                                        (N_CTA*n_task+n_cta) * smem_n:
                                        (N_CTA*n_task+n_cta+1) * smem_n,
                                        (k_iter * K_tasks + k_task) * smem_k:
                                        (k_iter * K_tasks + k_task) * smem_k + smem_k],
                                    n_cta=M_CTA, cta_stride=N_CTA, size0=smem_n, size1=smem_k, smem_box=smem_box_B
                                ) >> raw[:,n_cta]
                                for m_cta in cuda_threads(0, M_CTA, unit=cuda_cta_in_cluster):
                                    Arrive(cuda_temporal, 1) >> raw[m_cta,:] >> raw[:,n_cta]

                        with CudaWarps(name="consumer"):
                            for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                                for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                                    # Producer warpgroups
                                    Await(raw[m_cta,n_cta], cuda_generic_and_async_proxy, ~0)
                                    for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                        Fence(wgmma_fence_1, wgmma_fence_2)
                                        for k_mma in seq(0, smem_k / wg_k):
                                            Sm90_mma_async_tf32(D_rmem[m_cta,n_cta,wg,:,:],
                                                A_smem[m_cta,n_cta,k_iter % ring,wg*8:wg*8+8,:,k_mma*8:k_mma*8+8],
                                                B_smem[m_cta,n_cta,k_iter % ring,:,:,k_mma*8:k_mma*8+8], M=wg_m, N=wg_n)
                                        Arrive(wgmma_async, 1) >> cg[m_cta,n_cta,wg]
                                        if k_iter >= 1:
                                            Await(cg[m_cta,n_cta,wg], cuda_in_order, 1)
                                    Arrive(cuda_in_order, 1) >> war[m_cta,:] >> war[:,n_cta]

                    for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                        for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Await(cg[m_cta,n_cta,wg], cuda_in_order, 0)

                    if tma_to_gmem:
                        Fence(cuda_in_order, cuda_in_order)
                        for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                            for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                                C_smem: f32[smem_n, smem_m] @ CudaSmemLinear
                                with CudaWarps(name="consumer"):
                                    for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                        Sm90_mma_write_d_col_major_tf32(
                                            wg_m, wg_n, C_smem[:,wg * wg_m: wg * wg_m + wg_m],
                                            D_rmem[m_cta,n_cta,wg,:,:], M=wg_m, N=wg_n)
                                Fence(cuda_in_order, tma_to_gmem_async)
                                with CudaWarps(name="producer"):
                                    if enable_split_k:
                                        Sm90_reduce_tensor_to_gmem_linear_2f32(
                                            C_tensorMap[
                                                batch,
                                                (N_CTA*n_task + n_cta) * smem_n: (N_CTA*n_task + n_cta) * smem_n + smem_n,
                                                (M_CTA*m_task + m_cta) * smem_m: (M_CTA*m_task + m_cta) * smem_m + smem_m],
                                            C_smem[:, :],
                                            size0=smem_n, size1=smem_m, smem_box=smem_box_C,
                                        )
                                    else:
                                        Sm90_copy_tensor_to_gmem_linear_2f32(
                                            C_tensorMap[
                                                batch,
                                                (N_CTA*n_task + n_cta) * smem_n: (N_CTA*n_task + n_cta) * smem_n + smem_n,
                                                (M_CTA*m_task + m_cta) * smem_m: (M_CTA*m_task + m_cta) * smem_m + smem_m],
                                            C_smem[:, :],
                                            size0=smem_n, size1=smem_m, smem_box=smem_box_C,
                                        )
                                    tma_cg: barrier @ CudaCommitGroup
                                    Arrive(tma_to_gmem_async, 1) >> tma_cg
                                    Await(tma_cg, cuda_in_order, 0)
                                Fence(cuda_in_order, cuda_in_order)
                    else:
                        # Await(cg, cuda_in_order, 0) and write D_rmem -> C steps are fissioned.
                        # We must not arrive on the epilogue cluster sync until all wgmma retire.
                        cluster_sync: barrier @ CudaClusterSync
                        Arrive(cuda_in_order, 1) >> cluster_sync

                        for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                            for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                                with CudaWarps(name="consumer"):
                                    for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                                        Sm90_mma_write_d_col_major_tf32(
                                            M - ((M_CTA*m_task + m_cta) * smem_m + wg * wg_m),
                                            N - (N_CTA*n_task + n_cta) * smem_n,
                                            C[batch,
                                              (N_CTA*n_task + n_cta) * smem_n
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
    if tma_to_gmem:
        suffix = f"_tma_K{K_tasks}"
    else:
        suffix = ""
    xgemm_Sm90_wgmma = rename(xgemm_Sm90_wgmma, f"xgemm_Sm90_wgmma_n{N}{suffix}")
    xgemm_Sm90_wgmma.sync_check(L=2, M=500, N=800, K=256)
    return xgemm_Sm90_wgmma
