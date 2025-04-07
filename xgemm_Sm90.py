from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

Mw = 64
Nw = 64

M1 = 128
N1 = 256  # Does not change gracefully

K0 = 16
MMA_K = 4

RING = 3


@proc
def xgemm_Sm90(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    A_tensorMap = A[:,:] @ Sm90_tensorMap(0, M1, K0)
    B_tensorMap = B[:,:] @ Sm90_tensorMap(0, K0, N1)

    with CudaDeviceFunction(blockDim = 384, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code
                ringbar: cuda_mbarrier

                # Tiles (double buffered)
                A_smem : f32[RING, M1, K0] @ CudaSmemLinear
                B_smem : f32[RING, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD
                with CudaWarps(0, 8):
                    for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                        for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                            for m_seq in seq(0, Mw/16):
                                for n_seq in seq(0, Nw/8):
                                    Sm80_mma_zero_d_tf32(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, double buffered
                for k1 in seq(0, K / K0):
                    if True:
                        with CudaWarps(8, 9):
                            with CudaAsync(tma_to_smem_async):
                                # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                                ReverseAwait(ringbar, tma_to_smem_async, RING)
                                Sm90_copy_tensor_to_smem_linear_2f32(A_smem[k1 % RING,:,:],
                                                                     A_tensorMap[m2 * M1 : m2 * M1 + M1,
                                                                                 k1 * K0 : k1 * K0 + K0],
                                                                     box0=M1, box1=K0)
                                Sm90_copy_tensor_to_smem_linear_2f32(B_smem[k1 % RING,:,:],
                                                                     B_tensorMap[k1 * K0 : k1 * K0 + K0,
                                                                                 n2 * N1 : n2 * N1 + N1],
                                                                     box0=K0, box1=N1)
                                Arrive(tma_to_smem_async, ringbar)
                    else:
                        with CudaWarps(8, 12):
                            # Producer warpgroup
                            with CudaAsync(Sm80_cp_async):
                                # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                                ReverseAwait(ringbar, Sm80_cp_async, RING)

                                # Load A tile
                                for m1 in seq(0, M1 / 32):
                                    for m0 in cuda_threads(0, 32, unit=4 * cuda_thread):
                                        for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                            Sm80_cp_async_f32(A_smem[k1 % RING, m1 * 32 + m0, 4 * k0 : 4 * k0 + 4],
                                                              A[m2 * M1 + m1 * 32 + m0,
                                                              k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4], size=4)

                                # Load B tile
                                for k0_seq in seq(0, 8):
                                    for k0_par in cuda_threads(0, 2, unit=64 * cuda_thread):
                                        for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                            Sm80_cp_async_f32(B_smem[k1 % RING, k0_seq * 2 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                              B[k1 * K0 + k0_seq * 2 + k0_par,
                                                              n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4], size=4)
                                Arrive(Sm80_cp_async, ringbar)
                            # end CudaAsync(Sm80_cp_async)

                    with CudaWarps(0, 8):
                        # Consumer warpgroup

                        # Wait for ring buffer to be filled
                        Await(ringbar, cuda_classic)

                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                                             B_smem[(k1) % RING,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_a_tf32(A_rmem[k_seq,:,:],
                                                             A_smem[(k1) % RING,
                                                             mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                          A_rmem[k_seq,:,:],
                                                          B_rmem[k_seq,n_seq,:,:], K=MMA_K)
                        # Signal that it's safe to overwrite ring buffer entry
                        ReverseArrive(cuda_classic, ringbar)
                # for-k1 (K tiles) loop ends

                # Write out accumulator
                with CudaWarps(0, 8):
                    for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                        for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                            for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    Sm80_mma_store_d_tf32(
                                        C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                        n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                        D_rmem[mw,nw,m_seq,n_seq,:,:])

                Fence(cuda_classic, cuda_classic)

xgemm_Sm90 = simplify(xgemm_Sm90)
