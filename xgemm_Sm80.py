from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *

Mw = 96
Nw = 64

M1 = 192
N1 = 256  # Does not change gracefully

K0 = 16
MMA_K = 4


@proc
def xgemm_Sm80_fence(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code

                # Tiles (double buffered)
                A_smem : f32[2, M1, K0] @ CudaSmemLinear
                B_smem : f32[2, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                Sm80_mma_zero_d_tf32(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, double buffered
                # Don't accum tile in first iteration.
                # Don't load tile in last iteration.
                # 1 iteration delay between load and use.
                for k1 in seq(0, K / K0 + 1):
                    if k1 < K / K0:
                        with CudaAsync(Sm80_cp_async):
                            # Load A tile
                            for m1 in seq(0, M1 / 64):
                                for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                    for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                        Sm80_cp_async_f32(A_smem[k1 % 2, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                          A[m2 * M1 + m1 * 64 + m0,
                                                          k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4], size=4)

                            # Load B tile
                            for k0_seq in seq(0, 4):
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        Sm80_cp_async_f32(B_smem[k1 % 2, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                          B[k1 * K0 + k0_seq * 4 + k0_par,
                                                          n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4], size=4)
                        # end CudaAsync(Sm80_cp_async)
                # for-k1 (K tiles) loop continues
                    if k1 > 0:
                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                                             B_smem[1 - k1 % 2,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_a_tf32(A_rmem[k_seq,:,:],
                                                             A_smem[1 - k1 % 2,
                                                             mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                          A_rmem[k_seq,:,:],
                                                          B_rmem[k_seq,n_seq,:,:], K=MMA_K)

                    # Sm80_generic actor kind = (cuda_classic | Sm80_cp_async)
                    # Fence(Sm80_generic, Sm80_generic)
                    for tid in cuda_threads(0, 256):
                        cg : barrier @ CudaCommitGroup
                        Arrive(Sm80_cp_async, cg, 1)
                        Await(cg, cuda_classic, 0)
                        # Fence(Sm80_generic, Sm80_generic)
                    Fence(cuda_classic, Sm80_generic)

                # for-k1 (K tiles) loop ends

                # Write out accumulator
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                Sm80_mma_store_d_tf32(
                                    C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                    n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                    D_rmem[mw,nw,m_seq,n_seq,:,:])


xgemm_Sm80_fence = simplify(xgemm_Sm80_fence)

RING = 3
LAG = 1

@proc
def xgemm_Sm80_mbarrier(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code
                ringbar: barrier @ CudaMbarrier

                # Tiles (double buffered)
                A_smem : f32[RING, M1, K0] @ CudaSmemLinear
                B_smem : f32[RING, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                Sm80_mma_zero_d_tf32(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, double buffered
                # Don't accum tile in first LAG-many iterations.
                # Don't load tile in last LAG-many iterations.
                # LAG iteration delay between load and use.
                for k1 in seq(0, K / K0 + LAG):
                    if k1 < K / K0:
                        with CudaAsync(Sm80_cp_async):
                            # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                            ReverseAwait(ringbar, Sm80_cp_async, ~RING)

                            # Load A tile
                            for m1 in seq(0, M1 / 64):
                                for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                    for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                        Sm80_cp_async_f32(A_smem[k1 % RING, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                          A[m2 * M1 + m1 * 64 + m0,
                                                          k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4], size=4)

                            # Load B tile
                            for k0_seq in seq(0, 4):
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        Sm80_cp_async_f32(B_smem[k1 % RING, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                          B[k1 * K0 + k0_seq * 4 + k0_par,
                                                          n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4], size=4)
                            Arrive(Sm80_cp_async, ringbar, 1)
                        # end CudaAsync(Sm80_cp_async)
                # for-k1 (K tiles) loop continues
                    if k1 >= LAG:
                        # Wait for ring buffer to be filled
                        Await(ringbar, cuda_classic, ~0)

                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                                             B_smem[(k1 - LAG) % RING,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_a_tf32(A_rmem[k_seq,:,:],
                                                             A_smem[(k1 - LAG) % RING,
                                                             mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                          A_rmem[k_seq,:,:],
                                                          B_rmem[k_seq,n_seq,:,:], K=MMA_K)
                        # Signal that it's safe to overwrite ring buffer entry
                        ReverseArrive(cuda_classic, ringbar, 1)
                # for-k1 (K tiles) loop ends

                # Write out accumulator
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                Sm80_mma_store_d_tf32(
                                    C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                    n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                    D_rmem[mw,nw,m_seq,n_seq,:,:])

xgemm_Sm80_mbarrier = simplify(xgemm_Sm80_mbarrier)


Mw = 64
M1 = 128


@proc
def xgemm_Sm80_split(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    with CudaDeviceFunction(blockDim = 384, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code
                ringbar: barrier @ CudaMbarrier

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
                    # with CudaWarps(8, 12):
                    with CudaWarps(8, 12):
                        # Producer warpgroup
                        with CudaAsync(Sm80_cp_async):
                            # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                            ReverseAwait(ringbar, Sm80_cp_async, ~RING)

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
                            Arrive(Sm80_cp_async, ringbar, 1)
                        # end CudaAsync(Sm80_cp_async)

                    with CudaWarps(0, 8):
                        # Consumer warpgroup

                        # Wait for ring buffer to be filled
                        Await(ringbar, cuda_classic, ~0)

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
                        ReverseArrive(cuda_classic, ringbar, 1)
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

xgemm_Sm80_split = simplify(xgemm_Sm80_split)
