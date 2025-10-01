from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

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
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD(16, 8)
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
                # for-k1 (K tiles) loop continues
                    if k1 > 0:
                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB(8, MMA_K)
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                                             B_smem[1 - k1 % 2,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA(16, MMA_K)
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_a_tf32(A_rmem[k_seq,:,0:MMA_K],
                                                             A_smem[1 - k1 % 2,
                                                             mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                          A_rmem[k_seq,:,0:MMA_K],
                                                          B_rmem[k_seq,n_seq,:,:], K=MMA_K)

                    # Sm80_generic sync-tl = (cuda_in_order | Sm80_cp_async)
                    Fence(Sm80_generic, Sm80_generic)

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
                raw: barrier @ CudaMbarrier
                war: barrier(raw) @ CudaMbarrier

                # Tiles (ring buffer)
                A_smem : f32[RING, M1 / 8, 8, K0] @ Sm90_SmemSwizzled(64)
                # A_smem : f32[RING, M1 / 8, 8, K0] @ CudaSmemLinear
                B_smem : f32[RING, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD(16, 8)
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                Sm80_mma_zero_d_tf32(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, ring buffered
                # Don't accum tile in first LAG-many iterations.
                # Don't load tile in last LAG-many iterations.
                # LAG iteration delay between load and use.
                for k1 in seq(0, K / K0 + LAG):
                    if k1 < K / K0:
                        # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                        Await(war, Sm80_cp_async, ~RING)

                        # Load A tile
                        for m1 in seq(0, M1 / 64):
                            for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                    Sm80_cp_async_f32(A_smem[k1 % RING,
                                                             (m1 * 64 + m0) / 8,
                                                             m0 % 8,
                                                             4 * k0 : 4 * k0 + 4],
                                                      A[m2 * M1 + m1 * 64 + m0,
                                                      k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4], size=4)

                        # Load B tile
                        for k0_seq in seq(0, 4):
                            for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                    Sm80_cp_async_f32(B_smem[k1 % RING, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                      B[k1 * K0 + k0_seq * 4 + k0_par,
                                                      n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4], size=4)
                        Arrive(Sm80_cp_async, 1) >> raw
                # for-k1 (K tiles) loop continues
                    if k1 >= LAG:
                        # Wait for ring buffer to be filled
                        Await(raw, cuda_in_order, ~0)

                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB(8, MMA_K)
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                                             B_smem[(k1 - LAG) % RING,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA(16, MMA_K)
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_a_divided_tf32(
                                            A_rmem[k_seq,:,:],
                                            A_smem[(k1 - LAG) % RING,
                                            (mw*Mw + m_seq*16) / 8 : (mw*Mw + m_seq*16) / 8 + 2,
                                            0:8,
                                            k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                          A_rmem[k_seq,:,:],
                                                          B_rmem[k_seq,n_seq,:,:], K=MMA_K)
                        # Signal that it's safe to overwrite ring buffer entry
                        Arrive(cuda_in_order, 1) >> war
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
                Fence(cuda_in_order, cuda_in_order)

xgemm_Sm80_mbarrier = simplify(xgemm_Sm80_mbarrier)
xgemm_Sm80_mbarrier.sync_check(M=M1 * 2, N=N1 * 2, K=K0 * 10)


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
                raw: barrier @ CudaMbarrier
                war: barrier(raw) @ CudaMbarrier

                # Tiles (double buffered)
                A_smem : f32[RING, M1, K0] @ CudaSmemLinear
                B_smem : f32[RING, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD(16, 8)
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
                        # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                        Await(war, Sm80_cp_async, ~RING)

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
                        Arrive(Sm80_cp_async, 1) >> raw

                    with CudaWarps(0, 8):
                        # Consumer warpgroup

                        # Wait for ring buffer to be filled
                        Await(raw, cuda_in_order, ~0)

                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB(8, MMA_K)
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                                             B_smem[(k1) % RING,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA(16, MMA_K)
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
                        Arrive(cuda_in_order, 1) >> war
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

                Fence(cuda_in_order, cuda_in_order)

xgemm_Sm80_split = simplify(xgemm_Sm80_split)


# Gemms for presentation

@proc
def gemm_one_per_thread(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=256):
        for m2 in cuda_tasks(0, M / 16):
            for n2 in cuda_tasks(0, N / 16):
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        accum: f32 @ CudaRmem
                        accum = 0
                        for k in seq(0, K):
                            accum += A[m2*16 + m1, k] * B[n2*16 + n1, k]
                        C[n2*16 + n1, m2*16 + m1] = accum

@proc
def gemm_tile_per_thread(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=256):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        accum: f32[8, 16] @ CudaRmem
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m0, n0] = 0
                        for k in seq(0, K):
                            for m0 in seq(0, 8):
                                for n0 in seq(0, 16):
                                    accum[m0, n0] += A[m2*128 + m1*8 + m0, k] * B[n2*256 + n1*16 + n0, k]
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m0, n0]

@proc
def CudaWarps_example():
    with CudaDeviceFunction(blockDim=384):
        for task in cuda_tasks(0, 3):
            for i in cuda_threads(0, 128):
                pass
            with CudaWarps(8, 12):
                for i in cuda_threads(0, 128):
                    pass

@proc
def gemm_tile_per_cta(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=256):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                accum: f32[16, 16, 8, 16] @ CudaRmem
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m1, n1, m0, n0] = 0
                        for k in seq(0, K):
                            for m0 in seq(0, 8):
                                for n0 in seq(0, 16):
                                    accum[m1, n1, m0, n0] += A[m2*128 + m1*8 + m0, k] * B[n2*256 + n1*16 + n0, k]
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m1, n1, m0, n0]


@proc
def gemm_simple_smem(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=256):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                A_smem: f32[128, 32] @ CudaSmemLinear
                B_smem: f32[256, 32] @ CudaSmemLinear
                accum: f32[16, 16, 8, 16] @ CudaRmem

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m1, n1, m0, n0] = 0

                for k1 in seq(0, K / 32):
                    for m1 in seq(0, 16):
                        for m0 in cuda_threads(0, 8, unit=32*cuda_thread):
                            for k0 in cuda_threads(0, 32):
                                A_smem[m1*8 + m0, k0] = A[m2*128 + m1*8 + m0, k1*32 + k0]
                    for n1 in seq(0, 32):
                        for n0 in cuda_threads(0, 8, unit=32*cuda_thread):
                            for k0 in cuda_threads(0, 32):
                                B_smem[n1*8 + n0, k0] = B[n2*256 + n1*8 + n0, k1*32 + k0]

                    Fence(cuda_in_order, cuda_in_order)

                    for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                        for n1 in cuda_threads(0, 16, unit=cuda_thread):
                            for k0 in seq(0, 32):
                                for m0 in seq(0, 8):
                                    for n0 in seq(0, 16):
                                        accum[m1, n1, m0, n0] += A_smem[m1*8 + m0, k0] * B_smem[n1*16 + n0, k0]

                    Fence(cuda_in_order, cuda_in_order)

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m1, n1, m0, n0]

