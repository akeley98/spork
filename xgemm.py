from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.spork.cuda_memory import *

Sm80_cp_async = actor_kinds.Sm80_cp_async  # Maybe crappy, fixme

Mw = 96
Nw = 64

M1 = 192
N1 = 256  # Does not change gracefully

K0 = 16

@instr("exo_Sm80_cpAsync16B(&{smem_data}, &{gmem_data});")
def tmp_cpAsync16B_f32(smem: [f32][4] @ CudaSmemLinear, gmem: [f32][4] @ CudaGmemLinear):
    assert stride(smem, 0) == 1
    for i in seq(0, 4):
        smem[i] = gmem[i]

@instr("exo_Sm80_tmp_load_a({rmem_data}, &{smem_data}, {smem_layout});")
def tmp_load_a(rmem: [f32][16,8] @ Sm80_RmemMatrixA, smem: [f32][16,8] @ CudaSmemLinear):
    for m in seq(0, 16):
        for k in seq(0, 8):
            rmem[m,k] = smem[m,k]

@instr("exo_Sm80_tmp_load_b({rmem_data}, &{smem_data}, {smem_layout});")
def tmp_load_b(rmem: [f32][8,8] @ Sm80_RmemMatrixB, smem: [f32][8,8] @ CudaSmemLinear):
    for k in seq(0, 8):
        for n in seq(0, 8):
            rmem[k,n] = smem[k,n]

@instr("exo_Sm80_tmp_mma({D_data}, {A_data}, {B_data});")
def tmp_mma(D: [f32][16,8] @ Sm80_RmemMatrixD, A: [f32][16,8] @ Sm80_RmemMatrixA, B: [f32][8,8] @ Sm80_RmemMatrixB):
    for m in seq(0, 16):
        for n in seq(0, 8):
            for k in seq(0, 8):
                D[m,n] += A[m,k] * B[k,n]

@instr("exo_Sm80_tmp_store_d(&{gmem_data}, {rmem_data}, {gmem_layout});")
def tmp_store_d(gmem: [f32][16,8] @ CudaDeviceVisibleLinear, rmem: [f32][16,8] @ Sm80_RmemMatrixD):
    for m in seq(0, 16):
        for n in seq(0, 8):
            gmem[m,n] = rmem[m,n]

@instr("exo_Sm80_tmp_zero_d({rmem_data});")
def tmp_zero_d(rmem: [f32][16,8] @ Sm80_RmemMatrixD):
    for m in seq(0, 16):
        for n in seq(0, 8):
            rmem[m,n] = 0

@proc
def xgemm_cuda_fence(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
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
                                tmp_zero_d(D_rmem[mw,nw,m_seq, n_seq,:,:])

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
                                        tmp_cpAsync16B_f32(A_smem[k1 % 2, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                           A[m2 * M1 + m1 * 64 + m0,
                                                             k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4])

                            # Load B tile
                            for k0_seq in seq(0, 4):
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        tmp_cpAsync16B_f32(B_smem[k1 % 2, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                           B[k1 * K0 + k0_seq * 4 + k0_par,
                                                             n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4])
                        # end CudaAsync(Sm80_cp_async)
                # for-k1 (K tiles) loop continues
                    if k1 > 0:
                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/8, Nw/8, 8, 8] @ Sm80_RmemMatrixB
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / 8, pragma_unroll=0):
                                        tmp_load_b(B_rmem[k_seq,n_seq,:,:],
                                                   B_smem[1 - k1 % 2,
                                                          k_seq*8:(k_seq+1)*8,
                                                          nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8])

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/8, 16, 8] @ Sm80_RmemMatrixA
                                    for k_seq in seq(0, K0 / 8, pragma_unroll=0):
                                        tmp_load_a(A_rmem[k_seq,:,:],
                                                   A_smem[1 - k1 % 2,
                                                          mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                          k_seq*8:(k_seq+1)*8])
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / 8, pragma_unroll=0):
                                            tmp_mma(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                    A_rmem[k_seq,:,:],
                                                    B_rmem[k_seq,n_seq,:,:])

                    # Sm80_generic actor kind = (cuda_classic | Sm80_cp_async)
                    # NB codegen=... is a hack, should be removed once barrier lowering is implemented.
                    Fence(Sm80_generic, Sm80_generic)
                # for-k1 (K tiles) loop ends

                # Write out accumulator
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                tmp_store_d(C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                              n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                            D_rmem[mw,nw,m_seq,n_seq,:,:])

xgemm_cuda_fence = simplify(xgemm_cuda_fence)

RING = 3
LAG = 1

@proc
def xgemm_cuda_mbarrier(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code
                ringbar: cuda_mbarrier

                # Tiles (double buffered)
                A_smem : f32[RING, M1, K0] @ CudaSmemLinear
                B_smem : f32[RING, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                tmp_zero_d(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, double buffered
                # Don't accum tile in first LAG-many iterations.
                # Don't load tile in last LAG-many iterations.
                # LAG iteration delay between load and use.
                for k1 in seq(0, K / K0 + LAG):
                    if k1 < K / K0:
                        with CudaAsync(Sm80_cp_async):
                            # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                            ReverseAwait(ringbar, Sm80_cp_async, RING)

                            # Load A tile
                            for m1 in seq(0, M1 / 64):
                                for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                    for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                        tmp_cpAsync16B_f32(A_smem[k1 % RING, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                           A[m2 * M1 + m1 * 64 + m0,
                                                             k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4])

                            # Load B tile
                            for k0_seq in seq(0, 4):
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        tmp_cpAsync16B_f32(B_smem[k1 % RING, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                           B[k1 * K0 + k0_seq * 4 + k0_par,
                                                             n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4])
                            Arrive(Sm80_cp_async, ringbar)
                        # end CudaAsync(Sm80_cp_async)
                # for-k1 (K tiles) loop continues
                    if k1 >= LAG:
                        # Wait for ring buffer to be filled
                        Await(ringbar, cuda_classic)

                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/8, Nw/8, 8, 8] @ Sm80_RmemMatrixB
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / 8, pragma_unroll=0):
                                        tmp_load_b(B_rmem[k_seq,n_seq,:,:],
                                                   B_smem[(k1 - LAG) % RING,
                                                          k_seq*8:(k_seq+1)*8,
                                                          nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8])

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/8, 16, 8] @ Sm80_RmemMatrixA
                                    for k_seq in seq(0, K0 / 8, pragma_unroll=0):
                                        tmp_load_a(A_rmem[k_seq,:,:],
                                                   A_smem[(k1 - LAG) % RING,
                                                          mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                          k_seq*8:(k_seq+1)*8])
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / 8, pragma_unroll=0):
                                            tmp_mma(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                    A_rmem[k_seq,:,:],
                                                    B_rmem[k_seq,n_seq,:,:])
                        # Signal that it's safe to overwrite ring buffer entry
                        ReverseArrive(cuda_classic, ringbar)
                # for-k1 (K tiles) loop ends

                # Write out accumulator
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                tmp_store_d(C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                              n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                            D_rmem[mw,nw,m_seq,n_seq,:,:])

xgemm_cuda_mbarrier = simplify(xgemm_cuda_mbarrier)

xgemm_cuda_mbarrier = rename(xgemm_cuda_mbarrier, "xgemm_cuda")

