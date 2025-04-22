def xgemm_Sm80_mbarrier(M: size, N: size, K: size,
                        A: f32[M, K] @ CudaGmemLinear,
                        B: f32[K, N] @ CudaGmemLinear,
                        C: f32[M, N] @ CudaGmemLinear):
    assert M % 192 == 0
    assert N % 256 == 0
    assert K % 16 == 0
    with CudaDeviceFunction(256, 1, 1):
        for m2 in cuda_tasks(0, M / 192):
            for n2 in cuda_tasks(0, N / 256):
                # Each CTA generates a (192, 256) tile of C
                ringbar: cuda_mbarrier
                # Ring buffer (3) of A, B tiles in SMEM
                A_smem: f32[3, 192, 16] @ CudaSmemLinear
                B_smem: f32[3, 16, 256] @ CudaSmemLinear
                # Each warp (in 2 × 4 grid) holds (6*16, 8*8) of C
                # We call the intermediate accumulator for C, D.
                D_rmem: f32[2, 4, 6, 8, 16, 8] @ Sm80_RmemMatrixD
                # Each thread zeros its accumulators.
                for mw in cuda_threads(0, 2, unit=4 * cuda_warp):
                    for nw in cuda_threads(0, 4, unit=cuda_warp):
                        for m_seq in seq(0, 6):
                            for n_seq in seq(0, 8):
                                Sm80_mma_zero_d_tf32(D_rmem[mw, nw, m_seq, n_seq, 0:16, 0:8])
                for k1 in seq(0, K / 16 + 1):
                    if k1 < K / 16:
                        # Async loads to fill one ring buffer entry each for A and B 
                        with CudaAsync(Sm80_cp_async):
                            ReverseAwait(ringbar, Sm80_cp_async, ~3)
                            for m1 in seq(0, 3):
                                for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                    for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                        Sm80_cp_async_f32(
                                            A_smem[k1 % 3, m0 + 64 * m1, 4 * k0:4 + 4 * k0],
                                            A[m0 + 64 * m1 + 192 * m2, 4 * k0 + 16 * k1:4 + 4 * k0 + 16 * k1],
                                            size=4)
                            for k0_seq in seq(0, 4):
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        Sm80_cp_async_f32(
                                            B_smem[k1 % 3, k0_par + 4 * k0_seq, 4 * n0:4 + 4 * n0],
                                            B[k0_par + 4 * k0_seq + 16 * k1, 4 * n0 + 256 * n2:4 + 4 * n0 + 256 * n2],
                                            size=4)
                            Arrive(Sm80_cp_async, ringbar, 1)
                    # k1 loop continues
                    if k1 >= 1:
                        # Accumulate from A, B SMEM tiles (1 k1-iteration delay between load and use, hence the k1 >= 1 check)
                        Await(ringbar, cuda_classic, ~0)
                        for mw in cuda_threads(0, 2, unit=4 * cuda_warp):
                            for nw in cuda_threads(0, 4, unit=cuda_warp):
                                # We aggressively re-use fragments of B by storing them in registers.
                                B_rmem: f32[4, 8, 4, 8] @ Sm80_RmemMatrixB
                                for n_seq in seq(0, 8, pragma_unroll=0):
                                    for k_seq in seq(0, 4, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(
                                            B_rmem[k_seq, n_seq, 0:4, 0:8],
                                            B_smem[(-1 + k1) % 3,
                                                   4 * k_seq:4 + 4 * k_seq,
                                                   8 * n_seq + 64 * nw:8 + 8 * n_seq + 64 * nw],
                                            K=4)

                                # Accumulate from B (registers) and A (loaded from SMEM just in time)
                                for m_seq in seq(0, 6, pragma_unroll=0):
                                    A_rmem: f32[4, 16, 4] @ Sm80_RmemMatrixA
                                    for k_seq in seq(0, 4, pragma_unroll=0):
                                        Sm80_mma_load_a_tf32(
                                            A_rmem[k_seq, 0:16, 0:4],
                                            A_smem[(-1 + k1) % 3,
                                                   16 * m_seq + 96 * mw:16 + 16 * m_seq + 96 * mw,
                                                   4 * k_seq:4 + 4 * k_seq],
                                            K=4)
                                    for n_seq in seq(0, 8, pragma_unroll=0):
                                        for k_seq in seq(0, 4, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw, nw, m_seq, n_seq, 0:16, 0:8],
                                                          A_rmem[k_seq, 0:16, 0:4],
                                                          B_rmem[k_seq, n_seq, 0:4, 0:8],
                                                          K=4)
                        ReverseArrive(cuda_classic, ringbar, 1)
                    # k1 loop ends
                # Each warp writes out its D accumulators to C.
                for mw in cuda_threads(0, 2, unit=4 * cuda_warp):
                    for nw in cuda_threads(0, 4, unit=cuda_warp):
                        for m_seq in seq(0, 6, pragma_unroll=0):
                            for n_seq in seq(0, 8, pragma_unroll=0):
                                Sm80_mma_store_d_tf32(
                                    C[16 * m_seq + 96 * mw + 192 * m2:16 + 16 * m_seq + 96 * mw + 192 * m2,
                                      8 * n_seq + 64 * nw + 256 * n2:8 + 8 * n_seq + 64 * nw + 256 * n2],
                                    D_rmem[mw, nw, m_seq, n_seq, 0:16, 0:8])
