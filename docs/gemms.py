# TeX: version intro 3

# TeX: begin intro
@proc
def gemm_one_per_thread(M: size, N: size, K: size,
                        A: f32[M,K] @ CudaGmemLinear,   # Row-major in global memory
                        B: f32[N,K] @ CudaGmemLinear,   # Column-major in global memory
                        C: f32[N,M] @ CudaGmemLinear):  # Column-major in global memory
    # TeX: color line *
    #                                yyy
    with CudaDeviceFunction(blockDim=256):
        # TeX: color line intro[0]
        #                           gg
        for m2 in cuda_tasks(0, M / 16):
            # TeX: color line intro[0]
            #                           vv
            for n2 in cuda_tasks(0, N / 16):
                # TeX: color line intro[0]
                #             yyy                  gg   vv
                # Each CTA of 256 threads computes 16 x 16 output tile
                # TeX: color line intro[2]
                #                             vvvvvvvvvvvvvvvvvvvvv
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color remark intro[2]
                    #        yyy                              vv
                    # CTA of 256 threads splits into teams of 16 threads
                    # TeX: color line intro[2]
                    #                             gggggggggggggggg
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: color remark intro[2]
                        #                                gggggg
                        # Teams of 16 threads split into single threads
                        # Per-thread code
                        # TeX: remark intro[0]
                        # Each thread calculates one value of the output matrix
                        # TeX: color remark intro[1]
                        #                                                bbbbbbbb
                        # Accumulate one value of the output matrix in a register
                        # TeX: color line intro[1]
                        #            bbbbbbbb
                        accum: f32 @ CudaRmem
                        accum = 0
                        # TeX: remark intro[1]
                        # Sequentially accumulate dot(A[m,0:K], B[n,0:K])
                        for k in seq(0, K):
                            accum += A[m2*16 + m1, k] * B[n2*16 + n1, k]
                        # TeX: remark intro[1]
                        # Write accumulated value to output matrix, column major
                        C[n2*16 + n1, m2*16 + m1] = accum
# TeX: end intro

# TeX: version tile 2

@proc
def gemm_tile_per_thread(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    # TeX: begin tile
    # TeX: color line tile[0]
    #                                yyy
    with CudaDeviceFunction(blockDim=256):
        # TeX: color line tile[0]
        #                           ggg
        for m2 in cuda_tasks(0, M / 128):
            # TeX: color line tile[0]
            #                           vvv
            for n2 in cuda_tasks(0, N / 256):
                # TeX: remark tile[0]
                # Expanded tile size
                # TeX: color line tile[0]
                #             yyy                  ggg   vvv
                # Each CTA of 256 threads computes 128 x 256 output tile
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: end tile[1:]
                        # TeX: color line *
                        #                            rrrrrr                       bbbbbbbbb
                        # Each thread accumulates an 8 x 16 output matrix tile in registers
                        # TeX: begin tile[1:]
                        # TeX: color line tile[0]
                        #          rrrrr    bbbbbbbb
                        accum: f32[8, 16] @ CudaRmem
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m0, n0] = 0       # Clear accumulators to 0
                        for k in seq(0, K):
                            for m0 in seq(0, 8):
                                for n0 in seq(0, 16):
                                    accum[m0, n0] += A[m2*128 + m1*8 + m0, k] \
                                                     * B[n2*256 + n1*16 + n0, k]
                        # Write 8 x 16 output matrix tile to main memory
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m0, n0]
    # TeX: end tile
