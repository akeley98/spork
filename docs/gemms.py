# TeX: version intro 3
# TeX: version par 1

# TeX: begin intro
@proc
def gemm_one_per_thread(M: size, N: size, K: size,
                        # TeX: color line intro[0]
                        #             rrrrrrrrrrrrrr                   rrrrrr
                        A: f32[M,K] @ CudaGmemLinear,   # Row-major in global memory
                        # TeX: color line intro[0]
                        #             rrrrrrrrrrrrrr                      rrrrrr
                        B: f32[N,K] @ CudaGmemLinear,   # Column-major in global memory
                        # TeX: color line intro[0]
                        #             rrrrrrrrrrrrrr                      rrrrrr
                        C: f32[N,M] @ CudaGmemLinear):  # Column-major in global memory
    # TeX: color remark intro[0]
    #                                                              yyyyyyyyyyyyyyyyyy
    # CPU code here; CUDA device function (``kernel'') opened with CudaDeviceFunction
    # TeX: color line intro[0] par
    #    yyyyyyyyyyyyyyyyyy          yyy
    with CudaDeviceFunction(blockDim=256):
        # TeX: color line intro[0]
        #                           gg
        for m2 in cuda_tasks(0, M / 16):
            # TeX: color line intro[0]
            #                           vv
            for n2 in cuda_tasks(0, N / 16):
                # TeX: begin par
                # TeX: color line intro[0] par
                #                                yyy                  gg   vv
                # Per-CTA code here; each CTA of 256 threads computes 16 x 16 output tile
                # TeX: color line intro[2] par
                #   rr                        vvvvvvvvvvvvvvvvvvvvv
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color remark intro[2] par
                    #        yyy                              vv
                    # CTA of 256 threads splits into teams of 16 threads
                    # TeX: color line intro[2] par
                    #   bb                        gggggggggggggggg
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: color remark intro[2] par
                        #                                gggggg
                        # Teams of 16 threads split into single threads
                        # Per-thread code
                        # TeX: remark intro[0]
                        # Each thread calculates one value of the output matrix
                        # TeX: color remark intro[1]
                        #                                                rrrrrrrr
                        # Accumulate one value of the output matrix in a register
                        # TeX: color line intro[1]
                        #            rrrrrrrr
                        accum: f32 @ CudaRmem
                        # TeX: end par
                        accum = 0
                        # TeX: remark intro[1]
                        # Sequentially accumulate dot(A[m,0:K], B[n,0:K])
                        for k in seq(0, K):
                            accum += A[m2*16 + m1, k] * B[n2*16 + n1, k]
                        # TeX: remark intro[1]
                        # Write accumulated value to output matrix, column major
                        C[n2*16 + n1, m2*16 + m1] = accum
# TeX: end intro

# TeX: begin par

                # Lowered C++
                # TeX: color line *
                #       rrrrrrrrrrrr   vvvvvvvvvvvvvvvvvv
                if (int exo_16thr_m1 = (threadIdx.x / 16); 1) {
                  # TeX: color line *
                  #       bbbbbbbbbbb   gggggggggggggggggg
                  if (int exo_1thr_n1 = (threadIdx.x % 16); 1) {
                    float accum;
# TeX: end par

# TeX: version tile 3

@proc
def gemm_tile_per_thread(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    # TeX: begin tile
    # TeX: color line tile[0]
    #                                yyy
    with CudaDeviceFunction(blockDim=256):
        # TeX: color line tile[0:2]
        #   gg                      ggg
        for m2 in cuda_tasks(0, M / 128):
            # TeX: color line tile[0:2]
            #   vv                      vvv
            for n2 in cuda_tasks(0, N / 256):
                # TeX: remark! tile[0]
                # Expanded tile size
                # TeX: color line tile[0]
                #             yyy                  ggg   vvv
                # Each CTA of 256 threads computes 128 x 256 output tile
                # TeX: remark! tile[2]
            # Parallel loops: splits thread collectives: 1 sub-collective assigned per ``iteration''
                # TeX: color line tile[1]
                #   gg
                # TeX: color line tile[2]
                #         rrrrrrrrrrrrr                            r
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color line tile[1]
                    #   vv
                    # TeX: color line tile[2]
                    #         rrrrrrrrrrrrr                       r
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: end tile[1:]
                        # TeX: color line *
                        #                            bbbbbb                       rrrrrrrrr
                        # Each thread accumulates an 8 x 16 output matrix tile in registers
                        # TeX: begin tile[1:]
                        # TeX: color line tile[0]
                        #          bbbbb    rrrrrrrr
                        # TeX: color remark tile[0]
                        #           bbbbbbbbbbbbbbbbbbbb
                        # Lowers to float accum[8 * 16];
                        accum: f32[8, 16] @ CudaRmem
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m0, n0] = 0       # Clear accumulators to 0
                        # TeX: remark! tile[2]
                        # Sequential loops: same thread(s) performs all work
                        # TeX: color line tile[2]
                        #        bbbb    b
                        for k in seq(0, K):
                            # TeX: color line tile[1]
                            #                g
                            # TeX: color line tile[2]
                            #         bbbb    b
                            for m0 in seq(0, 8):
                                # TeX: color line tile[1]
                                #                vv
                                # TeX: color line tile[2]
                                #         bbbb     b
                                for n0 in seq(0, 16):
                                    # TeX: remark! tile[1]
                                    # Index expressions got more complicated due to work tiling
                                    # TeX: color line tile[1]
                                    #                     ggg      g
                                    accum[m0, n0] += A[m2*128 + m1*8 + m0, k] \
                                    # TeX: color line tile[1]
                                    #                       vvv      vv
                                                     * B[n2*256 + n1*16 + n0, k]
                        # Write 8 x 16 output matrix tile to main memory
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m0, n0]
    # TeX: end tile

# TeX: version gmem_remark 1

# TeX: begin gmem_remark
@proc
def gemm_tile_per_thread(M: size, N: size, K: size,
                         # TeX: color line *
                         #             rrrrrrrrrrrrrr                   rrrrrr
                         A: f32[M,K] @ CudaGmemLinear,   # Row-major in global memory
                         # TeX: color line *
                         #             rrrrrrrrrrrrrr                      rrrrrr
                         B: f32[N,K] @ CudaGmemLinear,   # Column-major in global memory
                         C: f32[N,M] @ CudaGmemLinear):  # Column-major in global memory
    with CudaDeviceFunction(blockDim=256):
# TeX: end gmem_remark
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                # TeX: summary
                # Distribute work over threads
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        accum: f32[8, 16] @ CudaRmem
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m0, n0] = 0
                        # TeX: begin gmem_remark
                        for k in seq(0, K):
                            for m0 in seq(0, 8):
                                for n0 in seq(0, 16):
                                    # TeX: remark *
                                    # Reading directly from global memory
                                    # TeX: remark! *
                                    # Inefficient
                                    # TeX: color line *
                                    #                rr                     r
                                    accum[m0, n0] += A[m2*128 + m1*8 + m0, k] \
                                    # TeX: color line *
                                    #                  rr                      r
                                                     * B[n2*256 + n1*16 + n0, k]
                        # TeX: end gmem_remark
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m0, n0]



# TeX: version dmem 5

@proc
def gemm_tile_per_cta(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
# TeX: begin dmem
    with CudaDeviceFunction(blockDim=256):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                # TeX: remark! dmem[0] dmem[4]
                # Tensor allocated by 256 threads (``alloc unit'')
                # TeX: remark! dmem[0] dmem[4]
                # CudaRmem requires allocation by 1 thread (``native unit'')
                # TeX: remark! dmem[1]
                # Distributed memory: distributed ``slices'' allocated per-thread
                # TeX: color line dmem[0]
               #rrrrr                       rrrrrrrr
                # TeX: color line dmem[1] dmem[4]
               #rrrrr              bbbbb    rrrrrrrr
                # TeX: remark! dmem[2:4]
                # Distribution deduced from later usage pattern
                # TeX: color remark dmem[2:]
                # ggggggggggggggggggg
                # m1: 256->16 threads
                # TeX: color remark dmem[3:]
                # vvvvvvvvvvvvvvvv
                # n1: 16->1 thread
                # TeX: color line dmem[2]
               #rrrrr      gg               rrrrrrrr
                # TeX: color line dmem[3]
               #rrrrr      gg  vv           rrrrrrrr
                # TeX: remark! dmem[4]
                # Goal met: remaining dimensions are not distributed
                accum: f32[16, 16, 8, 16] @ CudaRmem
                # TeX: color line dmem[2:4]
                #   gg                             gggggggggggggggg
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color line dmem[3]
                    #   vv                             vvvvvvvvvvv
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                # TeX: color line dmem[2]
                                #     gg
                                # TeX: color line dmem[3]
                                #     gg  vv
                                accum[m1, n1, m0, n0] = 0
                        for k in seq(0, K):
                            for m0 in seq(0, 8):
                                for n0 in seq(0, 16):
                                    # TeX: color line dmem[2]
                                    #     gg
                                    # TeX: color line dmem[3]
                                    #     gg  vv
                                    accum[m1, n1, m0, n0] += A[m2*128 + m1*8 + m0, k] \
                                                             * B[n2*256 + n1*16 + n0, k]
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = \
                                    # TeX: color line dmem[2]
                                    #     gg
                                    # TeX: color line dmem[3]
                                    #     gg  vv
                                    accum[m1, n1, m0, n0]
# TeX: end dmem

# TeX: version dmem_before 1
# TeX: version dmem_after 1

# TeX: begin dmem_before
# TeX: color line *
#   gg
for m1 in cuda_threads(
        0, 16, unit=16 * cuda_thread):
# TeX: color line *
#       vv
    for n1 in cuda_threads(
            0, 16, unit=cuda_thread):
# TeX: color line *
#       rrrrr      bbbbb    rrrrrrrr
        accum: f32[8, 16] @ CudaRmem

# Lowered C++
unsigned tid = threadIdx.x;  // for slides
# TeX: color line *
#       gggggggggggg
if (int exo_16thr_m1 = tid / 16; 1) {
# TeX: color line *
#         vvvvvvvvvvv
  if (int exo_1thr_n1 = tid % 16; 1) {
# TeX: color line *
#         rrrrr bbbbbb
    float accum[8 * 16];
# TeX: end dmem_before

# TeX: begin dmem_after
# TeX: color line *
rrrrr      gg  vv  bbbbb    rrrrrrrr
accum: f32[16, 16, 8, 16] @ CudaRmem
# TeX: color line *
#   gg
for m1 in cuda_threads(
        0, 16, unit=16 * cuda_thread):
# TeX: color line *
#       vv
    for n1 in cuda_threads(
            0, 16, unit=cuda_thread):

# Lowered C++
# TeX: color line *
#     rrrrr bbbbbb
float accum[8 * 16];
unsigned tid = threadIdx.x;  // for slides
# TeX: color line *
#       gggggggggggg
if (int exo_16thr_m1 = tid / 16; 1) {
# TeX: color line *
#         vvvvvvvvvvv
  if (int exo_1thr_n1 = tid % 16; 1) {
# TeX: end dmem_after


# TeX: version smem_alloc 1
# TeX: version broken_smem 3

@proc
def gemm_broken_smem(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[N,K] @ CudaGmemLinear, C: f32[N,M] @ CudaGmemLinear):
    # TeX: begin smem_alloc
    with CudaDeviceFunction(blockDim=256):
        for m2 in cuda_tasks(0, M / 128):
            for n2 in cuda_tasks(0, N / 256):
                # TeX: begin broken_smem
                # TeX: color line broken_smem[0] smem_alloc
                #                                               yyyy
                # Cache (128 x 32) A tile, (256 x 32) B tile in SMEM
                # TeX: color remark! smem_alloc
                #       yyyy
                # These SMEM allocations are per-CTA,
                # TeX: remark! smem_alloc
                # since we are outside any cuda_threads loops
                # TeX: color line broken_smem[0] smem_alloc
                #                      yyyyyyyyyyyyyy
                A_smem: f32[128, 32] @ CudaSmemLinear
                # TeX: color line broken_smem[0] smem_alloc
                #                      yyyyyyyyyyyyyy
                B_smem: f32[256, 32] @ CudaSmemLinear
                # TeX: end broken_smem smem_alloc
                # TeX: begin broken_smem[2]
                # TeX: color remark! *
                #              rrrrr                  bb
                # Need to lift accum outside of outer k1 loop ... but now this is allocated per-CTA!
                # TeX: color line *
               #rrrrr             rrrrrrrr
                accum: f32[???] @ CudaRmem
                # TeX: end broken_smem[2]

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                accum[m1, n1, m0, n0] = 0

                # TeX: begin broken_smem smem_alloc
                # TeX: color line *
                #   bb               bb                                   bb
                for k1 in seq(0, K / 32):  # Divide outer/inner k loop by 32
                # TeX: end broken_smem smem_alloc
                    for m1 in seq(0, 16):
                        for m0 in cuda_threads(0, 8, unit=32*cuda_thread):
                            for k0 in cuda_threads(0, 32):
                                # TeX: summary
                                # Fill 128 x 32 tile of A_smem
                                A_smem[m1*8 + m0, k0] = A[m2*128 + m1*8 + m0, k1*32 + k0]
                    for n1 in seq(0, 32):
                        for n0 in cuda_threads(0, 8, unit=32*cuda_thread):
                            for k0 in cuda_threads(0, 32):
                                # TeX: summary
                                # Fill 256 x 32 tile of B_smem
                                B_smem[n1*8 + n0, k0] = B[n2*256 + n1*8 + n0, k1*32 + k0]

                    # TeX: begin broken_smem smem_alloc
                    # TeX: remark! *
                    # __syncthreads()
                    Fence(cuda_classic, cuda_classic)
                    # TeX: color remark! broken_smem[0]
                    #                        rrrrr    yyyyyyyyyyyyyyy
                    # Let's fill in code for accum += A_smem @ B_smem
                    # TeX: end broken_smem smem_alloc

                    # TeX: begin broken_smem
                    for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                        for n1 in cuda_threads(0, 16, unit=cuda_thread):
                            # TeX: color remark! broken_smem[1]
                            #                     rrrrr                                       bb
                            # Problem: per-thread accum goes out-of-scope too soon (reset per k1 iteration)
                            # TeX: color line broken_smem
                           #rrrrr               rrrrrrrr
                            accum: f32[8, 16] @ CudaRmem
                            # TeX: color line *
                            #   bb           bb                                   bb
                            for k0 in seq(0, 32):  # Divide outer/inner k loop by 32
                                for m0 in seq(0, 8):
                                    for n0 in seq(0, 16):
                                        # TeX: summary
                                        # Accumulate A_smem @ B_smem
                                        accum[m0, n0] += A_smem[m1*8 + m0, k0] * B_smem[n1*16 + n0, k0]
                           # TeX: color remark broken_smem[1]
                           #  rrrrr
                           # (accum dies here)
                    # TeX: end broken_smem

                    # TeX: begin broken_smem smem_alloc
                    # TeX: remark! *
                    # __syncthreads()
                    Fence(cuda_classic, cuda_classic)
                    # TeX: end broken_smem smem_alloc

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m0, n0]
