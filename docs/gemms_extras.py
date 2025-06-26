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
        # Each thread owns a 8 x 16 tile
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
# The CTA owns a 128 x 256 tile (8 x 16 per thread)
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
                # since the collective unit is CTA outside any cuda_threads loops
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
                                # TeX: summary!
                                # Fill 128 x 32 tile of A_smem
                                A_smem[m1*8 + m0, k0] = A[m2*128 + m1*8 + m0, k1*32 + k0]
                    for n1 in seq(0, 32):
                        for n0 in cuda_threads(0, 8, unit=32*cuda_thread):
                            for k0 in cuda_threads(0, 32):
                                # TeX: summary!
                                # Fill 256 x 32 tile of B_smem
                                B_smem[n1*8 + n0, k0] = B[n2*256 + n1*8 + n0, k1*32 + k0]

                    # TeX: begin broken_smem smem_alloc
                    # TeX: remark! *
                    # __syncthreads()
                    # TeX: remark smem_alloc
                    # Collective unit = CTA here
                    Fence(cuda_in_order, cuda_in_order)
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
                                        # TeX: summary!
                                        # Accumulate A_smem @ B_smem
                                        accum[m0, n0] += A_smem[m1*8 + m0, k0] * B_smem[n1*16 + n0, k0]
                           # TeX: color remark broken_smem[1]
                           #  rrrrr
                           # (accum dies here)
                    # TeX: end broken_smem

                    # TeX: begin broken_smem smem_alloc
                    # TeX: remark! *
                    # __syncthreads()
                    # TeX: remark smem_alloc
                    # Collective unit = CTA here
                    Fence(cuda_in_order, cuda_in_order)
                    # TeX: end broken_smem smem_alloc

                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, 8):
                            for n0 in seq(0, 16):
                                C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m0, n0]


# TeX: version CudaWarps 2
@proc
def CudaWarps_example():  # XXX THIS CODE BARELY FITS THE SLIDES
    with CudaDeviceFunction(blockDim=384):
        for task in cuda_tasks(0, 3):
            # TeX: begin CudaWarps[0]
            # Example: before CudaWarps
            for i in cuda_threads(0, 128):
            # Lowered C++
            # TeX: color line *
            #       bbbbbbbbbbbbbbbbbbbbbbbb
            if (int exo_1thr_i = threadIdx.x;
                exo_1thr_i < 128) {
            # TeX: end CudaWarps[0]
            # TeX: begin CudaWarps[1]
            with CudaWarps(8, 12): # Example: after CudaWarps
                for i in cuda_threads(0, 128):
            # Lowered C++
            if (threadIdx.x >= 256) {
              # TeX: color line *
              #       bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
              if (int exo_1thr_i = threadIdx.x - 256; 1) {
            # TeX: end CudaWarps[1]
