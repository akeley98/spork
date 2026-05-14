"""
Good things about Exo
- Imperative mental model
- Explicit instruction selection, with valid usage checks
- Rewrite operations /as a convenience/

Good things about Exo-GPU
- Language-level parallelism; expresses cooperation and programmer intent
- Abstract machine
  - ``unsafe by default'': you can write whatever you want
      Sound and complete(ish) yes/no will-it-work answer for concrete sizes
  - But hopefully easier to analyze than plain C/CUDA.

Bad things about Exo
- Too imperative (LATER, tree traversal when analyzing, hard to substitute instructions)
  * Myopic
  * Slow IR analysis
  * Instr substitution is too hard
- Terrible user experience for error messages
- Rewrite operations /as an obligation/ (to check correctness)
- Scheduling Myopia (to be explained)
- Control/data separation, affine indexing, etc.

Bad things about Exo-GPU
- We are not CuTe
  * Can't reshape data at runtime
  * cudaMemcpyAsync example
  * Minimal ability to think about layouts wrt threads and values
  * Move this stuff to later!

- Constant integer restrictions cause grief
  * Can't support variable cluster size, slow parameter sweep, ...
- Implicit distributed memory is maybe a mistake
  * Hard to explain / debug when it goes wrong
  * Expressivity issues, instr can't write/read outside cooperating threads

Yuka's belief
  * Computers are imperative under the hood (alloc memories and write to them in program order)
  * Need to program imperatively if we want to optimize-to-the-metal
  * Program representation & analysis should be based on imperative code

Myopia

Programmer intuition
  * Functional specification
  * Buffer contents as function of spec + control variable values
  * GEMM example
  * Histogram example

Top-level idea
  * Not Myopic
  * Equal in expressivity, but richer information and optimized program IR
  * Imperative++, stmt w/ output function?

Top-level goals
  * Usable by LLMs
    - Pinpoint messages (value env, sync env, stretch goal perf info)
    - Compiler performance matters
    - Query & Rewrite API, although lower priority [except parameterization]
  * Expressivity, no language-imposed affine indexing or data-dependent control flow restrictions
    - Most handwavey part
  * Many levels of parallelism
    - Abstracting them is one thing, harder to do the engineering effort

MMA problems

A. IR basics
  - For loops, If guard, Async block
  - Alloc, Free, SyncStmt, Mutate, WindowStmt
  - Partial functional spec
  - Partial sync spec
  - None/Hope/Proof/Testing
B. Algebraic Number System
C. Typed Strides
D. Instr Substitution
E. Rich Loop Iterators

IR design [skip this]
  * Flat list of stmts: Alloc, Free, SyncStmt, Mutate, WindowStmt
  * Loop nest per-statement, if-guards, "async annotation"
  * Assignment includes functional annotation
  * Stitched together to form program
    - Loop identity will not be based on names, just for illustration!

What do I need to learn?
  * Static analysis
  * Proof assistant
  * Maybe hardware design languages, if there's a component of that?
  * Is there even research potential here, or just engineering?

Bonus: structured programming is bad
- Rachit's problem
- csv loop example
- interpreter problem
"""

from __future__ import annotations
from exo import *
from exo.platforms.cuda import *
from exo.stdlib.scheduling import *

# TeX: version gemm 4
M_tile, N_tile, K_tile = 128, 256, 32
@proc
def gemm(M: size, N: size, K: size, A: f32[M, K], B: f32[K, N], C: f32[M, N]):
    assert M % M_tile == 0
    assert N % N_tile == 0
    assert K % K_tile == 0
    # TeX: begin gemm
    # TeX: color line *
    #   gg
    for m1 in seq(0, M/M_tile):
        # TeX: color line *
        #   vv
        for n1 in seq(0, N/N_tile):
            # TeX: end gemm[1:]
            # TeX: summary
            # Declare cached tiles for A, B, C
            A_tile: f32[M_tile, K_tile]
            B_tile: f32[K_tile, N_tile]
            C_tile: f32[M_tile, N_tile]
            # TeX: begin gemm[1:]
            # ...
            # TeX: color line *
            #   bb
            for k1 in seq(0, K/K_tile):
                # TeX: color line *
                #   gg
                for m0 in seq(0, M_tile):
                    # TeX: color line *
                    #   bb
                    for k0 in seq(0, K_tile):
                        # TeX: color line gemm[1]
                      # yyyyyyyyyyyyyy
                        A_tile[m0, k0] = A[m0 + m1 * M_tile, k0 + k1 * K_tile]
                # TeX: color line *
                #   bb
                for k0 in seq(0, K_tile):
                    # TeX: color line *
                    #   vv
                    for n0 in seq(0, N_tile):
                        # TeX: color line gemm[2]
                      # yyyyyyyyyyyyyy
                        B_tile[k0, n0] = B[k0 + k1 * K_tile, n0 + n1 * N_tile]
                        # TeX: end gemm
                # TeX: summary
                # Omitted m0/n0/k0 loops
                for m0 in seq(0, M_tile):
                    for n0 in seq(0, N_tile):
                        for k0 in seq(0, K_tile):
                            # TeX: begin gemm
                            # TeX: color line gemm[3]
                          # yyyyyyyyyyyyyy
                            C_tile[m0, n0] += A_tile[m0, k0] * B_tile[k0, n0]
                            # TeX: end gemm

# TeX: version histogram 4
# TeX: begin histogram
def histogram(num_bins: size, bins: i32[num_bins], K: size, data: i32[K]):
    tmp_bins: i32[num_bins]
    for b in seq(0, num_bins):
        # TeX: color line histogram[1]
      # yyyyyyyyyyy
        tmp_bins[b] = 0
    for k in seq(0, K):
        # TeX: color line histogram[2]
      # yyyyyyyyyyyyyyyyy
        tmp_bins[data[k]] += 1
    for b in seq(0, num_bins):
        # TeX: color line histogram[3]
      #           yyyyyyyyyyy
        bins[b] = tmp_bins[b]
# TeX: end histogram


# TeX: version logical_mma 1
# TeX: begin logical_mma
# TeX: color line *
#                gg  bb            v  bb            gg  v
def mma(A: [f16][16, 16], B: [f16][8, 16], C: [f16][16, 8]):
    # TeX: color line *
    #   g
    for m in seq(0, 16):
        # TeX: color line *
        #   v
        for n in seq(0, 8):
            # TeX: color line *
            #   b
            for k in seq(0, 16):
                # TeX: color line *
                # g  v       g  b      v  b
                C[m, n] += A[m, k] * B[n, k]  # NB, (n, k) ordering matches PTX better
# TeX: end logical_mma

# A = mma.find_alloc_or_arg("A")
# mma = divide_dim(mma, A, 1, 2)
# mma = divide_dim(mma, A, 1, 4)
# mma = divide_dim(mma, A, 0, 8)
# mma = rearrange_dim(mma, A, [1, 3, 0, 2, 4])
# mma = simplify(mma)
# # print(mma)

# TeX: version mma 3
if False:
# TeX: begin mma[:2]
# TeX: color line *
#                rrrr  yyyy  v      rrrrrrrrrrrrrrrrrrrrr  yyyyyyyyyyy  vvvvvvvvvvv
                [8, 4, 2, 2, 2]  #  distributed (threads), register ID, bit-packing

# TeX: end mma[:2]
@proc
# TeX: begin mma
# TeX: color line *
#                g  b  g  b  b                               .....            .....
def mma(A: [f16][8, 4, 2, 2, 2] @ CudaRmemPacked32, B: [f16][8, 16], C: [f16][16, 8]):
    for m in seq(0, 16):
        for n in seq(0, 8):
            for k in seq(0, 16):
                # TeX: color line mma[0]
                # ....       g      b          g      b      b          ....
                # TeX: color line mma[1]
                # ....                         yyyyy  yyyyy             ....
                # TeX: color line mma[2]
                # ....         r      r   r      r      r      r        ....
                C[m, n] += A[m % 8, k / 2 % 4, m / 8, k / 8, k % 2] * B[n, k]
# TeX: end mma

mma = divide_loop(mma, "m", 8, ("mR", "mT"), perfect=True)
mma = divide_loop(mma, "k", 2, ("k", "kP"), perfect=True)
mma = divide_loop(mma, "k", 4, ("kR", "kT"), perfect=True)
mma = simplify(mma)
print(mma)

del mma

# TeX: version mma_fixed 1
# TeX: begin mma_fixed[0]
# TeX: color line *
#                                                              .....             .....
def mma(A : [f16][8, 4, 2, 2, 2] @ CudaRmemPacked32, B : [f16][8, 16], C : [f16][16, 8]):
  for mR in seq(0, 2):
    for mT in seq(0, 8):
      for n in seq(0, 8):  # TODO n has to be split too.
        for kR in seq(0, 2):
          for kT in seq(0, 4):
            for kP in seq(0, 2):
              # TeX: color line *
              # ..............                               .......................
              C[mT + 8 * mR, n] += A[mT, kT, mR, kR, kP] * B[n, kP + 2 * kT + 8 * kR]
# TeX: end mma_fixed[0]


"""
# TeX: version cutlass_pseudocode 1
RING = get_pipeline_depth()  # Some tuning constant
M_cta, N_cta, K_cta = cta_tile_size() # Some tuning constants
# Create ring buffers. $k^{th}$ tile goes to ring buffer slot k % RING
A_smem: f16[RING, M_cta, K_cta]  # Row major
B_smem: f16[RING, N_cta, K_cta]  # Column major
# Register tile, split into 2 logical halves.
# This is used to hide latency of SMEM to RMEM loads.
# TeX: color line *
#              .
A_rmem: f16[2, x]  # Figure out exact size yourself (warp tiles within CTA tiles)
# TeX: color line *
#              .
B_rmem: f16[2, x]  # NB the 2 halves don't have to be explicit, just shown here for clarity.
# For the real code, guard cp.async calls so they don't read out of bounds.
# TeX: begin cutlass_pseudocode[0]
for k in seq(0, RING - 1):  # SMEM is like a skewed sliding window
    cp.async tile k of A and B into SMEM  # Pre-populate first few sliding window entries
    cp.async.commit_group  # Not sure of any scheduling op that allows this skewed stage_mem
cp.async.wait_group RING-2
__syncthreads()
# TeX: color line *
#                                                                                    rrrr
Ld low half of tile 0 of A, B from SMEM into A_rmem[0,:], B_rmem[0,:]  # Consumed at (s0)
C = 0  # MMA accumulators
# The $k^{th}$ iteration of the loop accumulates the $k^{th}$ tiles of A and B into C,
# and starts GMEM $\to$ SMEM, SMEM $\to$ RMEM loads needed for future iterations.
for k in seq(0, K / K_cta):
# TeX: color line *
#                                                                                         bbbb
    Ld high half of tile k of A, B from SMEM into A_rmem[1,:], B_rmem[1,:]  # Consumed at (s1)
# TeX: color line *
#                                     rrrr
    C += A_rmem[0,:] @ B_rmem[0,:]  # (s0), using mma.sync

    cp.async tile (k + RING - 1) of A and B into SMEM
    cp.async.commit_group
    cp.async.wait_group RING-2
    __syncthreads()

# TeX: color line *
#                                                                                            rrrr
    Ld low half of tile (k+1) of A, B from SMEM into A_rmem[0,:], B_rmem[0,:]  # Consumed at (s0)
# TeX: color line *
#                                     bbbb
    C += A_rmem[1,:] @ B_rmem[1,:]  # (s1), using mma.sync
# TeX: end cutlass_pseudocode[0]
cp.async.wait_group 0
__syncthreads()
Write out C
"""
