"""
exocc nexto.py && python3 code_to_tex.py nexto.py nexto && xelatex </dev/null nexto.tex

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

A. Algebraic number system
B. Typed strides
C. IR data structure and functional specification
    * Partial proof

> Digress about instr woes in Exo
    > MMA example with / and %
    > cudaMemcpy mess
    > TMA mess

D. Instr with input buffers
E. Loop iterators ``rich type''

IR design [skip this]
  * Flat list of stmts: Alloc, Free, SyncStmt, Mutate (WindowStmt)
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
