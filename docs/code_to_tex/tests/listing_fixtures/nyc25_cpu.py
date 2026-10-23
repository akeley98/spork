# The `cpu` listing of spork/docs/nyc25.py (legacy column-letter colors),
# rewritten in the code_to_tex directive syntax.  Legacy output for
# comparison: spork/docs/nyc25_tex/cpu.{0..4}.tex.
from __future__ import annotations

from exo import *

if True:
# TeX: listing cpu 5
# TeX: begin cpu[:4]
# TeX: color cpu[0] blue "@proc"
  @proc
  # TeX: color cpu[1] blue "size" 3:*
  def nyc25_gemm_cpu(M: size, N: size, K: size,
  # TeX: color cpu[1] blue "f32[M, K]"
  # TeX: color cpu[2] blue "DRAM"
                     A: f32[M, K] @ DRAM,
  # TeX: color cpu[1] blue "f32[K, N]"
  # TeX: color cpu[2] blue "DRAM"
                     B: f32[K, N] @ DRAM,
  # TeX: color cpu[1] blue "f32[M, N]"
  # TeX: color cpu[2] blue "DRAM"
                     C: f32[M, N] @ DRAM):
    # TeX: begin cpu[4]
    # TeX: color cpu[3:] green "m" "seq"
    for m in seq(0, M):
      # TeX: color cpu[3:] violet "n" "seq"
      for n in seq(0, N):
        # TeX: color cpu[1] blue "f32"
        # TeX: color cpu[2] blue "DRAM"
        accum: f32 @ DRAM
        accum = 0
        # TeX: color cpu[3:] blue "k"
        # TeX: color cpu[3] blue "seq"
        for k in seq(0, K):
          # TeX: color cpu[4] green "m" violet "n"
          accum += A[m, k] * B[k, n]
        # TeX: color cpu[4] green "m" violet "n"
        C[m, n] = accum
        # TeX: end cpu[4]
# TeX: end cpu[:4]
