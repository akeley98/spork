"""Code listing source for amspork3 (main.tex and decks); must pass exocc.

Listing directive format: spork/docs/code_to_tex/README.md ("Source format").
"""
from __future__ import annotations

from exo import *
from exo.scalars import f16, bf16, f32
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

# TeX: listing intro_cp_async 1
# TeX: listing intro_mma 1
M = 64
N = 256
K = 32
@proc
def instr_usage_example(src: f32[4] @ CudaGmemLinear):
  with CudaDeviceFunction(blockDim=128):
    for task in cuda_tasks(0, 1):
      # I need a pile of BS code snippets to get the idea across.
      # Everything is at the same indent, in its own 1 thread or 1 warpgroup "loop".
      dst: f32[4] @ CudaSmemLinear
      for tid in cuda_threads(0, 1):
        # Manual loop for cp.async vs cp.async instr
        # TeX: begin intro_cp_async
        # TeX: mark intro_cp_async loop start
        for i in seq(0, 4):
          dst[i] = src[i]
        # Replace with 4-at-once $\textbf{async}$ copy
        # TeX: mark intro_cp_async instr start
        Sm80_cp_async_1d(
        # TeX: color intro_cp_async elide "dst=f32, src=f32, size0=4"
          dst[:], src[:], dst=f32, src=f32, size0=4)
        # TeX: end intro_cp_async
        C: f32[M, N] @ CudaRmemLinear
        A: f32[M, K] @ CudaRmemLinear
        B: f32[N, K] @ CudaRmemLinear
        # Manual loops for wgmma
        # TeX: begin intro_mma
        # TeX: mark intro_mma left start
        # TeX: mark intro_mma loop end
        for m in seq(0, M):
          for n in seq(0, N):
            for k in seq(0, K):
              C[m, n] += (
                A[m, k] * B[n, k])
        # TeX: end intro_mma
      C: f32[4, 16, N] @ Sm90_TkRmemTileD(N)
      A: f32[M, K] @ Sm90_SmemSwizzled(128)
      B: f32[N, K] @ Sm90_SmemSwizzled(128)
      for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
        # TeX: begin intro_mma
        # Replace with tensor core
        # TeX: mark intro_mma instr end
        Sm90_tk_mma_row_col(
          # TeX: color intro_mma elide ":, :, :" ":, :" 3:1 ":, :" 3:2 "D=f32, A=f32, B=f32, N=N, K=K"
          C[:, :, :], A[:, :], B[:, :], D=f32, A=f32, B=f32, N=N, K=K)
        # TeX: end intro_mma
      Fence(Sm80_generic, cuda_in_order)

del M
del N
del K


# TeX: listing intra_thread 2
@proc
def intra_thread_example(src: f32[4] @ CudaGmemLinear):
  with CudaDeviceFunction(blockDim=128):
    for task in cuda_tasks(0, 1):
      dst: f32[4] @ CudaSmemLinear
      for tid in cuda_threads(0, 1):
        whatever: f32 @ CudaRmemLinear
        # TeX: begin intra_thread
        # $\textbf{Single threaded}$ code!
        #
        # 4-at-once $\textbf{async}$ copy
        Sm80_cp_async_1d(
        # TeX: color intra_thread elide "dst=f32, src=f32, size0=4"
        # TeX: color intra_thread[0] red "dst[:]"
          dst[:], src[:], dst=f32, src=f32, size0=4)
        # TeX: remark intra_thread[0]
        # No good, didn't wait for copy to retire
        # TeX: end intra_thread[0]
        # TeX: color intra_thread[1] green "Sm80_cp_async"
        Fence(Sm80_cp_async, cuda_in_order)
        # TeX: begin intra_thread[0]
        # TeX: color intra_thread[0] red "dst[0]"
        whatever += dst[0]
        # TeX: remark intra_thread[1]
        # (`Fence` will be explained later)
        # TeX: end intra_thread


# TeX: listing lang_control 5
# TeX: begin lang_control
@proc
def example(
  M: size, N: size, K: size,
  # TeX: color lang_control[1] green "A: bf16[K, M]" elide "global memory"
  # TeX: color lang_control[2] violet "A" "@ CudaGmemLinear" elide "proc parameter"
  # TeX: color lang_control[0],lang_control[3:] elide "# proc parameter global memory"
  A: bf16[K, M] @ CudaGmemLinear # proc parameter global memory
):
# TeX: end lang_control
  assert M >= 256
  assert K >= 32
  # TeX: begin lang_control
  # TeX: color lang_control[1] blue "C: f32[N, M]" elide "global memory"
  # TeX: color lang_control[2] violet "C" "@ CudaGmemLinear" elide "allocated in-proc"
  # TeX: color lang_control[0],lang_control[3:] elide "# allocated in-proc global memory"
  C: f32[N, M] @ CudaGmemLinear # allocated in-proc global memory
  with CudaDeviceFunction(blockDim=128):
    # TeX: color lang_control[3] green "task" "cuda_tasks(0, 24)"
    # TeX: color lang_control[:3],lang_control[4:] elide  "# Parallel loop"
    for task in cuda_tasks(0, 24):  # Parallel loop
      # TeX: color lang_control[1] blue "A_tile: bf16[32, 256]" elide "shared memory"
      # TeX: color lang_control[2] yellow "A_tile" "@ CudaSmemLinear" elide "allocated in-proc"
      # TeX: color lang_control[0],lang_control[3:] elide "# allocated in-proc shared memory"
      A_tile: bf16[32, 256] @ CudaSmemLinear # allocated in-proc shared memory
      # TeX: color lang_control[3] blue "k" "seq(0, 32)"
      # TeX: color lang_control[:3],lang_control[4:] elide  "# Sequential loop"
      for k in seq(0, 32):  # Sequential loop
        # TeX: color lang_control[3] green "m" "cuda_threads(0, 32)"
        # TeX: color lang_control[:3],lang_control[4:] elide  "# Parallel loop"
        for m in cuda_threads(0, 32):  # Parallel loop
          # TeX: color lang_control elide "dst=bf16, src=bf16, size0=8"
          # TeX: color lang_control[4] yellow "A_tile[k, m:m+8]" violet "A[k, m:m+8]"
          Sm80_cp_async_1d(A_tile[k, m:m+8], A[k, m:m+8], dst=bf16, src=bf16, size0=8)
  # TeX: end lang_control

lang_control_example = example  # Preserve for exocc compilation (typo checking)
del example


# TeX: listing lang_scope 4
# TeX: begin lang_scope
# TeX: mark lang_scope[0] cpu_top start
@proc
# TeX: color lang_scope[1] blue "sz: size"
def lang_scope_example(sz: size):
  # TeX: end lang_scope
  # Weird to have alloc here but gives something for CPU code to point to.
  # TeX: begin lang_scope
  # TeX: color lang_scope[2] blue "S: f32[sz]"
  # TeX: mark lang_scope[0] cpu_bot start
  S: f32[sz] @ CudaGmemLinear
  # TeX: color lang_scope[0] red "with CudaDeviceFunction"
  with CudaDeviceFunction(blockDim=128):
    # TeX: color lang_scope[1] blue "task"
    # TeX: color lang_scope[3] red "for"
    # TeX: mark lang_scope[0] gpu_top start
    for task in cuda_tasks(
        # TeX: color lang_scope[3] blue "sz"
        0, (sz + 127) / 128):
      # TeX: color lang_scope[2] blue "tmp: f32[128]"
      tmp: f32[128] @ CudaSmemLinear
      # TeX: color lang_scope[1] blue "tid"
      for tid in cuda_threads(0, 128):
        # TeX: color lang_scope[3] red "if" blue "task" "tid" "sz"
        if task * 128 + tid < sz:
          # TeX: color lang_scope[3] red "tmp" "S" blue "tid" 2:* "task"
          # TeX: mark lang_scope[0] gpu_bot start
          tmp[tid] = S[task * 128 + tid]
  # TeX: end lang_scope


# TeX: listing tiling 3
@proc
def tiling_example(M: size, N: size, K: size):
  # We just need some content-free loops for illustrating loop tiling rewrites
  assert M % 32 == 0
  assert N % 32 == 0
  # TeX: begin tiling[0]
  # TeX: color tiling[0] green "m" elide "(0, M)"
  # TeX: mark tiling[0] m_end end
  for m in seq(0, M):
    # TeX: color tiling[0] violet "n" elide "(0, N)"
    # TeX: mark tiling[0] n_end end
    for n in seq(0, N):
      # TeX: color tiling[0] blue "k" elide "(0, K)"
      for k in seq(0, K):
        # TeX: end tiling[0]
        pass
  # TeX: begin tiling[1]
  # TeX: color tiling[1] green "mo" elide "(0, M/32)"
  # TeX: mark tiling[1] mo_start start
  for mo in seq(0, M/32):
    # TeX: color tiling[1] green "mi" elide "(0, 32)"
    # TeX: mark tiling[1] mi_start start
    # TeX: mark tiling[1] mi_end end
    for mi in seq(0, 32):
      # TeX: color tiling[1] violet "no" elide "(0, N/32)"
      # TeX: mark tiling[1] no_start start
      # TeX: mark tiling[1] no_end end
      for no in seq(0, N/32):
        # TeX: color tiling[1] violet "ni" elide "(0, 32)"
        # TeX: mark tiling[1] ni_start start
        for ni in seq(0, 32):
          # TeX: color tiling[1] blue "k" elide "(0, K)"
          for k in seq(0, K):
            # TeX: end tiling[1]
            pass
  # TeX: begin tiling[2]
  # TeX: color tiling[2] green "mo" elide "(0, M/32)"
  for mo in seq(0, M/32):
    # TeX: color tiling[2] violet "no" elide "(0, N/32)"
    # TeX: mark tiling[2] no_start start
    for no in seq(0, N/32):
      # TeX: color tiling[2] green "mi" elide "(0, 32)"
      # TeX: mark tiling[2] mi_start start
      for mi in seq(0, 32):
        # TeX: color tiling[2] violet "ni" elide "(0, 32)"
        for ni in seq(0, 32):
          # TeX: color tiling[2] blue "k" elide "(0, K)"
          for k in seq(0, K):
            # TeX: end tiling[2]
            pass
