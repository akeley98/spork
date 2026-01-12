from __future__ import annotations

import math
import numpy as np
import pytest
import random

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *

"""
# TeX: version run_sporkbench 1
# TeX: begin run_sporkbench[0]
./sporkbench.py examples/f16_experiment/
# For large problem sizes the naïve gemm is skipped to avoid wasting your time
examples/f16_experiment/bin/sporkbench/run_sporkbench f16.json
./plot_sporkbench.py f16.json f16
# TeX: end run_sporkbench[0]
"""

"""
# TeX: version grep 1
# TeX: begin grep[0]
grep -E "L__BB|sync|ldmatrix" \
examples/f16_experiment/bin/sporkbench/sporkbench_cutlass_Sm80_f16_f16.ptx  > grep_ptx.txt
# Note important nvcc flags to generate your own PTX
# See examples/f16_experiment/bin/sporkbench/build.ninja
--keep --keep-dir [dir]  # Put assembly files in [dir]
--lineinfo  # Useful debug info [file_id] [row] [col], see .file at the bottom for [file_id]
-arch compute_80  # Enable Ampere features (cutlass will silently not use Tensor Cores w/o this)
# TeX: end grep[0]
"""


@instr
class ldmatrix_test(InstrInfo):
    # TeX: version ldmatrix_behavior 1
    # TeX: begin ldmatrix_behavior[0]
    # Loads a $(8\texttt{nmat0} \times 8\texttt{nmat1})$ tile.
    def behavior(
            nmat0: size, nmat1: size,  # Substitute constants so nmat0 * nmat1 == 4
            rmem: [f16][
                8,          # 8 L-rows, distributed by 4*cuda_thread
                4,          # 4 registers per L-row, distributed by cuda_thread
                nmat0,      # Number of L-matrices in the outer dimension (M or N, for us)
                nmat1,      # Number of L-matrices in the inner dimenision (K, for us)
                2]          # 2 f16 values per register
                @ CudaRmemPacked32,
            src: [f16][8 * nmat0, 8 * nmat1] @ CudaSmemAtomicity16B):
        # Iterate over L-rows (ldmatrix PTX docs assumes row major)
        # Distributed by register index and threads (thread pitch 4)
        for oR in seq(0, nmat0):
            for oT in seq(0, 8):
                # "columns"
                # Distributed by register index, threads (thread pitch 1), bit pack
                for iR in seq(0, nmat1):
                    for iT in seq(0, 4):
                        for iB in seq(0, 2):
                            rmem[oT, iT, oR, iR, iB] = src[8 * oR + oT,
                                                           8 * iR + 2 * iT + iB]
    # TeX: end ldmatrix_behavior[0]

    def instance(self: InstrInfo):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        self.access_info["rmem"].distributed_coll_units = [4 * cuda_thread, cuda_thread]

    def codegen(self, args):
        return []

@proc
def p(gmem: f16[16, 16] @ CudaGmemLinear, out: f16[16, 16] @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=32):
        for task in cuda_tasks(0, 1):
            src: f16[16, 16] @ CudaSmemLinear
            for m_ld in cuda_threads(0, 16):
                for k_ld in seq(0, 16):
                    src[m_ld, k_ld] = gmem[m_ld, k_ld]
            Fence(cuda_in_order, cuda_in_order)
            rmem: f16[16, 16] @ CudaRmemPacked32
            for m in seq(0, 16):
                for k in seq(0, 16):
                    rmem[m, k] = src[m, k]
            Fence(cuda_in_order, cuda_in_order)
            for m_st in seq(0, 16):
                for k_st in seq(0, 16):
                    out[m_st, k_st] = rmem[m_st, k_st]


p = divide_dim(p, "rmem", 0, 8)
p = divide_dim(p, "rmem", 2, 8)
p = divide_dim(p, "rmem", 3, 2)
p = rearrange_dim(p, "rmem", [1, 3, 0, 2, 4])
p = divide_loop(p, "m", 8, ("mR", "mT"), perfect=True)
p = divide_loop(p, "k", 8, ("kR", "kT"), perfect=True)
p = divide_loop(p, "kT", 2, ("kT", "kB"), perfect=True)
p = divide_loop(p, "m_st", 8, ("mR_st", "mT_st"), perfect=True)
p = divide_loop(p, "k_st", 8, ("kR_st", "kT_st"), perfect=True)
p = divide_loop(p, "kT_st", 2, ("kT_st", "kB_st"), perfect=True)
p = set_loop_mode(p, "mT_st", CudaThreads(unit=4 * cuda_thread))
p = set_loop_mode(p, "kT_st", CudaThreads(unit=1 * cuda_thread))
p = simplify(p)
p = replace(p, p.find_loop("mR"), ldmatrix_test())
p.sync_check()

print(p)


ntid = 256
n = 4
@proc
def sample_commit_group():
    with CudaDeviceFunction(blockDim=ntid):
        for task in cuda_tasks(0, 1):
            # CTA-scope here
            # TeX: version commit_group 1
            # TeX: begin commit_group[0]
            cg: barrier[ntid] @ CudaCommitGroup  # Where ntid is the number of threads per CTA (blockDim)
            for tid in cuda_threads(0, ntid, unit=cuda_thread):
                # Thread-scope here
                Arrive(Sm80_cp_async) >> cg[tid]  # PTX: cp.async.commit_group
                Await(cg[tid], cuda_in_order, n)  # PTX: cp.async.wait_group n, with n constant, n $\ge$ 0
            # TeX: end commit_group[0]


"""
# TeX: version cutlass_pseudocode 1
# TeX: begin cutlass_pseudocode[0]
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
for k in seq(0, RING - 1):
    cp.async tile k of A and B into SMEM
    cp.async.commit_group
cp.async.wait_group RING-2
__syncthreads()
Load first half of tile 0 of A and B from SMEM into A[0,:], B[0,:]
C = 0  # MMA accumulators

# The $k^{th}$ iteration of the loop accumulates the $k^{th}$ tiles of A and B into C,
# and starts GMEM $\to$ SMEM, SMEM $\to$ RMEM loads needed for future iterations.
for k in seq(0, K / K_cta):
    Load second half of tile k of A and B from SMEM into A[1,:], B[1,:]
    C += A[0,:] @ B[0,:]  # Using mma.sync

    cp.async tile (k + RING - 1) of A and B into SMEM
    cp.async.commit_group
    cp.async.wait_group RING-2
    __syncthreads()

    Load first half of tile (k+1) of A and B from SMEM into A[0,:], B[0,:]
    C += A[1,:] @ B[1,:]  # Using mma.sync

cp.async.wait_group 0
__syncthreads()
Write out C

# TeX: end cutlass_pseudocode[0]
"""
