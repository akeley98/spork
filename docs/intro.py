from __future__ import annotations
from exo import *
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *
from exo.platforms.x86 import *

# TeX: version simple_vec_add 1
# TeX: begin simple_vec_add[0]
@proc  # param: type @ mem
# TeX: color line *
#                     yyyy     yyyyyy   gggg     yyyyyy   gggg
def simple_vec_add(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):  # Defines new variable i of type index
        a[i] += b[i]
# TeX: end simple_vec_add[0]

# TeX: version divide_loop 1
# TeX: begin divide_loop[0]
@proc
def avx_add(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    assert N % 8 == 0  # Assume N is a multiple of 8
    # TeX: color line *
    #   bb           bbbbb
    for ii in seq(0, N / 8):  # Outer loop
        # TeX: color line *
        #   bbbb           b
        for lane in seq(0, 8):  # Inner loop
            a[lane + 8 * ii] += b[lane + 8 * ii]  # [i] = [lane + 8 * ii]
# TeX: end divide_loop[0]

# TeX: version stage_mem 1
# TeX: begin stage_mem[0]
@proc
def avx_add(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    assert N % 8 == 0
    # TeX: color line *
    #   bb           bbbbb
    for ii in seq(0, N / 8):  # Outer loop
        # Allocate and load 8-vectors
        # TeX: color line *
        #      yyyyyy
        a_vec: f32[8] @ DRAM
        for i0 in seq(0, 8):
            a_vec[i0] = a[i0 + 8 * ii]
        # TeX: color line *
        #      yyyyyy
        b_vec: f32[8] @ DRAM
        for i0 in seq(0, 8):
            b_vec[i0] = b[i0 + 8 * ii]
        # Inner loop
        # TeX: color line *
        #   bbbb           b
        for lane in seq(0, 8):
            a_vec[lane] += b_vec[lane]
        # Only a_vec gets written back as b_vec is not modified.
        for i0 in seq(0, 8):
            a[i0 + 8 * ii] = a_vec[i0]
# TeX: end stage_mem[0]

# TeX: version set_memory 1
# TeX: begin set_memory[0]
@proc
def avx_add(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    assert N % 8 == 0
    # TeX: color line *
    #   bb           bbbbb
    for ii in seq(0, N / 8):  # Outer loop
        # Allocate and load 8-vectors
        # TeX: color line *
        #               gggg
        a_vec: f32[8] @ AVX2
        for i0 in seq(0, 8):
            a_vec[i0] = a[i0 + 8 * ii]
        # TeX: color line *
        #               gggg
        b_vec: f32[8] @ AVX2
        for i0 in seq(0, 8):
            b_vec[i0] = b[i0 + 8 * ii]
        # Inner loop
        for lane in seq(0, 8):
            a_vec[lane] += b_vec[lane]
        # Only a_vec gets written back as b_vec is not modified.
        for i0 in seq(0, 8):
            a[i0 + 8 * ii] = a_vec[i0]
# TeX: end set_memory[0]

# TeX: version mm256_iadd_ps 1
# TeX: begin mm256_iadd_ps[0]
# Example instruction definition: we specify the C syntax for AVX float32 addition,
# and we specify that this acts as an 8-vector addition.
@instr("{x_data} = _mm256_add_ps({x_data}, {y_data});")
def mm256_iadd_ps(x: [f32][8] @ AVX2, y: [f32][8] @ AVX2):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    for i in seq(0, 8):
        x[i] += y[i]
# TeX: end mm256_iadd_ps[0]

# TeX: version replace 1
# TeX: begin replace[0]
@proc
def avx_add(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    assert N % 8 == 0
    # TeX: color line *
    #   bb           bbbbb
    for ii in seq(0, N / 8):  # Outer loop
        # Allocate and load 8-vectors
        a_vec: f32[8] @ AVX2
        # TeX: color line *
      # vvvvvvvvvvvvvv
        mm256_loadu_ps(a_vec[0:8], a[8 * ii + 0:8 * ii + 8])
        b_vec: f32[8] @ AVX2
          # TeX: color line *
      # vvvvvvvvvvvvvv
        mm256_loadu_ps(b_vec[0:8], b[8 * ii + 0:8 * ii + 8])
        # Inner loop (replaced with vector add)
          # TeX: color line *
      # vvvvvvvvvvvvv
        mm256_iadd_ps(a_vec[0:8], b_vec[0:8])
        # Only a_vec gets written back as b_vec is not modified.
          # TeX: color line *
      # vvvvvvvvvvvvvvv
        mm256_storeu_ps(a[8 * ii + 0:8 * ii + 8], a_vec[0:8])
# TeX: end replace[0]

# TeX: version avx_add_full 5
# TeX: begin avx_add_full[0]
# The original proc
@proc
def simple_vec_add(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    assert N % 8 == 0
    for i in seq(0, N):
        a[i] += b[i]
# TeX: end avx_add_full[0]
# TeX: begin avx_add_full[1]
# Loop transformations, split i loop into outer ``ii'' and inner ``lane'' loops (8 lanes)
# Grab ``cursor'' (reference) to inner lane loop
avx_add = rename(simple_vec_add, "avx_add")
# TeX: color line *
#                                   b    bb    bbbb
avx_add = divide_loop(avx_add, "i", 8, ["ii", "lane"], perfect=True)
lane_loop = avx_add.find_loop("lane")
# TeX: end avx_add_full[1]

# TeX: begin avx_add_full[2]
# Cache 8-vectors of a and b, and used cached a_vec, b_vec within lane loop.
# This gives cursors to
#   * cache variable alloc
#   * loop loading into cache
#   * body of code using cache
#   * loop storing from cache
avx_add, (a_vec_alloc, a_vec_load, _, a_vec_store) = auto_stage_mem(
        avx_add, lane_loop, "a", "a_vec", rc=True)
avx_add, (b_vec_alloc, b_vec_load, _, _) = auto_stage_mem(
        avx_add, lane_loop, "b", "b_vec", rc=True)
avx_add = simplify(avx_add)
# TeX: end avx_add_full[2]

# TeX: begin avx_add_full[3]
# Change allocations to use AVX registers as memory type
# TeX: color line *
#                                          gggg
avx_add = set_memory(avx_add, a_vec_alloc, AVX2)
# TeX: color line *
#                                          gggg
avx_add = set_memory(avx_add, b_vec_alloc, AVX2)
# TeX: end avx_add_full[3]

# TeX: begin avx_add_full[4]
# Substitute AVX instructions at locations specified by cursors.
# TeX: color line *
#                                      vvvvvvvvvvvvvv
avx_add = replace(avx_add, a_vec_load, mm256_loadu_ps)
# TeX: color line *
#                                      vvvvvvvvvvvvvv
avx_add = replace(avx_add, b_vec_load, mm256_loadu_ps)
# TeX: color line *
#                                       vvvvvvvvvvvvvvv
avx_add = replace(avx_add, a_vec_store, mm256_storeu_ps)
# TeX: color line *
#                                     vvvvvvvvvvvvv
avx_add = replace(avx_add, lane_loop, mm256_iadd_ps)
# TeX: end avx_add_full[4]
