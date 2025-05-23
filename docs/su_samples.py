# TeX: version instr_class 3

# TeX: begin instr_class[0]
@instr
class Sm80_mma_load_a_tf32:
    # PARSED as Exo code; specifies instr behavior for LoopIR_unification
    def behavior(
        K: size,
        rmem: [f32][16, K] @ Sm80_RmemMatrixA,
        smem: [f32][16, K] @ CudaSmemLinear,
    ):
        # Essentially, this is what would have gone into the function body of an old-style instr
        for m in seq(0, 16):
            for k in seq(0, K):
                rmem[m, k] = smem[m, k]

    # EXECUTED as Python code; configures information needed for
    # compiler/codegen, and synchronization checking.
    def instance(self, K):
        # Essentially, this is what would have gone into the @instr(...) for old-style instr
        self.instr_format = # ...
        self.c_includes = # ...
        # ... but also a lot of new information
        self.coll_unit = cuda_warp
        # ...
# TeX: end instr_class[0]

# TeX: begin instr_class[1]
# TeX: color line *
#                                     ggg
concrete_instr = Sm80_mma_load_a_tf32(K=8)

# Behavior: substitute K=8, delete K argument
# TeX: color line *
#                                  g
def concrete_instr(rmem: [f32][16, 8] @ Sm80_RmemMatrixA,
# TeX: color line *
#                                  g
                   smem: [f32][16, 8] @ CudaSmemLinear,
    for m in seq(0, 16):
        for k in seq(0, 8):
            rmem[m, k] = smem[m, k]
# TeX: end instr_class[1]

# TeX: begin instr_class[2]
Sm80_mma_load_a_tf32(A_rmem[k_seq,:,:],  # Runtime: RMEM dst
                     A_smem[1 - k1 % 2,  # Runtime: SMEM src
                            mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                            k_seq*MMA_K:(k_seq+1)*MMA_K],
# TeX: color line *
#                    ggggggg
                     K=MMA_K)  # Template: K
# TeX: end instr_class[2]


# TeX: version avx_example 1
# TeX: begin avx_example
# TeX: color line *
#        y  y  b                                  b
foo: f32[4, 6, 8] @ AVX2  # Expected vector size: 8
# Lowered C++
# TeX: color line *
# bbbbbb     y  y
  __m256 foo[4][6];
# TeX: end avx_example

# TeX: version warp_example 1
# TeX: begin warp_example
# TeX: color line *
#        r  r  y  bb  b                                              bbbbbb
foo: f32[2, 4, 6, 16, 8] @ Sm80_RmemMatrixD  # Expected matrix tile: 16 x 8
for mw in cuda_threads(0, 2, unit=4 * cuda_warp):
    for nw in cuda_threads(0, 4, unit=cuda_warp):
        for array_idx in seq(0, 6):
            # Ownership of foo distributed to 2 x 4 grid of warps
            # TeX: color line *
            #                        rr  rr
            Sm80_mma_zero_d_tf32(foo[mw, nw, array_idx, :, :])
# Lowered C++
# TeX: color line *
#              y  b                             rrrrr
  unsigned foo[6][4];  // Executed uniformly by 2 x 4 grid of warps
# TeX: color line *
            # bbbbbb               b
           // 16 x 8 matrix tile = 4 registers per thread (32 threads/warp total)
  for (int array_idx = 0; array_idx < 6; array_idx++) {
    exo_CudaUtil::Sm80_mma_zero_d(foo[array_idx]);
  }
# TeX: end warp_example

# TeX: version avx_window 1
# TeX: begin avx_window
# TeX: color line *
#                 b
src: [f32][packed:8] @ AVX2
# TeX: end avx_window

# TeX: version tma_window 2
# TeX: begin tma_window[0]
# Innermost dimension cannot be strided
# TeX: color line *
#                  yyyy        yyyy
src: [f32][strided:box0, dense:box1] @ Sm90_CUtensorMap
# TeX: end tma_window[0]
# TeX: begin tma_window[1]
# TeX: color line *
#                           bbbbbbbb                          yyyyyyy
# Hardware defines swizzled 8 x box1 format; SMEM dst must be densely packed
# TeX: color line *
#                   rrrrrrr
# TMA multicasts to 2 CTAs' shared memory
# TeX: color line *
#                      r         yyyyyy         b  bbbb
dst: [f32][distributed:2, dense: box0/8, packed:8, box1] @ Sm90_CUtensorMap
# TeX: end tma_window[1]



# TeX: version pointer_strides 1
# TeX: begin pointer_strides
foo = bar[4:8, 6, :]  # Freely mix points/intervals

struct <sname>
{
    float * data;
    int_fast32_t strides[...];
};
float alloc[dim0 * dim1 * ... ];
# TeX: end pointer_strides

# TeX: version register_alloc 1
# TeX: begin register_alloc
float alloc[dim0][dim1][...];
# TeX: end register_alloc

# TeX: version pointer_offsets 1
# TeX: begin pointer_offsets
foo = bar[4:8, 6:, :]  # Intervals only
struct <sname>
{
    <pointer type> data;
    int32_t offsets[...];
};
# TeX: end pointer_offsets


# TeX: version 2f2c 1
# TeX: begin 2f2c
struct exo_win_2f32c{
    const float * const data;
    const int_fast32_t strides[2];
};
# TeX: end 2f2c

# TeX: version avx_before 1
# TeX: begin avx_before[0]
    def window(cls, basetyp, baseptr, indices, strides, srcinfo)
        # ...
        idxs = indices[:-1] or ""
        if idxs:
            idxs = "[" + "][".join(idxs) + "]"
    # ...
    foo[1,3,:]    # Lowers to foo[1][3]
    foo[1:9,3,6]  # Also lowers to foo[1][3]
# TeX: end avx_before[0]

# TeX: version is_win 1
# TeX: begin is_win[0]
    @classmethod
    # TeX: color line *
    #               yyyyyyyyyyyyyyyyyyyyyy
    def window(cls, basetyp: LoopIR.Tensor, baseptr, indices, strides, srcinfo):
        offset = generate_offset(indices, strides)

        # TeX: color line *
        #  yyyyyyyyyyyyyyyy
        if basetyp.is_win():
            baseptr = f"{baseptr}.data"

        return f"{baseptr}[{offset}]"
# TeX: end is_win[0]

# TeX: version tma_proc 1
# TeX: begin tma_proc[0]
@proc
# TeX: color line *
#            ggggggggggg
def tma_proc(tensorMap_a: [f32][128, 32] @ Sm90_tensorMap(128, 128, 32),
# TeX: color line *
#            vvvvvvvvvvv
             tensorMap_b: [f32][256, 32] @ Sm90_tensorMap(128, 256, 32)):

void tma_proc( void *ctxt,
# TeX: color line *
#             gggggggggggggggggggg                                                ggggggggggg
  CUtensorMap exo_data_tensorMap_a, struct exo_win_2f32_Sm90_tensorMap_128_128_32 tensorMap_a,
# TeX: color line *
#             vvvvvvvvvvvvvvvvvvvv                                                vvvvvvvvvvv
  CUtensorMap exo_data_tensorMap_b, struct exo_win_2f32_Sm90_tensorMap_128_256_32 tensorMap_b
) {
# TeX: end tma_proc[0]

# TeX: version encode_window 1
# TeX: begin encode_window
(struct {sname}) {{
    {dataptr} + ({strides[0]}) * ({array_offsets[0]}) + ({strides[1]}) * ({array_offsets[1]}) + ...,
    {{ ({strides[0]}), ({strides[1]}), ... }}
}}
# TeX: end encode_window

# TeX: version shfl_sync 1
# TeX: begin shfl_sync[0]
@instr
class cuda_shfl_sync_f32:
    def behavior(
    # TeX: color line *
    #                              rr                                     rr
            dst: [f32][distributed:32] @ CudaRmem, src: [f32][distributed:32] @ CudaRmem,
            i: idx):
        for tid in cuda_threads(0, 32, unit=cuda_thread):
            dst[tid] = src[i]
            # The dst register is the same as the one owned by the current thread
            # TeX: color line *
            #              rrr
            distribute(dst[tid])
            # src[tid] owned by thread tid of 32 in warp; not the same as the src[i] we peeked
            # TeX: color line *
            #              rrr
            distribute(src[tid])

    def instance(self):
        self.instr_format = "{dst_data} =__shfl_sync(0xFFFF'FFFF, {src_data}, {i});"
        self.coll_unit = cuda_warp
# TeX: end shfl_sync[0]

# TeX: version tma_instr 1
# TeX: begin tma_instr[0]
@instr
class tma_multicast_f32_2d_linear:
    def behavior(
            n_cta: size, box0: size, box1: size,
            # TeX: color line *
            #                       rrrrr         yyyy  yyyy
            dst: [f32][distributed: n_cta, dense: box0, box1] @ CudaSmemLinear,
            # TeX: color line *
            #          yyyy  yyyy
            src: [f32][box0, box1]):
        for cta in cuda_threads(0, n_cta, unit=cuda_warp_in_cluster):
            # TeX: color line *
            #              rrr
            distribute(dst[cta, :, :])
            for i0 in seq(0, box0):
                for i1 in seq(0, box1):
                    dst[cta, i0, i1] = src[i0, i1]

    def instance(self, n_cta, box0, box1):
        self.instr_format = # ... this is why I want the callback
        # 1 warp each selected from n_cta-many CTAs participate in multicast
        self.coll_unit = n_cta * cuda_warp_in_cluster
        # this syntax may change
        self.access_info["src"].mem = Sm90_tensorMap(0, box0 // n_cta, box1)
        # ...
# TeX: end tma_instr[0]
