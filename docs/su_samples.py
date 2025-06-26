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

# TeX: version mbarrier_var 2
# TeX: begin mbarrier_var
cta_bars: barrier @ CudaMbarrier  # Currently no explicit array bounds
# TeX: end mbarrier_var[0]
warp_bars: barrier @ CudaMbarrier
for cta in cuda_threads(0, 8, unit=cuda_cta_in_cluster):
    # Each CTA owns one queue barrier in cta_bars
    # TeX: color line *
    #              rrr
    Await(cta_bars[cta], cuda_temporal, ~2)
    # TeX: color line *
    #                        vv
    for w in cuda_threads(0, nW, unit=cuda_warp):
        # Each warp owns one queue barrier in warp_bars, i.e. all else being equal,
        # TeX: color line *
        #         vv
        # there's nW times as many mbarriers in warp_bars as in cta_bars
        # TeX: color line *
        #               rrr  r
        Await(warp_bars[cta, w], cuda_temporal, ~2)
    # ...
# TeX: end mbarrier_var[1:]

# TeX: version intro_multicast_mbarrier 1
# TeX: begin intro_multicast_mbarrier[0]
# 4 x 2 grid of CTAs in cluster (assume clusterDim = 8).
ringbar: barrier @ CudaMbarrier
for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
    for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
        # TeX: color line intro_multicast_mbarrier[0]
        #           bbbbbbbbbb
        # Arrive on this CTA's mbarriers
        # TeX: color line intro_multicast_mbarrier[0]
        #                                   bbbbbbbbbbbb
        Arrive(cuda_in_order, 1) >> ringbar[m_cta, n_cta]
for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
    for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
        # Arrive on this CTA's mbarriers, as well as on the mbarriers of any
        # CTA with the same m_cta value (redundancy to be explained).
        # TeX: color line intro_multicast_mbarrier[0]
        #                                                                   b
        Arrive(cuda_in_order, 1) >> ringbar[m_cta, n_cta] >> ringbar[m_cta, :]
for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
    for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
        # Arrive on the mbarriers of any CTA with the same n_cta value
        # TeX: color line intro_multicast_mbarrier[0]
        #                                                            b
        Arrive(cuda_in_order, 1) >> ringbar[m_cta, n_cta] >> ringbar[:, n_cta]
for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
    for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
        # Arrive on any mbarriers named in either expression.
        # i.e., any with the same m_cta or the same n_cta.
        Arrive(cuda_in_order, 1) >> ringbar[m_cta, :] >> ringbar[:, n_cta]
# TeX: end intro_multicast_mbarrier[0]

# TeX: version multicast_mbarrier_dist 1
# TeX: begin multicast_mbarrier_dist[0]
# 4 x 2 grid of CTAs in cluster (assume clusterDim = 8). Let $B$ = blockDim
ringbar: barrier @ CudaMbarrier
# TeX: color line *
#   ggggg                                                                      gggg
for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):  # Thread pitch $2B$
    # TeX: color line *
    #   vvvvv                                                                  vvv
    for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):  # Thread pitch $B$
        # TeX: color line *
        #             ggggg  vvvvv                        gggggggggggggggggggggg  vvvvvvvvvvvvvvvvvvvvv
        Await(ringbar[m_cta, n_cta], cuda_in_order, 1)  # m_cta: $8B \mapsto 2B$, n_cta: $2B \mapsto B$
for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
    for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
        # Arrive on the mbarriers of any CTA with the same m_cta value
        Arrive(cuda_in_order, 1) >> ringbar[:, n_cta]
# TeX: end multicast_mbarrier_dist[0]

# TeX: version multicast_convergence 1
# TeX: begin multicast_convergence[0]
# TeX: color line *
#   ggggg
for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
    # TeX: color line *
  # rrrrrrrrrrrrrrrr
    if foo_condition:
        # TeX: color line *
        #   vvvvv
        for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
            with CudaWarps(4, 8):
                # This is invalid, because m_cta is multicast, and there's an if statement between
                # this arrive statement and the m_cta cuda_threads loop.
                # TeX: color line *
                #                                       v          g
                Arrive(cuda_in_order, 1) >> bar1[m_cta, :] >> bar1[:, n_cta]
                # This is valid. Only n_cta is multicast, and there's no forbidden statement
                # between here and the n_cta cuda_threads loop (CudaWarps is OK).
                # TeX: color line *
                #                                       v
                Arrive(cuda_in_order, 1) >> bar2[m_cta, :] >> bar2[m_cta, n_cta]
# TeX: end multicast_convergence[0]

# TeX: version multicast_2cta 3
# TeX: begin multicast_2cta
for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
    do_something_to(foo[cta, :])
    # TeX: color line *
    #  rrrrrrrr
    if cta == 0:
        # TeX: remark multicast_2cta[1]
        # Write to foo[0, :] will synchronize-with this arrive when cta=0
        # TeX: remark multicast_2cta[1]
        # Write to foo[1, :] will synchronize-with this arrive when cta=1,
        # TeX: remark multicast_2cta[1]
        # but this case is skipped by the above if statement.
        # TeX: remark multicast_2cta[2]
        # CTA 0 executes $\abs{C}$-many mbarrier.arrive instructions each on barriers[0] and barriers[1]
        # TeX: remark multicast_2cta[2]
        # CTA 1 does not execute this code
        Arrive(cuda_in_order, 1) >> barriers[cta] >> barriers[:]
        # TeX: remark multicast_2cta[1]
        # foo[0, :] will have pending arrives (barriers[0], 0) and (barriers[1], 0)
        # TeX: remark multicast_2cta[1]
        # foo[1, :] will have no pending arrives
    Await(barriers[cta], cuda_in_order, 1)
    # TeX: remark multicast_2cta[1]
    # At this point, the write to foo[0, :] will be visible to both CTA 0 and CTA 1,
    # TeX: remark multicast_2cta[1]
    # with the latter due to foo[0, :] containing pending arrive (barriers[1], 0)
    # TeX: remark multicast_2cta[1]
    # The write to foo[1, :] will still be visible only to CTA 1
    # TeX: remark multicast_2cta[2]
    # Both mbarriers have only $\abs{C}$ pending thread arrivals, but the
    # TeX: remark multicast_2cta[2]
    # expected thread arrival count is $2\abs{C}$. Deadlock.
# TeX: end multicast_2cta

# TeX: version multicast_loop_nest 2
# TeX: begin multicast_loop_nest[0]
# TeX: color line *
#   ggggg
for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
    # TeX: remark! *
    # Begin hypothetical compound statement
    # TeX: color line *
    #   vvvvv
    for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
        # TeX: color line *
        #                                                                   v
        Arrive(cuda_in_order, 1) >> ringbar[m_cta, n_cta] >> ringbar[m_cta, :]
    # TeX: remark! *
    # End compound statement
# TeX: end multicast_loop_nest[0]

# TeX: begin multicast_loop_nest[1]
# TeX: remark! *
# Begin hypothetical compound statement
# TeX: color line *
#   ggggg
for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
    # TeX: color line *
    #   vvvvv
    for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
        # TeX: color line *
        #                                          v             g
        Arrive(cuda_in_order, 1) >> ringbar[m_cta, :] >> ringbar[:, n_cta]
# TeX: remark! *
# End compound statement
# TeX: end multicast_loop_nest[1]

# TeX: version multicast_pairing_fail 1
# TeX: begin multicast_pairing_fail[0]
with CudaWarps(name="producer"):
    for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
        for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
            ReverseAwait(ringbar[m_cta, n_cta], cuda_temporal, ~0)
            # ...
            # Will not arrive on ringbar[M, N] with M != m_cta and N != n_cta
            # TeX: color line *
            #                                b             b
            Arrive(..., 1) >> ringbar[m_cta, :] >> ringbar[:, n_cta]
with CudaWarps(name="consumer"):
    for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
        for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
            Await(ringbar[m_cta, n_cta], ..., ~0)
            # ...
            # Error: mismatch with paired arrive:
            # Will match ringbar[M, N] with M != m_cta and N != n_cta
            # TeX: color line *
            #                                                         rrrr
            ReverseArrive(..., 1) >> ringbar[m_cta, n_cta] >> ringbar[:, :]

# TeX: end multicast_pairing_fail[0]

# TeX: version tma_2_arrives 1
# TeX: begin tma_2_arrives[0]
Sm90_copy_tensor_to_smem_linear_2f32(...)
Arrive(tma_to_smem, 1) >> barrier_A
Arrive(tma_to_smem, 1) >> barrier_B
# ...
with CudaWarps(...):
    Await(barrier_A, cuda_in_order, 1)
    # ... stuff reliant on the TMA to complete
with CudaWarps(...):
    Await(barrier_B, cuda_in_order, 1)
    # ... stuff also reliant on the TMA to complete
# TeX: end tma_2_arrives[0]

# TeX: version tma_instr_barrier 1
# TeX: begin tma_instr_barrier[0]
@instr
class tma_multicast_f32_2d_linear:
    def behavior(
            n_cta: size, box0: size, box1: size,
            dst: [f32][distributed: n_cta, dense: box0, box1] @ CudaSmemLinear,
            src: [f32][box0, box1],
            # TeX: color line *
          # rrr
            bar: barrier @ CudaMbarrier):
        for cta in cuda_threads(0, n_cta, unit=cuda_warp_in_cluster):
            distribute(dst[cta, :, :])
            # TeX: color line *
            #          rrrrrrrr
            distribute(bar[cta])  # Multicast to n_cta-many adjacent CTA's mbarrier
            for i0 in seq(0, box0):
                for i1 in seq(0, box1):
                    dst[cta, i0, i1] = src[i0, i1]
# TeX: end tma_instr_barrier[0]


# TeX: version tma_pairing_multicast 1
# TeX: begin tma_pairing_multicast[0]
with CudaAsync(name="producer"):
    for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
        for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
            ReverseAwait(ringbar[m_cta, n_cta], cuda_temporal, 1)
    for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
        # TeX: color line *
        #                     ggggggggggggggggg
        tma_foo_instr(...) >> ringbar[m_cta, :]
    for n_cta in cuda_threads(0, 2,
            unit=CollUnit((2, blockDim), (1, blockDim), "every_other_cta", None)):
            # We need better syntax to express ``every other CTA''
        # TeX: color line *
        #                     vvvvvvvvvvvvvvvvv
        tma_bar_instr(...) >> ringbar[:, n_cta]
    for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
        for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
            # Both queue barrier expressions are needed, otherwise one of the previous
            # two TMA instructions' barriers won't be a subset of those named here.
            # TeX: color line *
            #                           ggggggggggggggggg    vvvvvvvvvvvvvvvvv
            Arrive(cuda_temporal, 1) >> ringbar[m_cta, :] >> ringbar[:, n_cta]
with CudaWarps(name="consumer"):
    for m_cta in cuda_threads(0, 4, unit=2 * cuda_cta_in_cluster):
        for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
            Await(ringbar[m_cta, n_cta], cuda_in_order, 1)
            # ...
            # The barriers expressions here are needed to match the previous Arrive statement
            # TeX: color line *
            #                                  ggggggggggggggggg    vvvvvvvvvvvvvvvvv
            ReverseArrive(cuda_in_order, 1) >> ringbar[m_cta, :] >> ringbar[:, n_cta]
# TeX: end tma_pairing_multicast[0]


# TeX: version tl_grammar 1
# TeX: begin tl_grammar
# Instruction Timeline
instr_tl = cpu_in_order_instr
         | cuda_in_order_instr
         | Sm80_cp_async_instr
         | tma_to_smem_async_instr
         | tma_to_gmem_async_instr
         | wgmma_async_instr
         | tcgen05_async_instr  # Proposed for Blackwell
# TeX: filbreak
# Usage Timeline
usage_tl = cpu_usage
         | cuda_ram_usage
         | cuda_sync_rmem_usage
         | cuda_async_a_rmem_usage
         | cuda_async_d_rmem_usage
         | cuda_async_a_tmem_usage  # Proposed for Blackwell
         | cuda_async_d_tmem_usage  # Proposed for Blackwell
# TeX: filbreak
qual_tl = (instr_tl instr_tl, usage_tl usage_tl)  # Qualitative Timeline
# TeX: filbreak
tl_sig = (instr_tl instr_tl, usage_tl usage_tl, int tid)  #  Timeline Signature
# TeX: filbreak
# Synchronization Timeline
sync_tl = (qual_tl* full_timeline_set,      # $L_1^F, L_2^F$
           qual_tl* temporal_timeline_set,  # $L_1^T, L_2^T$
           bool V1_transitive)
# TeX: end tl_grammar

# TeX: version abstract_machine_instructions 1
# TeX: begin abstract_machine_instructions
spork_instr = (instr_tl         instr_tl,  # $L^i$
               spork_fnarg*     arg_info)
# TeX: filbreak
fnarg_mode = ReadArg() | MutateArg() | ExemptArg()
# TeX: filbreak
spork_fnarg = (
      sym          name,          # name: basetype[shape...]
      basetype     basetype,
      expr*        shape,
      fnarg_mode   mode,
      usage_tl     usage_tl,      # $L^u$
      instr_tl*    ext_instr_tl,  # $L_X^i$
      usage_tl*    ext_usage_tl,  # $L_X^u$
      bool         out_of_order,  # $OOO$
      bool         access_by_owner_only)
# TeX: filbreak
barrier_param = (queue_barrier_array_id barriers, sym* iterators, multicast_flags multicast)
# TeX: filbreak
multicast_flags = (bool* flags)
# TeX: filbreak
expr = # Similar to Exo LoopIR
# TeX: filbreak
stmt = # Similar to Exo LoopIR, plus extra statements
  | AllocBarrierArray(queue_barrier_array_id barriers, int* shape)
# TeX: filbreak
  | FreeBarrierArray(queue_barrier_array_id barriers)
# TeX: filbreak
  | CheckAllocShard(
      sym       name,
      sym*      distributed_iterators)
# TeX: filbreak
  | CheckFreeShard(
      sym       name,
      sym*      distributed_iterators,
      instr_tl* ext_instr_tl,        # $L_X^i$
      usage_tl* ext_usage_tl)        # $L_X^u$
# TeX: filbreak
  | CheckRead(
      sym             name,
      expr*           idx,
      instr_tl        instr_tl,      # $L^i$
      usage_tl        usage_tl,      # $L^u$
      instr_tl*       ext_instr_tl,  # $L_X^i$
      usage_tl*       ext_usage_tl,  # $L_X^u$
      bool            out_of_order)  # $OOO$
# TeX: filbreak
  | CheckMutate(
      sym             name,
      expr*           idx,
      instr_tl        instr_tl,      # $L^i$
      usage_tl        usage_tl,      # $L^u$
      instr_tl*       ext_instr_tl,  # $L_X^i$
      usage_tl*       ext_usage_tl,  # $L_X^u$
      bool            out_of_order)  # $OOO$
# TeX: filbreak
  | CheckInstr(
      spork_instr     instr_info,
      expr*           args,
      barrier_param?  barrier)
# TeX: filbreak
  | Fence(
      sync_tl                 first_sync_timeline,  # $L_1$
      sync_tl                 second_sync_timeline) # $L_2$
# TeX: filbreak
  | Arrive(
      sync_tl                 first_sync_timeline,  # $L_1$
      int                     N,
      queue_barrier_array_id  barriers,
      sym*                    iterators,
      multicast_flags*        multicasts)
# TeX: filbreak
  | Await(
      queue_barrier_array_id  barriers,
      sym*                    iterators,
      sync_tl                 second_sync_timeline, # $L_2$
      int                     N)
# TeX: end abstract_machine_instructions

# TeX: version abstract_machine_variables 1
# TeX: begin abstract_machine_variables
position_id = (sym name, int* idx)
# TeX: filbreak
position = (number              numeric_value,
            assignment_record   assignment_record,
            int*                owner_tids)
# TeX: filbreak
assignment_record = (mutate_vis_record? mutate, read_vis_record* reads)
# TeX: filbreak
# Read Visibility Record
read_vis_record = (
      tl_sig*           sync_visibility_set,  # $V_S$
      tl_sig*           async_visibility_set, # $V_A$
      qual_tl           original_qual_tl,     # $L_O$
      pending_arrive*   pending_arrives)      # $A_p$
# TeX: filbreak
# Mutation Visibility Record
mutate_vis_record = (
      tl_sig*           sync_visibility_set,  # $V_S$
      tl_sig*           async_visibility_set, # $V_A$
      qual_tl           original_qual_tl,     # $L_O$
      pending_arrive*   pending_arrives)      # $A_p$
# TeX: filbreak
queue_barrier_id = (queue_barrier_array_id barriers, int* iterators)
# TeX: filbreak
queue_barrier = (int arrive_count, int await_count)  # $(Q^1, Q^2)$
# TeX: filbreak
pending_arrive = (queue_barrier_id barrier, int arrive_count)  # $(\text{id of }Q, Q^1)$
# TeX: end abstract_machine_variables
