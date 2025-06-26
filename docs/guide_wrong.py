# Snippets of code for spork_guide.pdf that are not expected to compile.

# TeX: version loop_mode_syntax 1
# TeX: begin loop_mode_syntax[0]
# TeX: color line *
#           gggggggggggggg         ggggggggggggggggggggggggggggg
for iter in loop-mode-name(lo, hi, optional-keyword=optional-arg, ...):
# TeX: end loop_mode_syntax[0]

# TeX: version cuda_threads_syntax 3
# TeX: begin cuda_threads_syntax[0]
for iter in cuda_threads(0, hi, unit=<collective-unit>):
    ...body
# TeX: end cuda_threads_syntax[0]

# TeX: version why_dist 1
# TeX: begin why_dist[0]
# TeX: remark! *
# Interpreted with M-semantics (multithreaded)
# TeX: color line *
#                                                rrrrrrrrrrrrrrrrrrr
x: f32 @ CudaRmem  # implicitly one x per thread (not valid Exo-GPU)
for tid in cuda_threads(0, 2, unit=cuda_thread):
    x = a[tid]  # x = a[0] in thread 0; x = a[1] in thread 1
for tid in cuda_threads(0, 2, unit=cuda_thread):
    # TeX: color line *
    #             ggggggggggg
    b[tid] = x  # b[0] = a[0]; b[1] = a[1]

# TeX: remark! *
# Interpreted with S-semantics (single-threaded)
# TeX: color line *
#                                                rrrrrrrrrrrrrrrrrrr
x: f32 @ CudaRmem  # implicitly one x per thread (not valid Exo-GPU)
for tid in cuda_threads(0, 2, unit=cuda_thread):
    x = a[tid]  # x = a[1], overwrites x = a[0]
for tid in cuda_threads(0, 2, unit=cuda_thread):
    # TeX: color line *
    #             rrrrrrrrrrr
    b[tid] = x  # b[0] = a[1]; b[1] = a[1]

# TeX: remark! *
# Correct Exo-GPU
x[2]: f32 @ CudaRmem  # We explicitly model there are 2 x's, distributed into 2 threads
for tid in cuda_threads(0, 2, unit=cuda_thread):
    x[tid] = a[tid]  # x[0] = a[0]; x[1] = a[1]
for tid in cuda_threads(0, 2, unit=cuda_thread):
    b[tid] = x[tid]  # b[0] = a[0]; b[1] = a[1]
# TeX: end why_dist[0]


@proc
def mbarrier_teams():
    with CudaDeviceFunction(blockDim=512):
        for task in cuda_tasks(0, xyzzy):
            # TeX: version mbarrier_teams 1
            # TeX: begin mbarrier_teams[0]
            foo: f32[2, 128] @ CudaSmemLinear
            my_mbarrier: barrier[2] @ CudaMbarrier
            # Split into two teams of 256 threads each
            for team in cuda_threads(0, 2, unit=256 * cuda_thread):
                with CudaWarps(0, 4):
                    for tid in cuda_threads(0, 128, unit=cuda_thread):
                        foo[team, tid] = 137
                    # Threads [0, 127] arrive on my_mbarrier[0]; threads [256, 383] arrive on my_mbarrier[1]
                    Arrive(cuda_in_order, my_mbarrier[team], 1)
            for team in cuda_threads(0, 2, unit=256 * cuda_thread):
                with CudaWarps(4, 8):
                    # Threads [128, 255] wait for my_mbarrier[0]; threads [384, 511] wait for my_mbarrier[1]
                    Await(my_mbarrier[team], cuda_in_order, ~0)
                    for tid in cuda_threads(0, 128, unit=cuda_thread):
                        bar: f32 @ CudaRmem
                        bar = foo[team, tid]
            # TeX: end mbarrier_teams[0]


# TeX: version with_grammar 1
# TeX: begin with_grammar[0]
# First 2 are for metaprogramming
<with-statement> ::=
    with exo : <body>
  | with python: <body>
  | with <with-context>: <body>
# TeX: filbreak
<with-context> ::= <async-ctx> | <warps-ctx>
# TeX: filbreak
# These make the with statement into an async block
<async-ctx> ::=
    CudaDeviceFunction(<clusterDim><blocks-per-sm> blockDim=<int>)
  | CudaDeviceFunction(<clusterDim><blocks-per-sm> warp_config=[<warp-configs>])
  | CudaAsync(<async-instr-tl>)
# TeX: filbreak
# These do not make the with statement into an async block
<warps-ctx> ::=
    CudaWarps(<int>, <int>)  # (lo, hi)
  | CudaWarps(name=<pystr>)  # name must match one of the warp-config
  | CudaWarps(<int>, <int>, name=<pystr>)  # (lo, hi, name=name)
# TeX: filbreak
<clusterDim> ::= clusterDim=<int>, |  # Defaults to 1 if not given
# TeX: filbreak
<blocks-per-sm> ::= blocks_per_sm=<int>, |  # Defaults to 1 if not given
# TeX: filbreak
<warp-config> ::=
    CudaWarpConfig(<pystr>, <int>)  # (Warp name, warp count)
  | CudaWarpConfig(<pystr>, <int>, setmaxnreg_dec=<int>)
  | CudaWarpConfig(<pystr>, <int>, setmaxnreg_inc=<int>)
# TeX: filbreak
<warp-configs> ::= <warp-config> | <warp-config>, <warp-configs>
# TeX: end with_grammar[0]

# TeX: version for_grammar 1
# TeX: begin for_grammar[0]
<for-loop> ::=
    for <name> in seq(<expr>, <expr>): <body>
  | for <name> in seq(<expr>, <expr>, pragma_unroll=<int>): <body>
  | for <name> in par(<expr>, <expr>): <body>  # OpenMP, predates Exo-GPU
  | for <name> in cuda_tasks(<expr>, <expr>): <body>
  | for <name> in cuda_threads(0, <int>, unit=<coll-unit>): <body>
# TeX: filbreak
<loop-mode> ::= seq | par | cuda_tasks | cuda_threads
# Each ``for <name>'' loop defines a new variable of type index; we call this a
# <loop-mode> iterator, with <loop-mode> being that of the defining for loop.
# TeX: end for_grammar[0]

# TeX: version coll_grammar 1
# TeX: begin coll_grammar[0]
<coll-unit> ::= cuda_thread | cuda_warp | cuda_warpgroup | cuda_cta_in_cluster
  | <int> * <coll-unit>
  | CollUnit([<coll-size-exprs>], [<coll-size-exprs>], <pystr>)
# TeX: filbreak
<coll-size-expr> ::= clusterDim | blockDim | <int>
# TeX: filbreak
<coll-size-exprs> ::= <coll-size-expr> | <coll-size-exprs>, <coll-size-expr>
# TeX: end coll_grammar[0]

# TeX: version sync_grammar 1
# TeX: begin sync_grammar[0]
<instr-tl> ::= cpu_in_order_instr | cuda_in_order_instr | <async-instr-tl>
# TeX: filbreak
<async-instr-tl> ::= Sm80_cp_async_instr
                     | tma_to_smem_async_instr
                     | tma_to_gmem_async_instr
                     | wgmma_async_instr
# TeX: filbreak
<usage-tl> ::= # TODO list them all
# TeX: filbreak
<sync-tl> ::= # TODO list them all
# TeX: filbreak
<barrier-mem> ::= CudaMbarrier | CudaCommitGroup | CudaEvent
# TeX: filbreak
<barrier-alloc> ::= <name> : barrier [<barrier-array-shape>] @ <barrier-mem>
                  | <name> : barrier @ <barrier-mem>
# TeX: filbreak
<barrier-array-shape> ::= <int> | <int> , <barrier-array-shape>
# TeX: filbreak
<sync-stmt> ::= <fence-stmt> | <arrive-stmt> | <await-stmt>
# TeX: filbreak
<fence-stmt> ::= Fence(<sync-tl>, <sync-tl>)  # (first timeline, second timeline)
# TeX: filbreak
<arrive-fname> ::= Arrive | ReverseArrive  # TODO consider changing this
# TeX: filbreak
<await-fname> ::= Await | ReverseAwait  # TODO consider changing this
# TeX: filbreak
# (first timeline, N)
<arrive-stmt> ::= <arrive-fname>(<sync-tl>, <int>) <trailing-barrier-exprs>
# TeX: filbreak
# (barrier, second timeline, N)
<await-stmt> ::= <await-fname>(<barrier-expr>, <sync-tl>, <int>)
# TeX: filbreak
<barrier-expr> ::= <name> | <name>[<barrier-idxs>]  # <name> : barrier
# TeX: filbreak
<barrier-idx> ::= : | <name>  # where <name> identifies a cuda_threads-iterator
# TeX: filbreak
<barrier-idxs> ::= <barrier-idx> | <barrier-idx>, <barrier-idxs>
# TeX: filbreak
<trailing-barrier-exprs> ::= >> <barrier-expr> | <trailing-barrier-exprs> >> <barrier-expr>
# TeX: end sync_grammar[0]

# TeX: version misc_grammar 1
# TeX: begin misc_grammar[0]
<special-window> ::=
    Sm90_tensorMap(<swizzle>, <int>)  # 1D box
  | Sm90_tensorMap(<swizzle>, <int>, <int>)  # 2D box
  | Sm90_tensorMap(<swizzle>, <int>, <int>, <int>)  # 3D box
  | Sm90_tensorMap(<swizzle>, <int>, <int>, <int>, <int>)  # 4D box
  | Sm90_tensorMap(<swizzle>, <int>, <int>, <int>, <int>, <int>)  # 5D box
# TeX: filbreak
<swizzle> ::= 0 | 32 | 64 | 128
# TeX: filbreak
<window-stmt> ::= <normal-window-stmt> | <special-window-stmt>
# TeX: filbreak
<normal-window-stmt> ::= <name> = <window-expr>
# TeX: filbreak
<special-window-stmt> ::= <name> = <window-expr> @ <special-window>
# TeX: filbreak
<window-expr> ::= # as defined in Exo currently
# TeX: filbreak
<call> ::= <normal-call> | <call-with-barrier>
# TeX: filbreak
<call-with-barrier> ::= <normal-call> >> <barrier-expr>
# TeX: filbreak
<normal-call> ::= # function call as defined in Exo currently
# TeX: end misc_grammar[0]
