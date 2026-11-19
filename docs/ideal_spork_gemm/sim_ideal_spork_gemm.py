# Simulator

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Union, Dict

deviations: List[str]
deviations = []

# Only dependence on Exo.
# Otherwise this idealized simulator is quitting Exo-GPU cold-turkey.
# As typical, I am hypocritical and import the private internals.
from exo.spork.timelines import (
    cuda_in_order,
    cuda_in_order_ram_qual,
    cuda_in_order_rmem_qual,
    Sm80_cp_async,
    Sm80_cp_async_qual,
    cuda_async_proxy_retired_qual,
    tma_to_smem_async,
    tma_to_smem_async_qual,
    tma_to_gmem_async,
    tma_to_gmem_async_qual,
    wgmma_fence_1,
    wgmma_fence_2,
    wgmma_async,
    wgmma_rmem_fenced_qual,
    wgmma_async_rmem_a_qual,
    wgmma_async_rmem_d_qual,
    wgmma_async_smem_qual,
    cuda_mbarrier_qual,
    Qual_tl,
    Sync_tl,
)


Qtl = Qual_tl

Qtl_mask_rmem = frozenset({
    cuda_in_order_rmem_qual,
    wgmma_async_rmem_a_qual,
    wgmma_async_rmem_d_qual,
    wgmma_rmem_fenced_qual
})

Qtl_mask_data_ram = frozenset({
    cuda_in_order_ram_qual,
    Sm80_cp_async_qual,
    cuda_async_proxy_retired_qual,
    tma_to_smem_async_qual,
    tma_to_gmem_async_qual,
    wgmma_async_smem_qual
})

Qtl_mask_mbarrier = frozenset({cuda_mbarrier_qual})

deviations.append("Observer thread ID is (CTA rank, threadIdx), not task ID + local thread index")

@dataclass(slots=True, frozen=True)
class Ob:
    """Observer"""
    cta_rank: int
    threadIdx: int
    qtl: Qtl


def ob_sort_key(ob: Ob):
    return (str(ob.qtl), ob.cta_rank, ob.threadIdx)



@dataclass(slots=True, frozen=True, eq=False)
class X:
    """For simple examples I plan to write, all allocs will be at the top-level,
    so we don't have to alpha-rename to distinguish same-named allocs in different
    loop iterations. But I add this X (var name) type anyway for now."""
    name: str


@dataclass(slots=True, frozen=True)
class Aw:
    """PendingAwait(x, idx, z_await)"""
    x: X
    idx: Tuple[int]
    z_await: int


@dataclass(slots=True, frozen=True)
class AwRay:
    """{PendingAwait(x, idx, z_await) \mid z_await >= await_inf}"""
    x: X
    idx: Tuple[int, ...]
    await_inf: int


infinity = float("inf")

def merge_aw_rays(a: Set[AwRay], b: Set[AwRay]):
    await_inf: Dict[Tuple[str, int], int]
    await_inf = {}
    for aw_param in (a, b):
        for aw_ray in aw_param:
            xz = (aw_ray.x, aw_ray.idx)
            await_inf[xz] = min(await_inf.get(xz, infinity), aw_ray.await_inf)
    return frozenset(AwRay(xz[0], xz[1], xz_inf) for (xz, xz_inf) in await_inf.items())


@dataclass(slots=True, frozen=True)
class EnvVis:
    ob: Set[Ob]  # frozenset really
    aw: Set[AwRay]  # frozenset really; no duplicate (x, idx) pairs.
    Qtl_mask: Set[Qtl]

    def augmented_ob(ob_out: Set[Ob]) -> "EnvVis":
        Qtl_mask = self.Qtl_mask
        ob_new = frozenset(ob for ob in (self.ob | ob_out) if ob.qtl in Qtl_mask)
        return EnvVis(ob_new, self.aw, Qtl_mask)

    def augmented_aw(aw_out: Set[AwRay]) -> "EnvVis":
        Qtl_mask = self.Qtl_mask
        aw_new = merge_aw_rays(self.aw, aw_out)
        return EnvVis(self.ob, aw_new, Qtl_mask)

    def ob_witness(ob_in: Set[Ob]) -> Optional[Ob]:
        intersection = self.ob & ob_in
        if not intersection:
            return None
        return min(intersection, key=ob_sort_key)

    def aw_witness(aw: Aw) -> Optional[Aw]:
        if self.aw.get((aw.x, aw.idx), infinity) <= aw.z_await:
            return aw
        return None


@dataclass(slots=True, frozen=True)
class M_task:
    """Symbolic evaluation: tasks[task_index].m"""
    task_index: int

    def __str__(self):
        return f"tasks[{self.task_index}].m"


@dataclass(slots=True, frozen=True)
class N_task:
    """Symbolic evaluation: tasks[task_index].n"""
    task_index: int

    def __str__(self):
        return f"tasks[{self.task_index}].n"


@dataclass(slots=True, frozen=True)
class A_tile:
    """Symbolic evaluation: A[tasks[task_index].m, k]"""
    task_index: int
    k: int

    def __str__(self):
        return f"A[tasks[{self.task_index}].m, {self.k}]"

    def __mul__(self, b):
        if b != B_tile(self.k, self.task_index):
            raise ValueError(f"Unexpected multiplication {self} * {b}")
        return AB_prod(self.task_index, self.k)


@dataclass(slots=True, frozen=True)
class B_tile:
    """Symbolic evaluation: B[k, tasks[task_index].n]"""
    k: int
    task_index: int

    def __str__(self):
        return f"B[{self.k}, tasks[{self.task_index}].n]"


@dataclass(slots=True, frozen=True)
class AB_prod:
    task_index: int
    k: int

    def __str__(self):
        return f"A[tasks[{self.task_index}].m, {self.k}] @ B[{self.k}, tasks[{self.task_index}].n]"

    def __radd__(self, zero):
        if zero == 0 and self.k == 0:
            return D_accum(self.task_index, 1)
        else:
            raise ValueError(f"Unexpected addition {zero} + {self}")


@dataclass(slots=True, frozen=True)
class D_accum:
    """Symbolic evaulation: A[tasks[task_index].m, 0:k] @ B[0:k, tasks[task_index].n]"""
    task_index: int
    k: int

    def __str__(self):
        if self.k == 0:
            return "0"
        else:
            return f"A[tasks[{self.task_index}].m, 0:{self.k}] @ B[0:{self.k}, tasks[{self.task_index}].n]"

    def __add__(self, ab):
        if ab != AB_prod(self.task_index, self.k):
            raise ValueError(f"Unexpected addition {self} + {ab}")
        return D_accum(self.task_index, self.k + 1)


@dataclass(slots=True, frozen=True)
class A_gmem:
    """Symbolic evaluation: magic A_gmem source."""
    def __getitem__(self, tup):
        m_task, k = tup
        assert isinstance(m_task, M_task)
        assert isinstance(k, int)
        return A_tile(m_task.task_index, k)


@dataclass(slots=True, frozen=True)
class B_gmem:
    """Symbolic evaluation: magic B_gmem source."""
    def __getitem__(self, tup):
        k, n_task = tup
        assert isinstance(k, int)
        assert isinstance(n_task, N_task)
        return B_tile(k, n_task.task_index)


@dataclass(slots=True, frozen=True)
class C_gmem:
    K: int

    """Symbolic evaluation: magic C_gmem destination."""
    def __setitem__(self, tup, rhs):
        m, n = tup
        assert isinstance(m, M_task)
        assert isinstance(n, N_task)
        assert m.task_index == n.task_index
        expect_rhs = D_accum(m.task_index, self.K)
        if rhs != expect_rhs:
            raise ValueError(f"Invalid assignment to C[{m}, {n}]; got {rhs}, expected {expect_rhs}")


@dataclass(slots=True, frozen=True)
class SrcOrigin:
    # TODO needs to encode the stmt and loop iterator values
    pass


@dataclass(slots=True, frozen=True)
class WrappedEnvVis:
    is_mutate: bool
    origin: SrcOrigin
    vis: EnvVis


@dataclass(slots=True)
class ArrayElement:
    x: X
    idx: Tuple[int, ...]
    value: object  # int or symbolic evaluation thing
    value_origin: Optional[SrcOrigin]
    vis_list: List[WrappedEnvVis]

    def map_vis(self, vis_lambda):
        self.vis_list = [WrappedEnvVis(w.is_mutate, w.origin, vis_lambda(w.vis)) for w in self.vis_list]


@dataclass(slots=True)
class ArrayLevel:
    entries: List[Union[ArrayElement, "ArrayLevel"]]

    def map_vis(self, vis_lambda):
        for e in entries:
            e.map_vis(vis_lambda)


@dataclass(slots=True)
class Array:
    context: "ProgramContext"
    x: X
    dims: int
    entries: Union[ArrayElement, ArrayLevel]
    Qtl_mask: Set[Qtl]
    qtl_convergent: bool
    expected_arrive_count: Optional[int]

    def __call__(
        self,
        Qtl_pre: Set[Qtl],
        qtl_init: Optional[Qtl],  # None for sync exempt
        Qtl_mask: Set[Qtl],
        arrive_on: Optional[Union[ArrayElement, str]]=None  # Special value "itself"
    ):
        assert qtl_init is None or Qtl_mask == self.Qtl_mask
        ctx = self.context
        return ArrayAccessor(self, ctx.cta_rank_slice, ctx.threadIdx_slice, Qtl_pre, qtl_init, Qtl_mask, arrive_on)

    def __getitem__(self, idx):
        raise ValueError(f"Require x(Qtl_pre, qtl_init, Qtl_mask)[idx]")

    def __setitem__(self, idx, value):
        raise ValueError(f"Require x(Qtl_pre, qtl_init, Qtl_mask)[idx]")

    def map_vis(self, vis_lambda):
        self.entries.map_vis(vis_lambda)

    def prepare_mbarrier_phase(self, *idx):
        """prev_phase = x[idx[:-1] + (idx[-1] - 1,)], i.e. 1 previous on last (phase) dimension;
        doesn't exist if idx[-1] = 0.
        Require prev_phase has been prepared (if it exists) and x[idx] has not been prepared.
        Adds obligation to await on prev_phase, if it exists,
        and that all accesses to prev_phase eventually arrive on x[idx]
        TODO free checks.
        """

        src_origin = SrcOrigin()  # XXX todo

        expected_arrive_count = self.expected_arrive_count
        assert isinstance(expected_arrive_count, int)

        try:
            idx_dims = len(idx)
        except TypeError:
            idx_dims = 1
            idx = (idx, )

        phase = idx[-1]
        assert phase >= 0

        assert len(idx) == self.dims, f"idx={idx} has wrong dimensionality; expect {self.dims}"

        level = self.entries
        for i in idx[:-1]:
            level = level.entries[i]

        phases = len(level.entries)
        if phases != phase:
            raise ValueError(f"prepare_mbarrier_phase expects phase={phases}")

        vis_list = []

        if phase > 0:
            aw_out = frozenset((AwRay(self.x, idx[-1] + (phase - 1,), expected_arrive_count - 1), ))
            env_vis = EnvVis(frozenset(), aw_out, Qtl_mask_mbarrier)
            vis_list.append(WrappedEnvVis(True, src_origin, env_vis))

        # Bottom-level ArrayLevel stores mbarriers indexed by phase (last coordinate)
        # Doesn't exist until it's initialized.
        level.entries.append(ArrayElement(self.x, idx, 0, src_origin, vis_list))



deviations.append("TODO mbarrier free checks")
deviations.append("qtl_convergent = False case (non-registers) omits all but the lowest and highest thread")


@dataclass(slots=True, frozen=True)
class ArrayAccessor:
    array: Array
    cta_rank_slice: slice
    threadIdx_slice: slice
    Qtl_pre: Set[Qtl]
    qtl_init: Optional[Qtl]  # None for sync exempt
    Qtl_mask: Set[Qtl]
    arrive_on: Optional[ArrayElement]

    def __getitem__(self, idx):
        elt, _ = self._item_helper(idx, False)
        return elt.value

    def __setitem__(self, idx, value):
        elt, src_origin = self._item_helper(idx, True)
        elt.value_origin = src_origin
        elt.value = value

    def _item_helper(self, idx, is_mutate: bool) -> Tuple[ArrayElement, SrcOrigin]:
        try:
            idx_dims = len(idx)
        except TypeError:
            idx_dims = 1
            idx = (idx, )

        assert self.array.dims == idx_dims, f"idx={idx} has wrong dimensionality; expect {self.array.dims}"

        elt = self.array.entries
        for i in idx:
            elt = elt.entries[i]

        src_origin = SrcOrigin()  # XXX

        # Check prior visibility, add visibility record
        if self.qtl_init is not None:
            qtl_convergent = self.array.qtl_convergent
            cta_rank_slice, threadIdx_slice = self.cta_rank_slice, self.threadIdx_slice
            c_range = range(cta_rank_slice.start, cta_rank_slice.stop)
            t_range = range(threadIdx_slice.start, threadIdx_slice.stop)
            assert isinstance(cta_rank_slice, slice), "Missing `with blk_thr(slice, slice)`"
            assert isinstance(threadIdx_slice, slice), "Missing `with blk_thr(slice, slice)`"
            assert c_range.start < c_range.stop
            assert t_range.start < t_range.stop

            # Check
            check_me = [v for v in elt.vis_list if is_mutate or v.is_mutate]
            def check_ob_in(v: WrappedEnvVis, ob_in: Set[Ob]):
                ob_witness = v.vis.ob_witness
                if not any(ob_witness(ob) for ob in ob_in):
                    # Nowhere near the quality of real camspork; good enough for now.
                    raise ValueError(f"Sync error at {src_origin}, qtl_convergent={qtl_convergent}, {v}")
            if qtl_convergent:
                ob_in = frozenset(Ob(c, t, q) for c in c_range for t in t_range for q in Qtl_pre)
                for v in check_me:
                    check_ob_in(v, ob_in)
            else:
                for c in c_range:
                    for t in t_range:
                        ob_in = frozenset(Ob(c, t, q) for q in self.Qtl_pre)
                        for v in check_me:
                            check_ob_in(v, ob_in)

            # Add
            arrive_on = self.arrive_on
            if arrive_on is None:
                aw_out = frozenset()
            elif isinstance(arrive_on, str):
                assert arrive_on == "itself", "Only recognized magic value"
                aw_out = frozenset((AwRay(elt.x, elt.idx, elt.value), ))
            else:
                aw_out = frozenset((AwRay(arrive_on.x, arrive_on.idx, arrive_on.value), ))
            if qtl_convergent:
                ob_out = frozenset(Ob(c, t, qtl_pre) for c in c_range for t in t_range)
                elt.vis_list.append(WrappedEnvVis(is_mutate, src_origin, EnvVis(ob_out, aw_out, self.Qtl_mask)))
            else:
                # Deviation: qtl_convergent = False case (non-registers) omits all but the lowest and highest thread
                ct_list = ((c_range.start, t_range.start), (c_range.stop - 1, t_range.stop - 1))
                if ct_list[0] == ct_list[1]:
                    ct_list = ct_list[:1]
                for (c, t) in ct_list:
                    ob_out = frozenset((Ob(c, t, self.qtl_init),))
                    elt.vis_list.append(WrappedEnvVis(is_mutate, src_origin, EnvVis(ob_out, aw_out, self.Qtl_mask)))

        assert elt.x == self.array.x
        assert elt.idx == idx

        return elt, src_origin


@dataclass(slots=True)
class BlkThrContext:
    context: "ProgramContext"
    cta_rank_slice: Optional[slice]
    threadIdx_slice: Optional[slice]

    def __enter__(self, *a):
        self.cta_rank_slice, self.context.cta_rank_slice = self.context.cta_rank_slice, self.cta_rank_slice
        self.threadIdx_slice, self.context.threadIdx_slice = self.context.threadIdx_slice, self.threadIdx_slice

    def __exit__(self, *a):
        self.cta_rank_slice, self.context.cta_rank_slice = self.context.cta_rank_slice, self.cta_rank_slice
        self.threadIdx_slice, self.context.threadIdx_slice = self.context.threadIdx_slice, self.threadIdx_slice


@dataclass(slots=True)
class BlkThrContextFactory:
    context: "ProgramContex"

    def __getitem__(self, tup):
        assert len(tup) == 2
        assert isinstance(tup[0], slice)
        assert isinstance(tup[1], slice)
        return BlkThrContext(self.context, *tup)


@dataclass(slots=True)
class ProgramContext:
    K: int
    A: A_gmem = A_gmem()
    B: B_gmem = B_gmem()
    C: C_gmem = None  # C_gmem(K)
    task_index: int = 0
    arrays: Dict[str, Array] = field(default_factory=dict)

    # When inside a for i in loop(name, lo, hi) loop
    # iter_env[name] maps to i; cleaned up upon loop exit.
    iter_env: Dict[str, int] = field(default_factory=dict)

    # Updated and restored by `with blk_thr`
    # We support lo:hi on CTA and threadIdx dimension.
    # Real Exo-GPU supports more than contiguous slices,
    # but this is good enough for the demo.
    cta_rank_slice: Optional[slice] = None
    threadIdx_slice: Optional[slice] = None

    def __post_init__(self):
        self.C = C_gmem(self.K)

    def next_task(self):
        mn = (M_task(self.task_index), N_task(self.task_index))
        self.task_index += 1
        return mn

    def loop(self, name: str, lo: int, hi: int):
        assert isinstance(name, str)
        assert isinstance(lo, int)
        assert isinstance(hi, int)
        iter_env = self.iter_env
        assert name not in iter_env

        for i in range(lo, hi):
            iter_env[name] = i
            yield i

        del iter_env[name]
        return

    def mk_dict(self):
        return dict(
            A=self.A,
            B=self.B,
            C=self.C,
            K=self.K,
            next_task=self.next_task,
            blk_thr=BlkThrContextFactory(self),
            alloc_rmem=self.alloc_rmem,
            alloc_smem=self.alloc_smem,
            alloc_mbarriers=self.alloc_mbarriers,
            alloc_async_group=self.alloc_async_group,
            loop=self.loop,
            Fence=self.Fence,
            Arrive=self.Arrive,
            Await=self.Await,
        )

    def _make_ob(sync_tl: Sync_tl):
        cta_rank_slice, threadIdx_slice = self.cta_rank_slice, self.threadIdx_slice
        c_range = range(cta_rank_slice.start, cta_rank_slice.stop)
        t_range = range(threadIdx_slice.start, threadIdx_slice.stop)
        assert isinstance(cta_rank_slice, slice), "Missing `with blk_thr(slice, slice)`"
        assert isinstance(threadIdx_slice, slice), "Missing `with blk_thr(slice, slice)`"
        assert c_range.start < c_range.stop
        assert t_range.start < t_range.stop

        return frozenset(Ob(c, t, q) for c in c_range for t in t_range for q in sync_tl._timeline_set)

    def _map_vis(self, vis_lambda):
        for a in self.arrays.values():
            a.map_vis(vis_lambda)

    def Fence(self, Qtl_arrive: Sync_tl, Qtl_await: Sync_tl):
        ob_in = self._make_ob(Qtl_arrive)
        ob_out = self._make_ob(Qtl_await)
        self._map_vis(lambda env_vis: env_vis.augmented_ob(ob_out) if env_vis.ob_witness(ob_in) else env_vis)

    def Arrive(self, Qtl_arrive: Sync_tl, bar: ArrayElement):
        # Only supports single bar
        ob_in = self._make_ob(Qtl_arrive)
        aw_out = frozenset((AwRay(bar.x, bar.idx, bar.value),))
        self._map_vis(lambda env_vis: env_vis.augmented_aw(aw_out) if env_vis.ob_witness(ob_in) else env_vis)

    def Await(self, bar: ArrayElement, Qtl_await: Sync_tl, z_lag: int):
        aw_in = Aw(bar.x, bar.idx, bar.value - z_lag - 1)
        ob_out = self._make_ob(Qtl_await)
        self._map_vis(lambda env_vis: env_vis.augmented_ob(ob_out) if env_vis.aw_witness(aw_in) else env_vis)

    def alloc_rmem(self, name: str, extents: List[int]):
        return self._alloc_common(name, extents, Qtl_mask_rmem, True, None)

    def alloc_smem(self, name: str, extents: List[int]):
        return self._alloc_common(name, extents, Qtl_mask_data_ram, False, None)

    def alloc_async_group(self, name: str, extents: List[int]):
        return self._alloc_common(name, extents, None, False, None)

    def alloc_mbarriers(self, name: str, extents: List[int], *, expected_arrive_count):
        assert isinstance(expected_arrive_count, int)
        assert extents[-1] == infinity
        return self._alloc_common(name, extents[:-1] + [0], Qtl_mask_mbarrier, False, expected_arrive_count)

    def _alloc_common(
            self,
            name: str,
            extents: List[int],
            Qtl_mask: Optional[Set[Qtl]],
            qtl_convergent: bool,
            expected_arrive_count: Optional[int],
    ):
        assert isinstance(name, str)
        assert all(isinstance(c, int) for c in extents)
        assert Qtl_mask is None or all(isinstance(q, Qtl) for q in Qtl_mask)
        assert isinstance(qtl_convergent, bool)
        assert expected_arrive_count is None or isinstance(expected_arrive_count, int)

        x = X(name)
        entries = self._make_array_entries(x, [], extents)
        return Array(self, x, len(extents), entries, Qtl_mask, qtl_convergent, expected_arrive_count)

    def _make_array_entries(self, x: X, idx, extents):
        if len(extents) == 1:
            entries = [ArrayElement(x, idx, None, None, []) for i in range(extents[0])]
        else:
            assert len(extents) >= 2
            entries = [self._make_array_entries(x, idx + [i], extents[1:]) for i in range(extents[0])]
        return ArrayLevel(entries)


def Sm90a_ping_pong(
    A, B, C, K,
    next_task, blk_thr,
    alloc_rmem, alloc_smem, alloc_mbarriers, alloc_async_group,
    loop, Fence, Arrive, Await,
  ):
  RING = 3
  As = alloc_smem("As", [RING])  # [i % RING]
  Bs = alloc_smem("Bs", [RING])  # [i % RING]
  Cs = alloc_smem("Cs", [1])  # [0]
  Dr = alloc_rmem("Dr", [2])  # [workgroup]

  # Mbarriers indexed by [ring buffer idx, phase]
  main_bar = alloc_mbarriers("main_bar", [2, infinity], expected_arrive_count=128)
  ep_bar = alloc_mbarriers("ep_bar", [2, infinity], expected_arrive_count=128)
  prod_bar = alloc_mbarriers("prod_bar", [RING, infinity], expected_arrive_count=32)
  cons_bar = alloc_mbarriers("prod_bar", [RING, infinity], expected_arrive_count=128)

  for task in loop("task", 0, 4):  # Unknown number of tasks per cluster; 4 for example
    ping = task % 2
    m, n = next_task()

    # Main loop
    for k in loop("k", 0, K):
      k_tot = k + K * task
      with blk_thr[0:1, 0:32]:
        ring_idx = k_tot % RING
        phase = k_tot // RING
        prod_bar.prepare_mbarrier_phase(ring_idx, phase)
        cons_bar.prepare_mbarrier_phase(ring_idx, phase)
        bar_info = dict(
            Qtl_pre={cuda_mbarrier_qual},
            qtl_init=cuda_mbarrier_qual,
            Qtl_mask=Qtl_mask_mbarrier,
            arrive_on="itself",
        )
        tma_info = dict(
            Qtl_pre={cuda_async_proxy_retired_qual},
            qtl_init=tma_to_smem_async_qual,
            Qtl_mask=Qtl_mask_data_ram,
            arrive_on=prod_bar(**bar_info)[ring_idx, phase],
        )
        As(**tma_info)[ring_idx] = A[m, k]
        Bs(**tma_info)[ring_idx] = B[k, n]
      with blk_thr[0:1, 128*ping + 128 : 128*ping + 256]:
        pass
    # End for-k; Epilogue begins
    with blk_thr[0:1, 128*ping + 128 : 128*ping + 256]:
      pass

Sm90a_ping_pong(**ProgramContext(5).mk_dict())
