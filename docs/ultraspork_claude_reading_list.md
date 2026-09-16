# Reading List and Working Advice

Distilled from our session. Scoped to your real constraints: solo, systems-flavored,
LLM-target user, hardware (Hopper async: TMA/mbarrier/wgmma/tcgen05) with no formal
spec, PhD-timeline flexible, no collaborator planned.

The organizing principle: cut the proof-assistant path cleanly, invest in operational
literacy so you can *borrow design patterns* from formal methods without paying the
mechanization tax, and pair that with concrete tool fluency for the parts of the
project that are actually implementable.

---

## Tier 0 — the single highest-leverage text

Bradley & Manna, *The Calculus of Computation: Decision Procedures with Applications
to Verification* (Springer, 2007). Covers first-order logic, the theories you will
actually query (linear integer arithmetic, uninterpreted functions, arrays), Nelson–
Oppen combination, and the decidability boundaries — precisely the mental model that
turns SMT from a black box into a tool you use with judgment. This is the one book to
read cover-to-cover. Everything else in the list is targeted rather than
comprehensive.

---

## Tier 1 — operational literacy for design vocabulary

Purpose: internalize enough separation logic, session types, and operational-semantics
practice to *design* your contract system in the shape a formal-methods person would,
without ever mechanizing it. Read for concepts and design patterns, skip the exercises
and mechanization. Rough budget: 3 months in parallel with implementation.

### Separation logic (concept level, no proof assistant)

- Peter O'Hearn, "Resources, Concurrency, and Local Reasoning" (2007) — the original
  Concurrent Separation Logic paper. The core idea to extract: **synchronization
  transfers ownership of resources between threads.** This is the pattern your
  contract algebra is already reinventing.
- Birkedal & Bizjak, *Lecture Notes on Iris: Higher-Order Concurrent Separation Logic*
  — read the first several chapters for concepts (resources, invariants, ghost state,
  ownership transfer at synchronization). Skip the Coq exercises. The goal is
  vocabulary and intuition for *what shape of thing* Iris ghost state is, so you can
  recognize when your contracts want it.
- Optional depth: Vafeiadis, "Concurrent Separation Logic and Operational Semantics"
  — good bridge from the logic to how you'd state semantics operationally.

### Session types and protocol-in-types

- Honda, Vasconcelos, Kubo, "Language Primitives and Type Discipline for Structured
  Communication-Based Programming" (1998) or the more accessible Wadler, "Propositions
  as Sessions" (2012). The pattern to extract: **communication protocols encoded as
  types the compiler checks.** This is the design lens for your ordered-phase-token
  and producer/consumer contracts — they are session-type-like protocols over GPU
  barriers.
- Skim one modern application, e.g. any of the Rust session-types crates' design
  writeups, to see the pattern in a real language.

### Operational semantics of real languages (pattern, not mechanization)

- Rustbelt papers (Jung et al., "Rustbelt: Securing the Foundations of the Rust
  Programming Language," POPL 2018) — read for the *shape* of the artifact: how you
  state an operational semantics for a real language, define a type system, and
  connect them. Skip the Iris proofs. The takeaway is what such a paper looks like
  and what parts of it a systems-flavored person can write without mechanization.
- One of the WebAssembly formal semantics papers (Haas et al., PLDI 2017) for a
  simpler, more digestible instance of the same pattern.

### Static analysis foundations

- Møller & Schwartzbach, *Static Program Analysis* lecture notes (free PDF, ~150pp).
  Lattices, monotone frameworks, fixpoints, widening. This is the retroactive
  explanation of what your existing sync checker is doing and what you'd need to make
  it static across loops. Read at least the lattice/monotone/fixpoint core and the
  widening section.

---

## Tier 2 — tool fluency for what you'll actually build

### SMT / Z3

- The Z3 guide / tutorial (Rise4Fun tutorial, or de Moura & Bjørner's papers). Get
  fluent in SMT-LIB, modeling with LIA, EUF, arrays, and quantifiers. Learn
  precisely where decidability ends (nonlinear integer arithmetic, quantified
  fragments) so you know when Z3 becomes a heuristic that might time out.
- Bradley & Manna covers the theory; the Z3 material covers the practice. Pair them.

### Equality saturation / e-graphs

- Willsey et al., "egg: Fast and Extensible Equality Saturation" (POPL 2021) and the
  follow-up egglog papers. This is the right tool for your polynomial-spec
  normalization and coordinate-vs-control equality checking.
- Skim the egg tutorial and the Herbie / Ruler papers as examples of the pattern in
  real use.

### Polyhedral model (deferred, reference only)

- Feautrier & Lengauer, "The Polyhedral Model" (Encyclopedia of Parallel Computing
  entry) — read as an orientation, not a course. Know it exists, know its exact
  boundary (affine loop nests, Presburger). Reach for ISL only if you find yourself
  reimplementing dependence analysis or mod-elimination heuristics.

---

## Tier 3 — memory models and GPU-specific formalization

Purpose: know what has and hasn't been formalized so your soundness disclaimers are
calibrated and your design borrows what exists.

- Lustig et al., "A Formal Analysis of the NVIDIA PTX Memory Model" (ASPLOS 2019).
  This axiomatizes the generic-proxy scoped fragment. It is *not* a model of async
  proxies / TMA / mbarrier / wgmma / tcgen05, and you should be explicit about that
  in your own writing.
- Batty et al. work on C/C++11 release/acquire — the axiomatic style your visibility
  records approximate. "Mathematizing C++ Concurrency" is a good entry point.
- Alglave et al., "Herding Cats" — the `herd` / cat-language framework for
  axiomatic memory models and litmus testing. This is the methodology reference for
  *empirically validating* memory models against hardware, which is your situation.

---

## Tier 4 — GPU-specific tools and prior art to know

- GPUVerify (Betts, Chong, Donaldson et al.) — static race-checking for CUDA via a
  two-thread reduction. Know what it does and, more importantly, *why* its two-thread
  abstraction predates and does not extend to Hopper async primitives. This is the
  prior work your contribution is delta'd against; you need to be able to explain the
  delta cleanly.
- CUTLASS documentation, particularly the ping-pong / cooperative kernel design docs
  and `OrderedSequenceBarrier` / `MathWarpGroupOrderBarrier` in the source. Not
  formal, but the closest thing to a written-down protocol algebra for real Hopper
  kernels.
- Exo and Exo-GPU papers (you know these). Worth re-reading through the contract-
  system lens now that you're framing it as a type discipline rather than a checker.
- The AsyncSparse paper you sent. Keep it as the stress-test target for whatever you
  design; if your contract system can't cleanly express its warp-specialized data-
  dependent ring buffer, the design isn't done.

---

## Explicitly out of scope — do not spend time here

- Software Foundations Vol 1+ as a full course. Given your constraints, the ROI is
  wrong. Skim only if you find a specific concept in the operational-literacy tier
  that requires it.
- Coq / Rocq / Iris as proof assistants (writing proofs, tactic engineering, ghost-
  state constructions). Cut cleanly. If a future collaborator or future-you with
  different constraints wants to mechanize soundness later, the reading you did in
  Tier 1 is the preparation you need at that point; the tool skills are acquired then.
- VST / CompCert / Verifiable C. Wrong target language, wrong scale.
- Deep polyhedral compiler internals (isl API mastery, Pluto, etc.). Reach for pieces
  only when you have a concrete affine-dataflow need.
- Full weak-memory formal-model construction from scratch. This is a research
  program you don't have collaborator support for. Take the parts of the Lustig
  model that apply, be explicit about the async gap, and validate empirically.

---

## Working advice accumulated over the session

### Frame the project honestly

The pitch is: **a low-level GPU language whose type discipline structurally prevents
the race patterns we enumerate, whose value-correctness properties are checkable
against symbolic specs via SMT, validated empirically against ptxas and hardware,
with case studies showing it handles irregular parallelism (AsyncSparse-class,
ping-pong with data-dependent trip counts) that existing languages express only
unsafely.** This is a real PL contribution. It is not "we mechanically proved GPUs
correct," and you should not pretend otherwise; but it is also not a lesser thing —
it is a systems/PL contribution aimed at the user who actually exists (LLM +
engineer, not proof-writer).

### The two-oracle validation discipline

You have two oracles for two properties, and they should be kept separate in the
harness and the write-up:

- **Ptxas-acceptability** (your wgmma-serialization property, the "mystery
  slowdown" category from your rant): oracle is ptxas itself, read from SASS via
  `cuobjdump --dump-sass`, grepping for `WARPGROUP.DEPBAR` / serialization
  instructions and the warning strings. This is a compile-time oracle; the H100 is
  not required for this loop, only the toolchain. You are reverse-engineering
  ptxas's undocumented static analysis; your checker is the open reimplementation.
- **Value / race correctness**: oracle is runtime, `compute-sanitizer --tool
  racecheck` plus result-diffing against a reference. Requires H100. Catches the
  cases where synchronization is under-specified in a way that corrupts data,
  independent of what ptxas thinks.

A program can be value-correct but ptxas-serialized (racecheck passes, SASS shows
serialization) or ptxas-accepted but value-wrong (rare, but possible). Your checker
should predict both verdicts; disagreement with either is a model bug. This dual-
oracle setup is a clean evaluation section.

### The empirical-escape-hatch heuristic

Ask this every time you reach for empirical validation instead of static reasoning:

- Is the property invariant across all inputs of a given syntactic shape (e.g. any
  well-formed producer/consumer over a managed ring buffer is race-free)? → this is
  **type-system territory**, not empirical. If you're testing it empirically you're
  leaving power on the table.
- Does the property depend on data that is genuinely runtime-only and unbounded (deep
  sparsity structure of an arbitrary matrix)? → **empirical / SMT-with-substitution
  territory**. Appropriate.
- Does the property have no formal spec against which to prove anything (ptxas
  conservatism)? → **empirical against the oracle that exists**. Appropriate.

If a property falls in the first bucket and you're testing it, that's the escape-
hatch misuse you asked about. If it falls in the second or third, you're calibrated.

### The design-decision writeup habit

For each significant design decision, write one paragraph with this structure:

1. The property or obligation you're trying to establish.
2. The approach you chose.
3. What a formal alternative would look like (in the vocabulary you're building from
   the Tier 1 reading).
4. Why you chose your approach given your constraints.

Two purposes: it builds the taste you asked about (you cannot compare to a formal
alternative you cannot describe), and it produces a corpus you can occasionally
share to get asynchronous feedback without a co-authorship commitment. A research
blog, a well-scoped question on the Iris Mattermost or a PL Discord, a workshop
paper. Low social cost, high signal when someone with taste bites.

### Structural design points to hold onto

- **Separate the resource protocol from the schedule that drives it.** The managed
  ring buffer conflated allocation/aliasing discipline with the release schedule.
  The protocol is static and identical for dense and sparse; the schedule is affine
  for dense, symbolic/data-dependent for sparse. Check pairing-completeness
  symbolically over the schedule; discharge the protocol obligation over the schedule
  once.
- **The contract menu needs at least three kinds:** (i) explicit acquire + explicit
  release, mbarrier-witnessed (your existing managed case); (ii) explicit acquire +
  implicit release riding on a required pre-existing edge (the sm_80 case, the
  release-witnessed-by-obligation form); (iii) ordered-phase-token / mutual-exclusion
  with rotation across sub-collectives (the CUTLASS `OrderedSequenceBarrier` case).
  These do not reduce to each other.
- **Do not model RMEM double-buffering with the same protocol as SMEM ring
  buffering.** Same syntactic shape, different contract kind. Selected by what kind
  of instruction consumes the buffer (synchronous → program-order; async wgmma →
  commit-group-witnessed).
- **Sync-correctness and value-correctness share a substrate.** Both bottom out in
  deciding equality of extended-polynomial index expressions modulo ring depth,
  under opaque functions for data-dependent trip counts. Build that decision
  procedure (Z3 + e-graph normalization + concrete-substitution fallback) once and
  reuse it for both.
- **The opaque data-dependent function drops out of the sync obligation** when the
  loop is monotone and at-most-one-trip-per-tile. The pairing modulo `RING` is
  independent of the sparsity map; sparsity re-enters only in value correctness.
  This is a strong result and worth stating precisely in whatever you write.
- **Design against needing a soundness proof.** Rust shipped its borrow checker
  years before Rustbelt mechanized its soundness. The type discipline's *design*
  does most of the work; the mechanized proof is a ratifying contribution, not a
  constitutive one. Aim for design + empirical validation and leave mechanized
  soundness as explicit future work.

### On the "circular reasoning" concern

The genuine version of it: if your project's *shape* is set by not knowing the
formal alternatives, no amount of implementation-tool advice will surface the design
you should have chosen. The Tier 1 reading is the answer to this — it is the
minimum you need to *see* the alternatives clearly enough to reject them for the
right reasons. Once you've done that reading, the design decisions you make are
informed rather than default, and the "am I missing a technique" anxiety has a
principled place to go: into the design-decision paragraphs, and occasionally into
lightweight external channels for sanity check.
