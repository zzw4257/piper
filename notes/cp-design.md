# Stage G design: what a collective is allowed to depend on

Status (2026-09-13): G-0 done (log F37 falsified F-c on the way; see §2 F-c′),
G-1a/b done — boundary outputs carry a forwarded flag, `ring_exchange` lowers as
the edge-spliced baseline, 67 CPU tests green. G-2 queued on the GPU gate
(`experiments/run_cp_gate.sh`). F38 found while composing with `replicate`.
Marks follow the project rule —
**[code]** verified by reading `src/` at `0b65fb3`, **[intent]** from upstream
issue #15 / README, **[proposed]** mine.

---

## 1. The question

The TP work (Stages A–F) concluded that Piper's parallelism is entirely "which
collective is inserted where", and that TP fits that abstraction perfectly —
too perfectly to test it. The open question left behind was whether context
parallelism (CP) fits.

Structurally it does: write each ring step as an annotated region and the K/V
exchange lands on a region boundary, where the Stage B/C machinery already
works. But ring attention exists to overlap the exchange of block `i+1` with
the attention compute on block `i`, and a boundary collective in Piper sits
*between* two regions. The earlier conclusion was therefore: **CP would run
correctly and lose the reason to use it.**

That conclusion named the symptom ("a region is the minimum schedulable unit")
rather than the cause. This design locates the cause, and it is narrower and
more actionable than the symptom suggested.

## 2. What the code actually does [code]

Three facts, each independently verified:

**F-a. Every collective is spliced onto a dataflow edge.**
`_insert_shard_a2a_comm_nodes` and `_insert_tp_all_reduce_comm_nodes` both
rewrite `u -> v` into `u -> c -> v`
(`directives.py:1143-1144`, `directives.py:1168-1169`). The comm node's
dependency is therefore *the producing region's completion*, by construction.

**F-b. Piper already materializes pass-through values explicitly.**
A value defined in segment `s` and used in segment `s+3` is threaded through
`s+1` and `s+2` as an input and re-emitted as an output
(`fx.py:591-598`: `for target_seg in range(seg + 1, max_seg + 1)`). In the
intermediate segment's GraphModule that output element **is literally a
placeholder** — the segment forwards it without touching it.

**F-c (retracted 2026-09-13, see log F37).** This fact originally claimed the
ordering pass pins a dependency-free collective next to its consumer. Lowering
a ZeRO-3 DAG showed the opposite: `ALL_GATHER_COMM` is created as a DAG root
(`directives.py:867-895`), lives on `default_stream` so the stream-anchor logic
in `resolve_total_order_per_stream` never applies to it, and is issued at
**topological level 0** by `_serial_topological_order` — every layer's
parameters are gathered before the first forward runs. The corrected fact:

**F-c′. A collective's issue point is a function of graph structure only.**
Spliced collectives issue when their source region completes; root collectives
issue first. There is no policy in between, and no knob.

## 3. The gap, stated precisely [proposed]

Piper's passes conflate two different things:

- the **edge a collective travels on** (which tensor moves, between whom), and
- the **dependency a collective has** (when it may start).

For all three shipped collectives these coincide, because all three are
*producer–consumer* collectives: the gradient all-reduce must wait for the
gradient, the EP all-to-all must wait for routing, the TP all-reduce must wait
for the partial sum. The conflation is invisible because it has never been
wrong.

Ring attention's K/V rotation is the first collective where they differ. Its
payload is ready when the producing region *begins*, not when it ends. F-b says
this is not a subtle property — it is visible in the segment's output tuple as a
bare placeholder.

So the limitation is not "a region is the minimum schedulable unit". It is:

> **Piper has no notion of where a collective is issued.** Position falls out
> of graph structure: an edge-spliced collective is issued at source-region
> completion (too late for CP — its payload was ready at region start); a root
> collective is issued at topological level 0 (too early for ZeRO-3 — its
> memory budget wanted one layer of lookahead, not all of them). These are the
> same missing concept seen from opposite ends: an earliest-ready point, and an
> issue budget between that point and the consumer. Today the only available
> budgets are 0 and ∞.

F-a is what must be lifted for CP. F-c′ is what makes the `distance` argument
in §4 G2 a *missing* knob rather than an optimization.

## 4. Proposal: derive the earliest-ready point [proposed]

Do **not** add placement/partition types to the IR. That would be fitting the
abstraction to TP, the one case that already worked. Add instead a single
analysis, and let every existing directive benefit from it.

**G1 — earliest-ready analysis.** For a boundary tensor `t` produced by segment
`s`, take the backward slice of `t` inside `s`'s GraphModule.
- slice touches only placeholders → `t` is ready at the start of `s`; the
  collective's true dependency is the set of DAG nodes defining those
  placeholders.
- slice is empty (sharded parameters, ZeRO-3) → no dependency; ready at step
  start.
- slice touches compute → unchanged, current behaviour is already tight.

By F-b the common case degenerates to "is output element `i` a placeholder",
which needs no slicer. Build that first; add the slicer only when a model needs
it (an explicit rotate op inside the region is the case that needs it).

**G2 — hoist with a budget.** Hoisting `k` regions keeps `k` extra buffers live.
Express it as a directive argument, not a heuristic:
`{"op": "prefetch", "filter": ..., "distance": 1}`. CP is `distance=1`;
ZeRO-3 parameter prefetch is the same directive with `distance>=1` and an
empty slice. One mechanism, two features.

**G3 — an issue point, not an anchor (revised per F37).** The hoist needs an
explicit *latest-issue* edge as well as the earliest-ready dependency: a
temporal edge from the compute node `distance` regions before the consumer.
Without it the hoisted node becomes a root and F-c′ issues it at slot 0 — the
ZeRO-3 failure reproduced on purpose. The same edge, applied to today's
`ALL_GATHER_COMM`, is the ZeRO-3 prefetch budget.

**G3′ — what G-1b taught about the hoist (2026-09-13).** Two refinements that
only became visible with the spliced baseline lowered:

*Only the forward ring is hoistable.* Forward, the payload of `ring_i` is the
K/V chunk `CP_i` received and forwards untouched, so `ring_i`'s true dependency
is `ring_{i-1}` (or the K/V definer for `i=0`), not `CP_i`. Backward, the
payload is the gradient of that chunk *after* `CP_{i+1}'` has added its own
attention contribution — a produced value. The backward ring must wait for the
compute; the forwarded flag from G-1a says so mechanically (`forwarded=False`
on the gradient edge is not recorded, but the payload is `inp_grads`, which
compute writes). In a region-granular IR, ring attention can overlap half its
communication. Whether that half is the half that matters is a G-3 measurement.

*The issue budget is not optional for CP either.* Hoisted, `ring_i` depends
only on `ring_{i-1}`; with nothing else, all `n-1` rotations run to completion
before `CP_0` finishes and every K/V chunk is resident at once — F37's pattern
reproduced on the feature whose purpose is to hold `1/n` of K/V. The budget
edge `CP_{i-1} -> ring_i` (temporal, `distance=1`) makes `ring_i` overlap
exactly `CP_i`. P1 (revised) is the test of this paragraph.

*The consumer must merge, not replace.* Today a compute node with a boundary
comm predecessor takes that comm node's buffer as its whole input set. Hoisted,
`CP_{i+1}` has two data predecessors — `CP_i` (accumulators) and `ring_i` (K/V)
— and must take the accumulator slots from one and the ring slots from the
other. This is the one executor change G-3 needs beyond the pass.

**G4 — mechanical prerequisites.** A boundary records one `tensor_idx`
(`fx.py:686`, `_select_boundary_tensor_idx` at `fx.py:495` is a float/
requires_grad scoring heuristic); CP moves K and V, so this becomes a list.
`send`/`recv` are `peer_pp_rank`-typed and use the PP groups
(`executors.py:42-71`); a ring needs intra-group P2P over the existing
`ep_group` ranks.

## 5. Why this is the right generalization

The test of an abstraction is whether it explains something it was not derived
from. G1/G2/G3 were derived from CP, and they immediately explain a missing
defect in a *shipped* Piper feature — though not the one first predicted.
ZeRO-3 does not under-prefetch; it gathers *every* layer before the first
forward (log F37), which by the dispatch order and the free point makes peak
parameter memory the unsharded total. F-c′ says why, and G3's issue budget is
the fix. A framing derived from CP that lands on ZeRO-3's actual failure —
after being wrong about its sign — is better evidence of being at the right
level than one that had merely confirmed a guess.

## 6. Ladder

Each rung is cheap-before-expensive and has a falsifiable exit.

**G-0 (CPU, no GPU).** Can Dynamo trace a ring-attention-shaped model into `n`
contiguous annotated segments, and does the K/V boundary output appear as a
placeholder in intermediate segments? This is the whole design's crux and costs
nothing. *Exit:* dump the segments and assert the placeholder property, or
record that it fails and stop.

**G-1 (CPU) — done.** Boundary `outputs[]` with a forwarded flag (G-1a,
`2e480d3`) and `ring_exchange` spliced on the edge, no hoisting (G-1b,
`2995530`). Lowered order is `CP_i -> ring -> CP_{i+1}` both ways. The pass
acts on edges *between* matched regions, the exact set `shard_tensor` skips.

**G-2 (2 GPU) — queued.** Ring P2P executor over `ep_group` + numerics. Gate first,
outside Piper: `torchrun` ring attention with online-softmax/LSE accumulation
against a single-GPU reference, tolerance from a measured noise floor, with a
negative control that drops the rotation. Only then inside Piper.
*Exit:* correct, and **measurably non-overlapped** — this is the baseline the
rest is measured against.

**G-3 (2 GPU).** G1 analysis + G2 hoist + G3 anchor release.
*Exit:* the ring comm node's DAG predecessors no longer include the producing
region, and the concurrency metric (kernel-sum / wall-span) rises against G-2
on the same inputs, interleaved A/B.

**G-4.** Apply the same directive to ZeRO-3 parameter prefetch on an existing
shipped example. *Exit:* it works with no new mechanism, or it does not and the
reason is recorded.

**G-5 (backward).** Piper builds BWD by reversing FWD data edges. Whether that
produces a correct reverse ring for K/V gradients is genuinely unknown and is
the deepest risk; it is deliberately last so that a failure here does not
invalidate G-0..G-4.

## 7. Pre-registered predictions

Written before running anything, so a later result cannot be retrofitted.

**P1 (revised per F37).** Lifting F-a alone makes the ring-exchange node a
root, and F-c′ will issue it at slot 0 — every step's K/V exchange fired before
step 0's compute. Correct, and the memory-worst schedule. If G-3 without the
latest-issue edge shows *bounded* buffer growth, F-c′ is wrong.

**P2.** The Stage E/F finding — every collective re-pays a rank arrival skew of
157–2831 us, and cost is near-independent of payload — predicts that hoisting
may raise the concurrency metric without improving step time, because the wait
moves rather than disappears. A concurrency gain with no wall-clock gain is
therefore a *confirmation*, not a disappointment, and must be reported as such.

**P3.** A ring P2P couples rank `r` only with `r±1`, so arrival skew should
*propagate* rather than be globally summed as in an all-reduce. Prediction: for
equal payload, ring P2P shows materially lower iteration-to-iteration variance
than all-reduce. This is testable independently of whether CP itself succeeds,
and it isolates whether the "skew tax" is a property of global synchronization
or of NCCL. **Run this first — it is a standalone result.**

**P4 — resolved (log F37).** Tested on CPU before any GPU work. The anchor
rule does not apply to the collective in question and the sign of the effect
was the reverse of the prediction. The Stage F `stream` finding stands as
originally stated; the competing explanation is withdrawn.

## 8. Risks, cheapest check first

1. Dynamo segmentation of the ring loop (G-0, free). Highest probability of
   early failure, same as in Stage A.
2. The placeholder property may be destroyed by graphargs lifting before it
   reaches the DAG (G-0, free).
3. Online-softmax numerics — a CP bug and a Piper bug look identical in the
   loss; the out-of-framework gate exists to separate them (G-2).
4. Overlap measurement on a contended host is exactly the measurement that was
   least reliable in Stage E/F. Reuse the median/min contamination check and
   report concurrency, not wall time alone.
5. Backward ring correctness (G-5, expensive, deferred by design).

## 9. What would make me drop this

If G-0 shows the K/V boundary does not survive as a placeholder, the cheap form
of G1 is gone and the design needs a real slicer — at which point the honest
move is to write down that Piper's segmenter erases the information the
analysis needs, and stop, rather than build the slicer to save the thesis.
