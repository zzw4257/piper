# Piper-TP: status

Branch `feat/tp-sharding` on `zzw4257/piper`, 98 commits off `uw-syfi/main@439e960b`.
Addresses upstream issue [#15 "CP/TP support"](https://github.com/uw-syfi/piper/issues/15):
TP is done and verified (Stages A–F); CP is built and CPU-verified, GPU rungs
queued (Stage G, at the end of this page). Detail and evidence for every claim
here is in `notes/log.md` as `F1`–`F39`; this page is the synthesis, not a
duplicate. Retractions (F16→F17, F31→F32→F33, F34, cp-design F-c→F37) are kept
in place.

## What works

Tensor parallelism as a schedule directive: `shard_tensor` inserts activation
all-reduces at a TP region's boundary, executed as `TP_COMM` nodes.

```json
{ "op": "shard_tensor", "filter": {"TP": "*"}, "devices": [0, 1], "stream": "tp_stream" }
```

Verified:

| claim | evidence |
|---|---|
| TP=2 matches the unsharded MLP | out-of-band under torchrun: output 1.5e-08, input grad 2.2e-09 (`test/test_tp_equivalence.py`) |
| TP=2 matches TP=1 **inside Piper**, across optimizer steps | 7.2e-07 worst, both TP ranks identical (`experiments/check_tp_equivalence.py`) |
| TP x PP=2 on 4 GPUs under a generated 1F1B order | ~1e-6 vs a 1-GPU 2-stage baseline (`--pp`) |
| the IR rewrite is right | 12 CPU tests; 39 pass total, CI-safe (`-m "not gpu"`) |
| dropping the collectives breaks all of the above | negative control in each check |
| `fuse_collectives` merges a group's collectives into one call | bit-identical losses; 12-19% faster where communication is ~75% of the step; `order` takes priority, so 1F1B blocks it (F34) |
| fusion vs 1F1B, the choice it forces | indistinguishable over nine interleaved repetitions, criterion declared first; fusion's useful home is single-stage TP, where `order` buys nothing anyway (F35) |
| TP composes with PP, ZeRO-3, and split backward | TP x PP verified on 4 GPUs; ZeRO-3 on a separate region lowers cleanly; zero-bubble's BWD_I/BWD_W split anchors the backward collective correctly (F15, F23, F25) |
| **EP is unchanged by my refactor** | the shipped Qwen MoE example lowers byte-identically against a worktree of `upstream/main` (F28) |
| both shipped examples still run | Qwen MoE (EP+ZeRO+PP x DP) and LLaMA (PP x DP) on 4 GPUs, the first run on real inputs since the zero-input fix (F30) |

`src/` diff is ~410 lines across 12 files. TP needed **no new node kind
mechanics, no new process group, and no launcher change**: it reuses the
device-group SPMD model, the boundary-comm node pattern, and the otherwise-idle
`ep_group`.

## Three findings that are not about TP (each PR-able alone)

1. **Piper trained on zero inputs**, the shipped LLaMA example included. On torch
   2.10 Dynamo attaches `meta["grapharg"]` to *every* placeholder during the
   backend call, so `_placeholder_is_runtime_input` rejected the real model input
   too and segment 0 got `input_idxs=[]`; `_load_stage` then zero-filled it. Found
   because the MSE came out at `mean(labels²)` to seven digits. One-line fix. (F9)
2. **The loss was computed for the backward and dropped in four places**, so
   `piper_exec_dag` always returned `[]`. This is *why* (1) survived: no observable
   depended on the model's arithmetic. (F8)
3. **`order` is mandatory when `pp_degree > 1`** and nothing says so. The
   fwd→bwd bridge is added only at the *globally* last forward, so a non-last
   stage's forward reaches its backward only through the next stage — which the
   P2P cut severs. The error said "expected distinct device sets"; it now
   diagnoses the cause. (F14)

## What TP exposes about the abstraction

- **The IR has no representation of a partitioned tensor.** Shapes are frozen at
  trace time as meta tensors, `device` is a group, and parallelism is entirely
  "which collective goes where". `shard` (EP) partitions nothing either. So
  TP-local shapes must live in the *model*, and `tp` cannot be varied without
  retracing — an autotuner has to drive model construction, not the scheduling
  language. (F1, F16)
- **No third parallel axis**: `world = pp_degree x dp_degree`. TP x PP composes
  (TP borrows dp, PP uses stages); TP x DP does not. (F4, F14)
- **Every consumer restates the sharding rule.** Parameter overrides must be
  per-rank values, because the runtime has nowhere to keep a partition spec. (F10)
- **`split` replicates the DAG, it does not partition the batch.** Easy to
  misread, and it makes two different questions about microbatches look like one.
  (F18)

## Unstated invariants (the pattern worth reporting)

Five findings turned out to be one shape: **the IR has invariants nothing
declares, and violating them fails silently.** Each is now an explicit check or
test, but the pattern is the interesting part for a system whose thesis is
user-programmable scheduling.

| invariant | how it failed | status |
|---|---|---|
| one boundary-comm directive per region | `shard`+`shard_tensor` on one region: whichever ran second found the edge already rewritten and inserted nothing. Order in the JSON decided the semantics; one order dropped TP entirely (F20) | rejected |
| a region boundary carries one interesting tensor | a TP region with a partial sum *and* a replicated side output gets exactly one of them all-reduced, chosen by a scoring heuristic (F23). Same assumption blocks CP, which must move K and V (F21) | rejected |
| every comm kind feeding COMPUTE is awaited by the consumer | `DagExecutor.run` matches predecessors on exact `task_type`; an unregistered kind is skipped and the consumer reads the wrong inputs. `TP_COMM` was in that state mid-project (F22) | tested |
| `pp>1` needs an `order` directive | the fwd→bwd bridge exists only at the globally last forward, so the P2P cut disconnects every other stage. The error named device sets, not the cause (F14) | diagnosed |
| `devices` is a symbolic group, not GPU ids | `devices=[41, 99]` compiles on a 7-GPU box; only the count and set-equality are ever read (F24) | documented |
| a sharding group needs at least two devices | `devices=[0]` was accepted and would call `dist.all_reduce` on an uninitialized group, failing deep in the executor. Same hole upstream for `shard` (F29) | rejected |

The common mechanism: each pass validates its own preconditions against the DAG,
and a pass that matches nothing does nothing rather than complaining. Composition
is unchecked, so the failure mode is wrong arithmetic with no diagnostic.

## Performance, and two retractions

- TP communication is **~4–6% of GPU kernel time** at dim 8192 / hidden 32768 /
  bf16 on 2xB200; one 32 MiB all-reduce ~93 µs, ~360 GB/s. (F11)
- **`stream` is a no-op at one microbatch** — concurrency 0.99–1.00x over ten
  iterations, a saturated timeline with exactly zero overlap, exactly as the
  lowered DAG predicts. At four microbatches it is 1.06x, 10/10, with no `order`
  written. (F12)
- **`order` bought nothing on a single stage**: 1F1B is 3.7% *slower*. No bubble
  to fill, and it only constrains an ordering that was already good. (F13)
- **With a bubble, `order` pays: 1F1B beats GPipe by 16%** under TP x PP=2 on four
  GPUs, 3/3 repetitions, ranges not overlapping — same schedule, same collectives,
  only the `order` directive differs. The two results are one statement:
  `order`'s value is proportional to the bubble it can fill, and TP alone creates
  none. This is the only measurement where TP and programmable scheduling produce
  value together. (F26)
- **TP=2 → TP=4 is 30% faster** (4.85 vs 6.92 ms), with compute matching the
  roofline to 1.44x against a predicted 1.50x. Collectives are **bandwidth-bound
  **almost independent of payload**: sweeping 2 → 64 MB (32x) moves the cost only
  138 → 322 us (2.3x), so it is fixed-cost dominated everywhere tested. Not
  dispatch (CUDA graphs bought 2%, F19) and not bandwidth (this sweep) — what is
  left is the two ranks' **arrival difference at each collective**, caused by the
  compute between them: four back-to-back all-reduces cost 57 us each against
  220-320 us for the same collective with compute in between. Fusion measures
  1.1x-2.4x out of context and is a lower bound in situ, but it forces all
  microbatches to arrive before any collective runs, so it is **incompatible with
  1F1B** and must beat it rather than beat nothing.
  (F31-F33; F32 retracts F31's reading, F33 corrects F32's mechanism)
- **At constant total work, more microbatches make TP 2.6x worse.** So TP wants
  few microbatches while PP wants many — opposing preferences on one knob, and the
  first genuine scheduling question here that the IR does not already answer. (F18)
- **Retracted:** F16's "one GPU beats both parallel configurations by 25–42%".
  With interleaving and a minimum, `pp=2` is 27% *faster* than one GPU. I took one
  sample per configuration, and the NVLink-using samples are the ones that drift
  2x — violating my own rule from F11 in the one place it mattered. (F17)
- **Retracted:** the per-collective cost model that replaced it. CUDA graphs
  capture a TP-shaped step fine and buy **2%**, so dispatch drift is not the cost.
  It is NCCL latency: 105 µs for a 4.2 MB two-rank all-reduce against an 11.7 µs
  bandwidth bound. **Piper's runtime overhead is real, measured, and not what
  makes TP slow.** The cost is `collective count x NCCL latency`. (F19)

## Measurement rules this project had to learn

1. Report the **minimum**, never the mean: communication varies up to 66x across
   iterations while compute stays within 1.1%.
2. **Owning your GPUs is not enough.** SMs are per-GPU and can be held
   exclusively; NVLink/NVSwitch is machine-wide. Compute was rock-steady on two
   0%-util cards while the collectives were squeezed anyway. Check a window
   before trusting it: **`comm median / comm min`** was 1.55 in a usable window
   and 4.46 in one where the median could not separate a 32-collective schedule
   from a 4-collective one (F36).
3. **The first collective of a step is not a measurement** — it absorbs rank
   arrival skew. rank 0 showed 2858–30485 µs where rank 1 showed ~94 µs for the
   identical payload. Use a later collective.
4. Interleave A/B/A/B. Comparing two runs, or two builds, measures the drift.

`experiments/profile_tp.py` encodes 1–3 and warns when they are being violated.

## Suggested upstream contributions, in order

1. The zero-input fix (F9) — affects every example, one line, has a regression
   test written the only way that catches it (asserting inside the backend call).
2. The loss plumbing (F8) — prerequisite for any correctness lane on any
   parallelism dimension.
3. The `order`-requirement diagnosis (F14) — message-only.
4. `shard_tensor` itself, against issue #15 (the TP half).
5. `replicate(prefetch_distance)` (F37/F39) — shipped ZeRO-3 gathers every layer
   before the first forward; one temporal edge per gather gives it the schedule
   hand-written ZeRO-3 uses. Off by default; both behaviours are pinned by tests.
   GPU memory slope measurement queued.
6. The parameter-free `REDUCE_COMM` guard (F38) — two lines, brings
   `_insert_reduce_comm_nodes` in line with the all-gather pass; today a dead
   node per parameter-free region per microbatch.
7. `ring_exchange` (the CP half of #15) with `hoist`/`distance` — after the
   queued G-2/G-3 runs, not before.

Open design questions for #15: should `shard` grow a `collective` field instead of
a sibling op? Should `shard_tensor` refuse to compose with `replicate` (current
behaviour) or silently drop the DP sync as `shard` does? Should the fwd→bwd bridge
be per-stage, removing the hidden `order` dependency?

## Reproducing

```bash
python -m pytest -m "not gpu" test                      # 74 tests, no GPU
python experiments/dump_dag.py --schedule examples/base-schedules/tp2.json
python experiments/dump_dag.py --model ring --schedule examples/base-schedules/cp2_ring_dp_hoist.json --steps 3
python experiments/dump_dag.py --schedule examples/base-schedules/zero3_s3_mb1_prefetch1.json --stages 3 --tp 1
PYTHONPATH=examples:. python experiments/probe_cp_segments.py --steps 4  # G-0: K/V survive as placeholders
torchrun --nproc_per_node=2 test/test_tp_equivalence.py # TP math, no Piper
torchrun --nproc_per_node=2 test/test_cp_equivalence.py # ring math, no Piper (queued)
python experiments/check_tp_equivalence.py              # TP=2 vs TP=1 in Piper
python experiments/check_cp_equivalence.py --cp 2       # CP=2 ring vs dense in Piper (queued)
python experiments/check_tp_overlap.py                  # mb=1 cannot overlap, mb=4 does
```

Environment is on catalyst-fleet1; `notes/log.md` F5 records the layout and the
host's traps.

## Stage G: context parallelism as the probe that found the floor

*Status 2026-09-13. CPU side complete; GPU rungs queued behind the card gate.
Design: `notes/cp-design.md`. Findings F37–F39 in `notes/log.md`.*

TP fit Piper too well to test it (§ "What TP exposes"). The open question it
left was whether CP fits, and the earlier answer — "CP runs correctly and loses
the reason to use it, because a region is the minimum schedulable unit" — named
a symptom. Stage G located the cause, and it is narrower and more actionable.

**The cause.** Piper's passes conflate the *edge a collective travels on* with
the *dependency a collective has*. For all three shipped collectives these
coincide, because all three are producer–consumer (gradient, routed tokens,
partial sum). Ring attention's K/V rotation is the first where they differ: its
payload is ready when the source region *starts*. The IR can express the
resulting diamond — it is a general DAG with per-node streams and events — the
directive language could not.

**What was wrong in the design, and how it was found.** The design predicted a
second mechanism: the ordering pass pins a dependency-free collective next to
its consumer. One CPU lowering of a ZeRO-3 DAG (F37) showed the opposite sign:
`ALL_GATHER_COMM` is a DAG root on `default_stream`, so the anchor rule never
applies, and `_serial_topological_order` issues *every* gather at topological
level 0 — all layers' parameters before the first forward. Peak parameter
memory under shipped ZeRO-3 is, by the dispatch order and the free point, the
unsharded total. The corrected statement: **Piper has no notion of where a
collective is issued; position falls out of graph structure — spliced
collectives at source completion (too late for CP), roots at level 0 (too early
for ZeRO-3).** One missing concept, seen from both ends. Retracted in place.

**What was built (all CPU-verified, 74 tests).**

| rung | what | evidence |
|---|---|---|
| G-0 | ring-shaped model traces to `n` contiguous CP regions; K/V survive as bare placeholders in every intermediate region (Piper already threads cross-segment values, `fx.py:591-598`) | `experiments/probe_cp_segments.py` |
| G-1a | every boundary output recorded with a `forwarded` flag — the earliest-ready fact, free at segmentation | `test_boundary_outputs.py` |
| G-1b | `ring_exchange` directive: acts on edges *between* matched regions (the set `shard_tensor` skips), refuses produced tensors, rotates +1 forward / −1 backward; `RING_COMM` kind, `batch_isend_irecv` on `ep_group`, two dispatch arms, contract test updated | `test_ring_directive.py` |
| G-3 | `hoist=true, distance=d`: forward ring depends on its supplier (previous ring / definer), source→successor edge restored for produced slots, consumer merges; backward stays spliced (its payload is *produced*); `record_stream` on merged tensors | 11 ring tests, `dump_dag --model ring` |
| G-4 | `replicate(prefetch_distance=d)`: one temporal edge per gather turns F37's `AG_0…AG_8 │ compute_0…` into `AG_0 compute_0 AG_1 compute_1 …` with no new mechanism (F39) | `test_zero3_prefetch_budget.py` |

**Three things the hoist taught (design § G3′).** Only the forward ring is
hoistable — backward's payload is a gradient the compute produces, so at most
half the ring can overlap in a region-granular IR. The issue budget is not
optional for CP either: without `distance`, every rotation completes before
step 0 finishes and all K/V chunks are resident — F37 on the feature meant to
hold `1/n`. And at `distance=1` the budget edge is also what makes buffer reuse
safe: the codebase has no `record_stream`; spliced topologies were safe only
because every comm node waited on its consumer's predecessor, which hoisting
removes.

**F38.** `replicate` emits a `REDUCE_COMM` for regions with no parameters
(`_insert_reduce_comm_nodes` never checks; the all-gather pass does). The
executor's `all_reduce_grads` guard returns 0 — a dead node per parameter-free
region per microbatch, not a wrong answer. Invisible until CP's attention-only
regions, the first parameter-free regions Piper has seen. Pinned as a
characterization test; fixed after the queued run has exercised upstream
behaviour.

**Queued on GPU, with predictions written first.**

- *F37 slope* — **done, prediction failed, mechanism found (F42, F43).** Plain
  DP 4.00 (predicted 4.0), shipped ZeRO-3 3.61 (3.0), `prefetch_distance=1`
  3.99 (2.0). The dispatch order changed as the CPU tests say; the peak did not
  follow, because it sits at the end of the backward where full gradients
  accumulate ahead of their deferred frees, and allocation happens on the host
  at dispatch. Blocking the host on the budget predecessor gives slope 2.01
  and 11.6 GiB at 8 stages against DP's 20.0 — the pre-registered test of the
  mechanism. An issue budget is a DAG edge *and* a runtime allocation policy;
  Piper has neither and the first without the second is nothing. **The policy
  is now built (F44): a bounded buffer pool reused on the stream-side free
  event, no host wait — with the edge, slope 2.01; alone, 3.01; both predicted.
  Replicated on H200 (F47), the three race-free arms reproduce to two decimals
  (4.00 / 2.01 / 3.01) and the two racing arms do not (3.61→3.06, 3.99→2.85) —
  so without the pool, ZeRO-3's peak memory is a property of the machine, not
  of the program, and the two schedules even swap places.**
- *G-2 numerics* — **done (F40, F41).** torchrun: out 4.2e-07, dQ 1.2e-06,
  dK 2.9e-06, dV 1.9e-06, both controls broke. In Piper: CP=2 mean-over-ranks
  equals dense CP=1 to 1.7e-06 across three optimizer steps; dropping the ring
  moves it by 3.2e-03. The hoisted run's per-rank losses are digit-identical to
  the spliced run's.
- *G-3* (`run_cp_hoist.sh`): hoisted numerics on 2 cards; on 4 cards spliced vs
  `distance=1` vs `distance=0` — P1 revised predicts `distance=0` peak memory
  grows with steps and `distance=1` does not. Two cards cannot show it: one
  rotation.
- *P2*: hoisting may raise the concurrency metric without moving step time,
  because the per-collective skew tax (Stage E/F) moves rather than disappears.
  A concurrency gain with no wall-clock gain is a confirmation.
- *P3* (`probe_skew_topology.py`, 4 cards): is the skew tax a property of global
  synchronization or of NCCL? Ring couples only neighbours; the rank opposite
  the skewed one should not pay on that step. Standalone result, independent of
  whether CP succeeds.

**What this says about "going up a level".** The abstraction to add is not a
placement/partition type — that fits TP, the case that already worked. It is
an *issue point* per collective: an earliest-ready dependency, derivable from
the segment (a bare placeholder in the output tuple), and a budget edge between
that point and the consumer. Derived from CP; it explained a missing
optimization in shipped ZeRO-3 after being wrong about its sign. That is
better evidence of being at the right level than a confirmed guess would have
been.
