# Piper-TP: status

Branch `feat/tp-sharding` on `zzw4257/piper`, 19 commits off `uw-syfi/main@439e960b`.
Addresses upstream issue [#15 "CP/TP support"](https://github.com/uw-syfi/piper/issues/15).
Detail and evidence for every claim here is in `notes/log.md` as `F1`–`F19`; this
page is the synthesis, not a duplicate.

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
| TP composes with PP, ZeRO-3, and split backward | TP x PP verified on 4 GPUs; ZeRO-3 on a separate region lowers cleanly; zero-bubble's BWD_I/BWD_W split anchors the backward collective correctly (F15, F23, F25) |
| **EP is unchanged by my refactor** | the shipped Qwen MoE example lowers byte-identically against a worktree of `upstream/main` (F28) |

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
   0%-util cards while the collectives were squeezed anyway.
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
4. `shard_tensor` itself, against issue #15.

Open design questions for #15: should `shard` grow a `collective` field instead of
a sibling op? Should `shard_tensor` refuse to compose with `replicate` (current
behaviour) or silently drop the DP sync as `shard` does? Should the fwd→bwd bridge
be per-stage, removing the hidden `order` dependency?

## Reproducing

```bash
python -m pytest -m "not gpu" test                      # 39 tests, no GPU
python experiments/dump_dag.py --schedule examples/base-schedules/tp2.json
torchrun --nproc_per_node=2 test/test_tp_equivalence.py # TP math, no Piper
python experiments/check_tp_equivalence.py              # TP=2 vs TP=1 in Piper
python experiments/check_tp_overlap.py                  # mb=1 cannot overlap, mb=4 does
```

Environment is on catalyst-fleet1; `notes/log.md` F5 records the layout and the
host's traps.
