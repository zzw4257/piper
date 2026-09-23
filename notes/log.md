# Piper-TP research log

Append-only. One entry per validation-ladder rung or per surprise. Entries are
dated and numbered `F<n>`. Retractions are marked in place, never deleted.

Headings per entry: **Tested / Changed / Unexpected / IR-runtime assumption
discovered / Open question / Next experiment.** Omit a heading only when it has
nothing in it.

Distinguish throughout:
- **[code]** verified by reading or running the code, with `file:line`;
- **[intent]** what the paper or issue #15 says should happen;
- **[proposed]** our own extension.

---

## Index

| | entry | |
|---|---|---|
| **F1** | Piper has no partitioned-tensor representation; `shard` is EP-only and partitions nothing | |
| **F2** | `REDUCE_COMM` is a parameter-gradient collective, not an activation collective | |
| **F3** | Per-rank random weight init makes numerical comparison impossible as shipped | |
| **F4** | The 2-GPU TP placement is expressible with zero launcher changes | |
| **F5** | Environment works on B200 (torch 2.10+cu128, sm_100); disk blocker was misdiagnosed | |
| **F6** | TP's conjugate pair is two *outgoing* edges; boundary-only insertion is semantics, not an optimization | |
| **F7** | TP=2 runs on two B200s; the DAG matches F6's prediction and D1 agrees to 1.5e-08 | |
| **F8** | The loss was computed for the backward and dropped, in four places | |
| **F9** | Piper trained on **zero inputs**, the shipped LLaMA example included | |
| **F10** | TP=2 equals TP=1 inside Piper to 7.2e-07 once parameters can be supplied | |
| **F11** | TP comm ~7.1% of GPU time; stream placement buys nothing yet; the first collective measures rank skew, not communication | |
| **F12** | TP comm hides only past one microbatch (1.00x vs 1.06x concurrency); the IR predicted it | |
| **F13** | `order` buys nothing for TP on a single stage: 1F1B is 3.7% slower, no bubble to fill | |
| **F14** | TP composes with PP with no new process group; `order` is mandatory when pp>1 | *narrows F4, corrects F11* |
| **F15** | TP x PP numerically correct on 4 GPUs to ~1e-6, under a generated 1F1B order | |
| **F16** | ~~A hardware cost model ranks TP configs backwards; rank skew is the missing term~~ | **partly retracted by F17** |
| **F17** | F16's ranking was contention: pp=2 is 27% *faster* than one GPU | *retracts F16; its ~300us guess retracted by F18* |
| **F18** | At constant work, more microbatches make TP 2.6x worse; TP and PP want opposite microbatch counts | *distinguishes F12; retracts F17's guess* |
| **F19** | CUDA graphs capture a TP step fine and buy 2%: the cost is NCCL latency, not dispatch | *retracts F18's hypothesis* |
| **F20** | EP+TP on one region silently drops one of them, order-dependent; the directive layer has no model of composition | |
| **F21** | CP's correctness fits Piper; ring attention's multi-tensor boundary and intra-region overlap do not | |
| **F22** | Executor happens-before contract made explicit; a general checker deliberately not built | |
| **F23** | A two-output TP region is silently half-reduced (now rejected); split backward composes fine | *same root as F21* |
| **F24** | `devices` is symbolic, not GPU ids: topology-aware placement cannot be expressed | |
| **F25** | TP composes with ZeRO-3 on separate regions; the dp/TP overload *enables* it | *positive side of F4* |
| **F26** | With a pipeline bubble, `order` pays: 1F1B beats GPipe by 16% under TP x PP | *closes F13* |
| **F27** | zerobubble does not beat 1F1B at pp=2/mb=4; F26's win is interleaving F and B at all | *scopes F26* |
| **F28** | EP's lowering is byte-identical to upstream after my refactor, on the shipped Qwen MoE example | |
| **F29** | A single-device `shard_tensor` was accepted and would crash in the executor; same hole upstream for `shard` | |
| **F30** | Both shipped examples still run, on real inputs for the first time | *closes the regression question* |
| **F31** | TP=2→4 is 30% faster; compute matches the roofline. ~~bandwidth-bound at 67MB~~ | *bandwidth reading retracted by F32* |
| **F32** | A TP all-reduce costs ~250us almost regardless of payload (32x payload -> 2.3x time); fusion is now worth building | *retracts F31's reading; revises F19* |
| **F33** | The fixed cost is synchronization *between* collectives, not NCCL per-call; fusion is 1.1-2.4x and competes with 1F1B, not with nothing | *corrects F32's mechanism* |
| **F34** | `fuse_collectives` shipped: bit-identical, 12-19% where comm matters, and it does *not* cost overlap | *corrects my own hypothesis* |
| **F35** | `GPipe+fusion` ties with `1F1B`; fusion's home is single-stage TP, where `order` buys nothing | *answers F33* |
| **F36** | Idle cards are not a quiet machine; `comm median/min` tells you how contaminated a window is | *makes F11 checkable* |

---

## 2026-09-10 — F1: Piper has no representation of a partitioned tensor; `shard` is EP-only and partitions nothing

**Tested.** Read all of `src/` at `upstream/main@439e960b`. No execution yet.

**IR-runtime assumption discovered.** [code] Three facts that together fix the
shape of any TP design here:

1. Parameters enter the IR as **meta tensors** — `fx.py:_meta_tensor_like` builds
   `torch.empty(example.shape, dtype, device="meta")`, stored in
   `AnnotationSegment.graphargs`. The IR therefore carries shape and dtype and
   nothing else; there is no tensor identity and no partition attribute.
2. `TrainingDAGNode.device` (`dag.py:44`) is a **device group**, not one GPU. The
   same logical sub-DAG is executed SPMD by every member. Physical binding is not
   read from the directive at all: `actor.py:_join_process_groups` sets
   `cuda:{global_rank % device_count}`.
3. Consequently **none of the three placement verbs partitions a parameter**:
   - `replicate` does not copy nodes; it only inserts `REDUCE_COMM` after `BWD_W`
     (`directives.py:_insert_reduce_comm_nodes`), or `ALL_GATHER_COMM` /
     `REDUCE_SCATTER_COMM` for ZeRO-2/3.
   - `shard` (used only for EP) sets `apply_zero=False`, deletes DP sync comms on
     the matched nodes, and inserts `A2A_COMM` on the incoming and outgoing
     activation edges at `a2a_boundary_after.tensor_idx`
     (`directives.py:_insert_shard_a2a_comm_nodes`). Every rank runs the *same*
     serialized GraphModule with *identical* shapes. Distinct expert weights arise
     **only** from per-rank random init (see F3).

So parallelism in Piper today is entirely "which collective is inserted where".
No node or edge carries a dim, mesh, or placement type.

**Unexpected.** `shard`'s name suggests a general sharding verb; it is
specifically a token-routing all-to-all insertion. That is why we propose a
distinct `shard_tensor` op rather than a `collective` field on `shard`.

**Open question.** [intent] Issue #15 asks the runtime to "manage the necessary
sharding and collectives". That is strictly more than the current code does. Is
the intended direction a placement type on IR edges (SPMD/GSPMD-style), or
directive-stated collectives with the model authored in local-shard shapes (what
EP does today)? This is the question to put to Stephanie.

**Next experiment.** Stage B — insert `TP_COMM` nodes and assert the IR
transformation on CPU, before spending GPU time.

---

## 2026-09-10 — F2: `REDUCE_COMM` is a parameter-gradient collective, not an activation collective

**Tested.** Read `executors.py:CommunicationExecutor` and `DagExecutor.run`.

**IR-runtime assumption discovered.** [code] `all_reduce_grads`
(`executors.py:74`) iterates `bucket.trainable_param_idxs` and all-reduces
`param.grad` on `dp_group`. It never touches an activation. So TP cannot reuse
`REDUCE_COMM`: TP needs an all-reduce on a **boundary activation** in forward and
on an **input gradient** in backward — a different payload at a different
insertion point.

What *is* reusable is `A2A_COMM`. Autograd is cut at every segment boundary:
forward outputs are detached and the DAG moves gradients explicitly. `FWD_A2A`
mutates `detached_outs[tensor_idx]` and re-attaches `requires_grad_(True)`;
`BWD_A2A` mutates `inp_grads[tensor_idx]`. That is structurally *exactly*
Megatron's `f`/`g` conjugate pair with `all_to_all_single` swapped for
`all_reduce`. TP's comm node is therefore a near-copy of an existing pattern, not
a new mechanism.

**Next experiment.** Stage C — `all_reduce_activation` on `ep_group` plus two
`match` arms mirroring `FWD_A2A`/`BWD_A2A`.

---

## 2026-09-10 — F3: per-rank random weight init makes numerical comparison impossible as shipped

**Tested.** Read `actor.py:_load_stage`.

**Unexpected.** [code] `_load_stage` seeds with
`g.manual_seed(1000 * self.runtime.global_rank + stage_id)` and fills every
trainable slot with `torch.nn.init.normal_(t, mean=0.0, std=0.02, generator=g)`.
Weights are therefore **different on every rank** — DP replicas are not even
mutually consistent. There is no checkpoint-loading path.

This is coherent with Piper's purpose: the shipped examples report iteration time,
throughput and peak memory (`results.csv`), never loss agreement. But it means a
"correctness vs unsharded baseline" comparison is **not possible end-to-end today**
for any parallelism dimension, TP included.

Non-trainable slots *do* have a push path: `actor.py:load_const_attrs` sends
CPU tensors that `_load_stage` matches by placeholder name (after stripping
Dynamo's `l_self_` prefix) before falling back to `zero_()`. Extending that same
mechanism to trainable slots is the minimal fix and is what Stage D2 will do.

**Open question.** Does upstream want a weight-injection path? It would also
unblock any numerical testing of PP/DP/EP and is adjacent to issue #13.

**Next experiment.** Stage D1 — validate the TP math out-of-band with a
`torchrun --nproc_per_node=2` test whose weights are sliced from one global seed,
so the collective placement is proven before the Ray/DAG stack is involved.

---

## 2026-09-10 — F4: the 2-GPU TP placement is expressible with zero launcher changes

**Tested.** Traced the launch path by hand through `schedule.py`,
`coordinator.py`, `compile.py`, `actor.py`. Not yet executed.

**IR-runtime assumption discovered.** [code] `derive_schedule_info` defines
`pp_degree` = number of distinct device-sets across `place` directives and
`dp_degree` = the size of each. `world_size = dp_degree * pp_degree` — there is
**no third axis**. For `place PP=0 devices=[0,1]`:

- `pp_degree=1`, `dp_degree=2`, `world_size=2`;
- placement group `[{"CPU":1,"GPU":1}] * 2`; `_create_actors(num_actors=1)` per driver;
- `global_rank = pp_rank + dp_rank*pp_degree = dp_rank` → `cuda:0`, `cuda:1`;
- `_join_dp_process_group`: `num_dp_groups = 2//2 = 1`, `group_ranks = [0,1]`, and it
  builds **two** NCCL communicators over that pair — `dp_group` and `ep_group` —
  deliberately, so all-reduce and all-to-all get separate NCCL proxy streams.

With no `replicate` directive on the TP region, `dp_group` goes unused and
`ep_group` is idle. Stage C binds `TP_COMM` to `ep_group`.

**Open question.** This shortcut holds only while `tp_degree` equals the place-group
size, i.e. **TP cannot compose with DP**. A real `tp_group` and a third degree in
`derive_schedule_info` are needed for that, and that is the first place TP forces a
change to Piper's world-shape assumption rather than just adding a pass.

**Next experiment.** Stage A — run the shipped 4-GPU PP×DP×EP example first, to
confirm the environment and read the rendered DAGs.

---

## 2026-09-10 — F6: TP's conjugate pair is two *outgoing* edges, and the rewrite is smaller than planned

**Tested.** `pytest -m "not gpu" test` → **30 passed** (18 upstream + 12 new in
`test/test_tp_directive.py`). CPU only; nothing executed on a GPU yet.

**Changed.** Stage B. New `shard_tensor` directive and `TP_COMM` node kind:

| file | change |
|---|---|
| `src/directives.py` | `_insert_tp_all_reduce_comm_nodes`; lifted `_boundary_info_for_edge` out of `_insert_shard_a2a_comm_nodes`; accept the op; dispatch it |
| `src/tasks.py` | `FWD_TP_ALL_REDUCE` / `BWD_TP_ALL_REDUCE` + mapping keyed on `tag["PASS"]` |
| `src/schedule.py` | accept `shard_tensor` in `_validate_directive_shape` |
| `src/dag.py`, `src/ordering.py`, `src/visualization.py` | register `TP_COMM` (docstring, critical-path comm set, node label) |

**Unexpected — the design got simpler.** [code] I expected TP to need insertion on
both sides of the region, as `shard` does for EP. It does not. Megatron's `f`/`g`
pair maps onto **two outgoing edges**:

- `g` (region exit, forward): the output is a partial sum → all-reduce the FWD
  edge leaving the region;
- `f` (region entry, backward): the gradient w.r.t. the region's input is a partial
  sum, and backward edges are reversed in this IR (`FWD: u→v` becomes `BWD: v'→u'`),
  so it is *also* an edge leaving the region.

So `_insert_tp_all_reduce_comm_nodes` only ever walks outgoing edges — half the
cases `_insert_shard_a2a_comm_nodes` handles. EP needs both sides because each
expert segment is independently sharded; TP needs one because the region is a unit.

Two further consequences fell out:

1. **Boundary-only insertion is required, not an optimization.** A TP region
   spanning two segments (column-parallel `up` feeding row-parallel `down`) has an
   internal edge carrying a tensor that is sharded *on purpose*. Reducing it would
   be wrong. `shard` inserts on every matched node's edges, which is right for EP
   and would be a silent correctness bug for TP. The pass therefore skips edges
   whose destination is inside the matched set — locked by
   `test_shard_tensor_does_not_reduce_edges_internal_to_the_region`.
2. **`_boundary_info_for_edge` already does exactly what TP needs.** For a backward
   edge it resolves the tensor index via the edge *destination's* `fwd_uid`, which
   is precisely the region's input-tensor slot, i.e. the right index into
   `inp_grads`. So TP reuses EP's resolver unchanged rather than re-deriving the
   rule. It was a closure inside the EP pass; lifting it to module scope is the only
   change to existing code, and it is locked by a test that drives it through both
   passes including the backward branch.

**IR-runtime assumption discovered.** A TP region with **no upstream segment gets
no backward all-reduce**, correctly — there is no consumer for the input gradient.
This settles plan uncertainty (4) and constrains the Stage C model: the minimal
example must keep a replicated segment on *each* side of the TP region, or half the
conjugate pair is never exercised. Locked by
`test_shard_tensor_inserts_nothing_without_a_downstream_consumer`.

**Open question.** `shard_tensor` **rejects** composition with `replicate` on the
same region, where `shard` silently deletes the DP sync comms it finds
(`_insert_shard_a2a_comm_nodes`, "shard replaces replicate-style grad/param sync
comms"). Refusing surfaces the limitation; deleting hides it. Which does upstream
want? Silent deletion is convenient for EP because an expert region genuinely has
no DP counterpart, but for TP the two directives express contradictory intents about
the same weights and I would rather the user be told.

**Next experiment.** Stage C — the executor arms and `all_reduce_activation`, then
the 2-GPU run. Blocked on GPUs: all 7 are held by other users (`haizhonz` on six,
`shaow` on one), so the code goes in first and the run waits for a free pair.

---

## 2026-09-10 — F7: TP=2 runs end to end on two B200s, and the DAG is what it should be

**Tested.** `experiments/dump_dag.py` (compiler only, no Ray, no GPU) plus a
2-GPU run of `examples/base-schedules/tp2.json` through
`test_harness.py --schedule custom`. `--schedule custom` passes a base schedule
through untouched, so the harness needed no TP-specific change.

**Changed.** Stage C: `all_reduce_activation` on `ep_group`, two dispatch arms,
`_FWD_BOUNDARY_COMM_TASKS` / `_BWD_BOUNDARY_COMM_TASKS`, plus
`examples/models/tp_mlp.py`, `tp2.json` and `experiments/dump_dag.py`.

**Result.** The lowered DAG for one device group `(0,1)`:

```
  0 forward                 COMPUTE  {PP=0}        default_stream  s0.seg0
  1 forward                 COMPUTE  {PP=0,TP=0}   default_stream  s0.seg1
  2 forward_tp_all_reduce   TP_COMM  {PP=0,TP=0}   tp_stream       [outgoing idx=0]
  3 forward                 COMPUTE  {PP=0}        default_stream  s0.seg2
  4 backward                COMPUTE  {PP=0}        default_stream  s0.seg2.bwd
  5 backward                COMPUTE  {PP=0,TP=0}   default_stream  s0.seg1.bwd
  6 backward_tp_all_reduce  TP_COMM  {PP=0,TP=0}   tp_stream       [outgoing idx=0]
  7 backward                COMPUTE  {PP=0}        default_stream  s0.seg0.bwd
  8 update                  UPD                    default_stream  upd.0
```

Exactly two comm nodes, in the two positions F6 predicted, no DP or PP
collectives, one device group. Loss falls `0.891450 -> 0.814014 -> 0.706653`,
and two runs of the same schedule agree bit for bit.

Correctness is carried by `test/test_tp_equivalence.py` (Stage D1), out of band
under `torchrun --nproc_per_node=2`: **TP=2 matches the unsharded MLP to 1.49e-08
on the output and 2.21e-09 on the input gradient**, and dropping the collectives
changes the result, so the check is not vacuous.

**Unexpected — the plan's Stage C/D ordering was wrong.** Stage C "runs without
error" turned out to be nearly worthless as evidence, because two upstream gaps
(F8, F9) meant *no observable in the system depended on what the model computed*.
The first honest number required fixing both. Getting the loss to move at all was
more work than inserting the collective.

**Open question.** `--viz` is unavailable: `graphviz`'s `dot` binary is not
installed on the host and there is no root. `dump_dag.py` covers the need for
reading node placement, and it works without a GPU, which `--viz` does not.

**Next experiment.** Stage D2 — weight injection, for numerical equivalence
inside Piper rather than beside it.

---

## 2026-09-10 — F8: the loss was computed for the backward and then dropped, in four places

**Tested.** `piper_exec_dag` returned `[]` on every run.

**Unexpected.** [code] The loss is live in four places and was stored in none:

1. the `BWD` and `BWD_I` arms build `loss_fn(...)` to drive
   `torch.autograd.backward`, and never keep the value;
2. `_update` did `losses = loss_buffer` then `loss_buffer.clear()` — one list,
   aliased and emptied, so the return was empty regardless;
3. the `UPD` dispatch arm discarded `_update`'s return value;
4. `PiperActor.run_dag` returned `None`, so `ray.get` gave the driver nothing.

So `piper_exec_dag`'s documented return value could never be non-empty. Fixed by
appending a *detached* tensor (calling `.item()` in the dispatch loop would block
on the GPU mid-DAG and perturb the schedule Piper exists to measure) and
converting in `_drain_losses` after the synchronize `_update` already performs.

**IR-runtime assumption discovered.** This is why F9 survived. With no observable
that depends on the model's arithmetic, and examples reporting only iteration
time, throughput and peak memory, a bug that zeroes the input is invisible.

**Open question.** Is upstream interested? It is a prerequisite for any
correctness lane on any parallelism dimension, not just TP.

---

## 2026-09-10 — F9: Piper trained on zero inputs, including in the shipped LLaMA example

**Tested.** The TP MLP's MSE loss came out at `0.991093`, and
`mean(labels**2)` for the same seed is `0.9910929203033447` — agreement to seven
digits, i.e. the model output contributed less than 1e-7. With no biases in the
MLP the output was in fact *exactly* zero.

**Unexpected.** [code] `fx.py:_placeholder_is_runtime_input` rejected any
placeholder carrying `meta["grapharg"]`. On torch 2.10.0 — the pinned version —
Dynamo attaches that key to **every** placeholder while the backend is running
and clears it once `torch.compile` returns. Measured inside the backend call:

```
l_self_modules_pre_parameters_weight_   runtime_input=False  grapharg_in_meta=True  is_Param=True
l_x_                                    runtime_input=False  grapharg_in_meta=True  is_Param=False
seg0 tag={'PP': 0}  input_idxs=[]  param_idxs=[0, 1]
```

The real input `l_x_` was classified as a parameter, so segment 0 had
`input_idxs=[]`. Nothing downstream objects: `_load_stage` zero-fills any slot
that is neither a trainable parameter nor a matching const attr, and the FWD arm
only substitutes real tensors for indices in `input_idxs`, so `load_input`'s data
was discarded silently.

**This is not specific to TP.** The same measurement on the shipped LLaMA debug
model gives `l_tokens_ runtime_input=False`. After the fix, `seg0 input_idxs=[0]`
for both models, and the TP MLP's loss starts moving.

The trap was ordering: the predicate returns `True` when evaluated *after*
`torch.compile` returns, so a test written the obvious way passes either way.
`test/test_runtime_inputs.py` asserts inside the backend, and asserts the meta
key is present there, so the guard fails loudly if a future torch stops
attaching it.

**Open question.** Is this a regression against a newer Dynamo rather than an
original bug? Worth checking against the torch version the paper's measurements
were taken on, because it decides whether published throughput numbers were
gathered on zero inputs — plausible for a scheduling benchmark, since shapes and
therefore timings are unaffected, but it should be stated rather than assumed.

**Next experiment.** Stage D2, then Stage E profiling.

---

## 2026-09-10 — F10: TP=2 equals TP=1 inside Piper, once parameters can be supplied

**Tested.** `experiments/check_tp_equivalence.py` on 2xB200, dim 512, hidden
2048, fp32, 3 iterations, identical global weights sliced per rank.

**Changed.** Stage D2: `piper_setup(param_overrides=...)` keyed by
`named_parameters()`, translated by `dynamo_param_placeholder_name`;
`PiperActor.load_param_overrides` mirroring `load_const_attrs`; `_load_stage`
prefers an override over `normal_()`. Unmatched keys raise. `tp1.json` baseline,
`--init fixed` in the example, `global_weights`/`shard_weights` shared with D1.

**Result.**

| | iter 1 | iter 2 | iter 3 |
|---|---|---|---|
| TP=1 unsharded | 2.951430 | 1.893224 | 1.588651 |
| TP=2 rank 0 | 2.951430 | 1.893223 | 1.588651 |
| TP=2 rank 1 | 2.951430 | 1.893223 | 1.588651 |
| no directive, rank 0 | 1.902967 | 1.328529 | 1.043335 |

Worst |diff| TP=2 vs TP=1: **7.15e-07**. Negative control: **1.10e+00**. Six
orders of magnitude apart, so the tolerance (1e-4) is not doing the work.

**Two claims the earlier evidence could not reach.**

1. The two TP ranks now agree *exactly*. That is the forward all-reduce actually
   replicating the region output — previously masked because per-rank seeding
   gave the replicated `pre`/`post` segments different weights, so the ranks
   differed for a reason unrelated to TP.
2. Agreement holds *across optimizer steps*. This is what implicates the
   backward all-reduce specifically: TP weight gradients are shard-local and
   need no collective, but the input gradient is a partial sum. Drop `f` and
   iteration 1 still matches while iteration 2 diverges. Comparing only the first
   iteration would not have tested `f` at all.

**IR-runtime assumption discovered.** Overrides have to be *per-rank values*,
not a global tensor plus a partition spec, because the runtime has nowhere to put
the partition spec (F1). `shard_weights` therefore lives in the example, next to
where the model already declares its TP-local shapes. That is consistent, but it
means **every** consumer of TP has to re-state the sharding rule the model
already implies — the concrete cost of F1, and the thing an auto-TP search
(Stage F) would have to fix first.

**Open question.** `dynamo_param_placeholder_name` hard-codes Dynamo's
lifted-attribute spelling (`l_self_modules_<path>_parameters_<attr>_`). It is
locked by a test, but it is a private naming convention. Would upstream rather
key overrides on `(bucket_key, param_index)`, which is stable but opaque, or
expose the placeholder names so callers can look them up?

**Next experiment.** Stage E — one profiling question: what fraction of step time
is the two `TP_COMM` all-reduces, and does moving them off `tp_stream` onto
`default_stream` change it.

---

## 2026-09-10 — F11: TP communication is ~7% here, stream placement buys nothing yet, and the first collective of an iteration is not a measurement

**Tested.** `dim 4096, hidden 16384, batch 4096, bf16, 2xB200 (GPU 3,5), 20
profiled iterations`, via `--pytorch-profiler` and `experiments/profile_tp.py`.
The box was shared throughout; GPUs 0/1/2/4/6 were at 100% util under other
users, and GPU 5 went from 0% to 100% mid-experiment.

**Changed.** `experiments/profile_tp.py` (per-DAG-node GPU time attribution),
`experiments/summarize_runs.py`, `examples/base-schedules/tp2_default_stream.json`.

### The number

Piper labels every GPU event with the DAG node that issued it, so the trace
aggregates per node uid rather than per kernel name:

| node | task | min (us) | max/min |
|---|---|---|---|
| `s0.seg1.bwd` | backward | 764.4 | 1.01x |
| `upd.0` | update | 531.8 | 1.02x |
| `s0.seg2.bwd` | backward | 455.8 | 1.02x |
| `s0.seg1` | forward | 401.5 | 1.02x |
| `s0.seg0` | forward | 96.4 | 1.02x |
| `s0.seg2` | forward | 92.0 | 1.06x |
| `s0.seg0.bwd` | backward | 87.5 | 1.09x |
| `tp_all_reduce.1` | backward all-reduce | **93.0** | 26.6x |
| `tp_all_reduce.0` | forward all-reduce | 458.2 | 66.5x |

One all-reduce costs **~93 us** for a 32 MiB bf16 payload, i.e. ~360 GB/s
effective over NVLink. Two of them against 2435 us of compute:

> **TP communication is ~7.1% of GPU time** in this configuration.

That is an upper-confidence *lower bound*: nothing on a contended machine makes a
kernel faster than its uncontended time, so the minimum converges to the truth
from above.

### Stream placement buys nothing here, as the IR predicts

`shard_tensor` takes a `stream`. Compare `tp_stream` against `default_stream`:

| | compute min-sum | backward all-reduce min |
|---|---|---|
| `tp_stream` | 2434.7 us | 93.0 us |
| `default_stream` | 2448.6 us | 90.6 us |

Compute agrees to 0.6% and the collective to 2.6% — no effect. This was
predictable from the lowered DAG without running anything: the serial order is
`s0.seg1 -> tp_all_reduce.0 -> s0.seg2`, so with one region and one microbatch
**the collective has nothing to overlap with**, and a separate stream only adds
event synchronization.

So Piper's stream programmability — the thing the paper is about — cannot pay for
itself on TP until there is concurrent work: several microbatches, several TP
regions, or TP composed with PP/DP. That is the natural next experiment, and it
is also the first place TP will need a real `tp_group` (F4).

### Three methodology findings, all load-bearing

1. **Report the minimum, never the mean.** Communication varied up to 66x across
   iterations while compute stayed within 1.1%. A mean tracks whoever else is
   using the machine.
2. **Owning your GPUs is not enough to measure TP.** Compute was rock-steady on
   two 0%-util cards, yet the collectives were squeezed anyway: SMs are per-GPU
   and can be held exclusively, but **NVLink/NVSwitch is machine-wide**. Any TP
   communication number needs the whole box quiet, not two free cards.
3. **The first collective of an iteration is not a communication measurement.**
   Same payload, same code path, but per rank:

   | | rank 0 | rank 1 |
   |---|---|---|
   | `tp_all_reduce.0` (forward) | min 458, values 2858–30485 | min **93.8**, values ~94 |
   | `tp_all_reduce.1` (backward) | min 93.0 | min 93.6 |

   Perfectly complementary: rank 0 arrives early and spins inside the NCCL
   kernel waiting for rank 1. Each device is driven by its own Ray actor and
   `piper_exec_dag` fans out with `ray.get`, so iteration start times differ by
   milliseconds and **the first collective absorbs the skew**. The backward
   all-reduce is clean because the forward one already synchronized the ranks.

   `profile_tp.py` now detects this and says so. It matters beyond TP: any
   overlap analysis on Piper that reads the first collective's duration as
   communication cost will overstate it, which is exactly the quantity a
   DualPipe-style schedule claims to hide.

**Open question.** Does the skew shrink with more microbatches (the first
collective absorbs it once per iteration, so its relative cost should fall), or
is it per-collective? That decides whether it is a measurement artifact or a real
cost of the Ray-per-device driver model.

**Next experiment.** Multi-microbatch TP, which is the first configuration where
`stream` and `order` can actually do something for TP — and where a real
`tp_group` becomes necessary.

---

## 2026-09-10 — F12: TP communication hides only with more than one microbatch, and the IR said so before the GPU did

**Tested.** `dim 8192, hidden 32768, bf16, 2xB200 (GPU 3,5, both 0% util —
first genuinely clean measurement), 10 profiled iterations per configuration`,
via `experiments/check_tp_overlap.py`.

**Changed.** `experiments/check_tp_overlap.py`, `tp2_mb1.json`, `tp2_mb4.json`,
and a sum-vs-span concurrency metric in `profile_tp.py`.

**Method.** Summed kernel durations do not shrink when work overlaps, so the sum
alone cannot answer "was the collective hidden". Compare it against the wall span
of the iteration's GPU timeline: `sum/span > 1` means two streams ran
concurrently.

**The prediction came from the IR.** With one microbatch the lowered serial order
is `s0.seg1 -> tp_all_reduce.0 -> s0.seg2`, so the collective has nothing to
overlap with and `shard_tensor`'s `stream` field is inert. With four microbatches
`tp_all_reduce.0` (MB 0) depends only on `s0.seg1` (MB 0) and feeds only
`s0.seg2` (MB 0), while `s0.seg1.splitMB1` (MB 1) depends on neither — so
`_insert_tp_all_reduce_comm_nodes` leaves room to hide, and the only
`TP_COMM -> TP_COMM` edges are the temporal chain within `tp_stream` that a
single stream requires anyway.

**Result** (concurrency = sum/span, 10 iterations each):

| config | min | median | max | comm share of GPU time |
|---|---|---|---|---|
| mb1, batch 1024 | 0.72x | 0.79x | 0.80x | 3.0% |
| mb1, batch 8192 | 0.99x | 1.00x | 1.00x | 4.2% |
| mb4, batch 1024 | 0.85x | 1.11x | 1.41x | 5.6% |
| mb4, batch 8192 | **1.06x** | **1.06x** | 1.28x | 6.4% |

Three things, all confirmed rather than assumed:

1. **One microbatch never overlaps.** `mb1, batch 8192` gives 0.99–1.00x on
   every one of ten iterations. That is the cleanest control in this log: the GPU
   timeline is fully saturated *and* has exactly zero concurrency, because the
   only non-default-stream work in the DAG is two collectives with nothing to run
   beside them. The IR predicted it; the measurement matched.
2. **Four microbatches do overlap**, 10/10 iterations above 1.0. TP
   communication is hidden behind other microbatches' compute with no `order`
   directive written — microbatch independence after `split` is enough.
3. **The earlier 0.72–0.80x was Piper's dispatch cost, not a TP property.** At
   batch 1024 the GPU timeline sat ~20–28% idle; the gaps are per-node Python
   dispatch in `DagExecutor.run` (33 nodes at mb4), and they vanish once GPU work
   per node dominates. This is why F11's wall-clock `iter_time` comparisons were
   useless: at that scale the step was mostly not on the GPU.

**This answers F11's open question and closes Stage E.** `shard_tensor`'s
`stream` field is a no-op for a single-region, single-microbatch TP schedule, and
Piper's scheduling language starts paying for TP at two or more microbatches. So
the interesting TP schedules are exactly the composed ones — which is where a
real `tp_group` becomes necessary (F4), since `tp_degree` is currently pinned to
the place-group size.

**Open question.** Overlap here comes free from microbatch independence. Does an
explicit `order` directive beat it — e.g. deliberately interleaving MB *i*'s
collective with MB *j*'s backward, DualPipe-style? That is the first experiment
where TP would actually exercise the part of Piper the paper is about, rather
than just riding on it.

**Next experiment.** Either the `order` question above, or Stage F. Stage F now
has its evidence base: F1 (no partitioned-tensor representation), F4 (no third
parallel axis), F10 (every consumer must restate the sharding rule) are the three
concrete things a TP-selection search would need the IR to express.

---

## 2026-09-10 — F13: `order` buys nothing for TP on a single stage, and the reason is structural

**Tested.** Three interleaved repetitions per arm (A/B/A/B/A/B, to blunt time
drift on a shared box), `dim 8192, hidden 32768, batch 8192, bf16, 2xB200`,
10 profiled iterations each, minimum span per run.

| arm | span min (us) | concurrency max |
|---|---|---|
| default order | 60059 / 57651 / **57063** | 1.09–1.17x |
| hand-written 1F1B `order` | 59610 / **59152** / 60770 | 1.03–1.05x |

1F1B is **~3.7% slower** and overlaps *less*. Explicable, not surprising:

- With one stage there is **no pipeline bubble to fill**, which is what 1F1B
  exists for. It has no upside to deliver here.
- The directive adds four `ORDER_DUMMY` nodes and temporal edges that only
  **constrain** `_resolve_default_stream_order`, which was already free to order
  same-level compute by downstream count.
- The default order was already good: `_serial_topological_order` gives
  critical-path comm (including `TP_COMM`) priority 1 against compute's 3, so at
  equal topological level the collective is issued first and lands beside another
  microbatch's compute. That is where F12's free 1.06x came from.

So the overlap TP gets today is not something `order` improved on. `order` should
pay for TP where it pays for anything — **across pipeline stages**.

---

## 2026-09-10 — F14: TP composes with PP without a third axis, but `order` is mandatory there

**Tested.** `examples/base-schedules/pp2_tp2_mb4_1f1b.json`, compiled on CPU via
`experiments/dump_dag.py`. No GPU run yet: `cmu-gpu pick --count 4` reports zero
idle cards.

**Changed.** `TPMlp(…, n_stages)` wraps each block in its own PP scope with the
TP region nested inside; weight keys moved under `blocks.<i>.`; `dump_dag.py` and
the driver gained `--stages`.

**Result — it composes, and needs no new process group.** TP borrows the dp
dimension while PP uses the stage dimension:

```
2 per-PP-rank DAGs, 41 nodes each, devices (0,2) and (1,3)
rank 0: 12 forward, 12 backward, 4 SEND, 4 RECV,
        4 forward_tp_all_reduce, 4 backward_tp_all_reduce, 1 UPD
```

**This corrects F11.** I had guessed TP x PP would also need a `tp_group`. It
does not — only **TP x DP** needs the third axis Piper lacks, because there TP
and DP would both want the same dimension. So the useful composed TP schedules
are reachable today, and F4's limitation is narrower than recorded.

**Unexpected — with `pp_degree > 1`, `order` is not optional.** The same schedule
without an `order` directive fails:

```
ValueError: expected distinct device sets across split components,
got [(0, 2), (1, 3), (0, 2), (0, 2), (0, 2), (0, 2)]
```

Four of the six components are PP=0 forward-only chains, one per microbatch:
`s0.seg0 -> s0.seg1 -> tp_all_reduce.0 -> s0.seg2 -> send.0`. The cause is in
`build_training_dag`: it bridges forward to backward only at the **globally**
last forward node, which belongs to the last stage. So a non-last stage's forward
chain reaches its own backward chain only *through* the next stage, and
`_insert_send_recv_comm_nodes` severs precisely that path (SEND and RECV are
deliberately unconnected). `order`'s temporal edges are the only thing that
reconnects them — which is why the harness always appends one and why every
shipped base schedule is run with `--schedule 1f1b` or similar.

This is a real precondition that nothing states: `--schedule custom` will happily
accept a `pp_degree > 1` base schedule with no `order` and fail with a message
about device sets. `src/piper.py` now detects the signature (components sharing a
device set, some forward-only), names one, and says what to add.

**Open question.** Does the fwd->bwd bridge belong per stage rather than only at
the globally last forward? That would make each PP rank's sub-DAG connected on
its own and remove the hidden dependency on `order`. It would also change
scheduling freedom, so it is a design question for upstream, not an obvious fix.

**Next experiment.** `pp2_tp2_mb4_1f1b` on four GPUs: numerical equivalence
against `tp1`/`pp1` baselines, then whether `order` finally beats the default on
TP once there is a pipeline bubble to fill. Blocked on four idle cards.

---

## 2026-09-10 — F15: TP x PP is numerically correct on four GPUs

**Tested.** `experiments/check_tp_equivalence.py --pp`, `dim 512, hidden 2048,
batch 32, fp32, stages=2, 4 microbatches`, identical global weights sliced per TP
rank. Small on purpose: this is a correctness gate, and three of the four cards
were shared, so nothing timing-related is claimed.

| | iter 1 | iter 2 |
|---|---|---|
| baseline: 1 GPU, stages=2, tp=1 | 44.384857 | 23.111019 |
| TP x PP: 4 GPU, stages=2, tp=2, 1F1B, rank 0 | 44.384819 | 23.111044 |
| TP x PP: 4 GPU, stages=2, tp=2, 1F1B, rank 1 | 44.384819 | 23.111044 |

Relative difference ~1e-6, fp32 reduction-order noise. Three things hold at once:

- both TP ranks are **identical**, so the forward all-reduce is replicating the
  region output on every stage, not just the last;
- agreement survives the optimizer step, so the backward all-reduce is right
  under pipelining too;
- it matches a baseline with **no** TP, **no** PP and **no** P2P communication,
  so `shard_tensor` composes with `place`, `split` and `order` without any of
  them corrupting the others.

`shard_tensor` therefore works in the configuration that matters: composed with
Piper's pipeline scheduling, four microbatches deep, under a generated 1F1B
order. That is the target the plan set for Stage C/D, reached one axis further
than planned.

**Open question.** Only performance is left unanswered here, and it needs four
idle cards. The specific question is F13's, re-asked where it can be answered:
with a pipeline bubble to fill, does `order` finally beat the default schedule
for TP?

**Next experiment.** Either the four-GPU `order` measurement when the box frees
up, or Stage F, whose evidence base is now F1 (no partitioned-tensor
representation), F4/F14 (no third axis, so TP x DP is out but TP x PP is in), and
F10 (every consumer restates the sharding rule).

---

## 2026-09-10 — F16: a hardware cost model ranks Piper's TP configurations backwards; the missing terms are implementation overheads

**Tested.** `experiments/tp_search.py`, calibrated on F11–F13, checked against
three configurations measured on 2xB200 at `global batch 8192, dim 4096,
hidden 16384, stages 2, bf16, mb 4`.

### What is even searchable, and why it is this little

Before building anything, the limits this log already recorded fix the space:

- **`tp` is not a schedule-level choice.** Piper's IR has no representation of a
  partitioned tensor (F1), so TP-local shapes live in the *model*; changing `tp`
  means rebuilding and retracing it. Any TP search therefore spans model
  construction, not just directives. This is the concrete, priced cost of F1.
- **`pp * tp <= gpus` with `tp` = place-group size** (F4), so TP x PP is in the
  space and TP x DP is out (F14).
- **Microbatches replicate the DAG, they do not split the batch**
  (`_apply_split_directive` copies nodes; `load_input` gives every copy the same
  tensor). Holding the global batch fixed means per-microbatch batch is
  `global / mb`.

### The pure roofline got the answer backwards

A model over compute, bandwidth, pipeline bubble and per-node dispatch:

| config | predicted | measured (end-to-end) |
|---|---|---|
| one GPU, mb 4 | 12494 us | **18720 us** |
| pp=2, mb 4 | 7846 us | 23364 us |
| tp=2, mb 4 | 7571 us | 26523 us |

Predicted order `tp=2 < pp=2 < one GPU`. Measured order **exactly reversed**:
staying on one GPU is 25–42% *faster* than either parallel configuration. A cost
model that says "parallelize" where the machine says "do not" is worse than no
model.

### The missing term is not bandwidth, it is rank skew

Per-rank per-iteration GPU kernel totals, which do not depend on iteration
splitting:

| config | kernel sum | of which communication | payload at 360 GB/s |
|---|---|---|---|
| one GPU | 14244 us | 0 | — |
| tp=2 | 25085 us | **12875 us** | ~745 us |

Communication kernels run **17x longer than their payload justifies**. That is
F11 at full strength: each device is driven by its own Ray actor,
`piper_exec_dag` fans out with `ray.get`, iteration start times differ by
milliseconds, and the first collective absorbs the difference by spinning inside
the NCCL kernel. So **communication cost is not a function of communication
volume here** — it is a function of the driver's scheduling jitter.

The compute side, by contrast, is fine: one GPU predicted 12494 against 14244 of
measured kernel time (1.14x), and F12's calibration holds to 2%.

### Adding two measured implementation terms restores the useful decision

`DRIVER_OVERHEAD_US = 6200` (Ray driver + actor RPC per step, fitted on the
one-GPU run) and `SKEW_US = 12000` (charged once per step as soon as more than
one rank exists, from the 17x measurement above):

| config | predicted | measured | ratio |
|---|---|---|---|
| one GPU | 18694 us | 18720 us | 1.00x *(in-sample)* |
| pp=2 | 26046 us | 23364 us | **1.11x** *(out-of-sample)* |
| tp=2 | 25771 us | 26523 us | 0.97x |

It now picks the fastest configuration. It still **cannot separate TP from PP**:
it puts them 1% apart where the measurement puts them 13% apart. So it is a
filter — "does parallelising pay at all" — not a ranking. `--ranking-check` says
so rather than claiming a win.

### What this means for automatic TP selection

Neither overhead term scales with problem size, so both dominate at small and
medium scale and decide it wrongly if omitted. Three consequences:

1. **A TP autotuner for Piper must model the runtime, not the hardware.** Every
   plausible hardware-only model would have chosen wrong here, and would keep
   choosing wrong until the model is large enough for compute to swamp a ~18 ms
   fixed cost.
2. **The highest-value optimization for TP is not a schedule.** It is lowering
   dispatch and skew — batching node dispatch, or driving all devices from one
   process instead of one Ray actor each. That would do more for TP at this scale
   than any choice `shard_tensor`, `stream` or `order` can express, and it also
   explains F13 (`order` bought nothing) and F11 (the first collective is not a
   measurement) as the same root cause.
3. **The searchable space is genuinely small**, and F1 is why: because `tp`
   cannot be varied without retracing, a search has to drive model construction,
   so it cannot live inside the scheduling language that Piper's design is
   otherwise built around.

**Open question.** Is the ~12 ms skew intrinsic to Ray actors, or is it
`piper_exec_dag`'s fan-out pattern specifically (one `ray.get` over N actors per
step)? A single-process multi-device driver would answer it, and is a bounded
experiment. Until it is answered, no performance conclusion about TP in Piper is
about TP.

**Next experiment.** Measure the skew directly — timestamp iteration entry per
actor and histogram the differences — rather than inferring it from inflated NCCL
kernel durations. That is the cheapest way to confirm the root cause, and it needs
only two GPUs.

---

## 2026-09-10 — F17: **F16's ranking was contention, not a result.** Retracting it, and a better hypothesis

**Tested.** The same three configurations as F16, but interleaved A/B/C three
times and reduced by **minimum** rather than reported from one run each.
`dim 4096, hidden 16384, batch 2048/microbatch, stages 2, tp per config, bf16,
12 iterations`. Machine state identical at start and end (GPUs 0/1/2/4 at 100%
under other users, 3/5/6 idle).

| config | rep1 | rep2 | rep3 | **min** | F16 reported |
|---|---|---|---|---|---|
| pp=2 | 11.96 | 21.74 | 12.20 | **11.96 ms** | 23.36 ms |
| one GPU | 16.49 | 17.64 | 17.68 | **16.49 ms** | 18.72 ms |
| tp=2 | 17.83 | 16.62 | 27.15 | **16.62 ms** | 26.52 ms |

### Retraction

**F16's headline — "staying on one GPU is 25–42% faster than either parallel
configuration" — is withdrawn.** With interleaving and a minimum, `pp=2` is
**27% faster** than one GPU, not 25% slower. F16 took one sample per
configuration, and the samples that involved NVLink were the ones that drift by
a factor of two. Single-GPU numbers barely moved (16.49 against 18.72) because
they use no interconnect; `pp=2` moved by 2x and `tp=2` by 1.6x.

This is my own methodology finding (F11: report the minimum, never the mean)
applied everywhere except the place it mattered most. The `SKEW_US = 12000`
constant that "restored" the ranking was fitted to a contaminated sample, so it
was fitting the machine's other users.

### What survives

- The **compute** side of the model still holds: one GPU predicted 12494us
  against 16.49ms measured wall clock, and F12's GPU-side calibration is still
  2%.
- **Direct measurement** replaces the inference: median entry spread is
  **157–2831us**, not 12000us. Ranks do not enter steps 12ms apart.
- Communication kernels really do inflate — 12875us per rank per iteration
  against ~745us of payload — but that inflation is **contention plus arrival
  differences at each collective**, not a per-step constant.

### The better hypothesis, and it is testable

`tp=2` is 39% slower than `pp=2` (16.62 against 11.96 ms) where the compute model
separates them by only 4% (7497 against 6247us). The gap is 4.66 ms. Count the
synchronizing operations per step:

| config | collectives per step per rank | 4.66 ms / count |
|---|---|---|
| tp=2 | 16 all-reduces (4 mb x 2 stages x 2 passes) | **291 us** |
| pp=2 | 8 P2P send/recv (4 mb x 2), which are pairwise, not group-wide | — |

So the missing term looks like a **fixed cost of ~300us per collective**, not a
per-step skew. That also explains why TP loses to PP despite similar FLOPs: TP
inserts a group-wide synchronization point per region per pass per microbatch,
and PP inserts only pairwise transfers.

**Prediction to falsify it:** the cost should scale with the *number* of
collectives, so `tp=2, mb=1` (4 all-reduces) should sit about 3.5 ms below
`tp=2, mb=4` (16 all-reduces) once compute is held constant. If instead the two
differ by the compute ratio alone, the per-collective model is wrong too.

**Open question.** If ~300us per collective is real, where does it come from?
Candidates, in order of how cheaply they can be separated: the per-step
`torch.cuda.synchronize()` in `_update`; cross-stream event waits around each
comm node; NCCL launch cost on a fresh kernel each time; genuine arrival
differences from the two ranks' independent dispatch loops.

**Next experiment.** The falsification above: hold compute constant, vary the
collective count via microbatches, and see whether the gap tracks the count.

---

## 2026-09-10 — F18: at constant total work, more microbatches make TP *worse*, and F12 was answering a different question

**Tested.** Global batch held at 8192 while microbatches vary, so
`batch = 8192 / mb` and total FLOPs are constant. `dim 4096, hidden 16384,
stages 2, tp 2, bf16`. Step time from three interleaved repetitions, minimum.
GPU breakdown from one profiled run each, so it carries F11's caveat.

| mb | collectives | batch | **step (min of 3)** | compute | upd | comm | payload/coll |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 8192 | **10.46 ms** | 6826 us | 1069 | 1223 us | 67 MB |
| 2 | 8 | 4096 | **10.32 ms** | 7680 | 1076 | 7733 | 33.5 MB |
| 4 | 16 | 2048 | **16.15 ms** | 9082 | 1084 | 9626 | 16.8 MB |
| 8 | 32 | 1024 | **27.03 ms** | 11028 | 1103 | 32318 | 8.4 MB |

`upd` is flat, as it must be — the parameter count does not change — which is a
useful internal check that the decomposition is attributing correctly.

### F12 and F18 are not in conflict; they ask different things

This looks like it contradicts F12 ("more microbatches let TP communication
hide"). It does not, and the difference is Piper's microbatch semantics:
**`split` replicates the DAG, it does not partition the batch**.

- **F12** held *per-microbatch* batch fixed, so total work grew with `mb`. Question:
  given a batch, does adding microbatches expose overlap? **Yes** — concurrency
  1.06x, 10/10 iterations.
- **F18** holds *total* batch fixed, so work is constant and `mb` only changes
  granularity. Question: given a total batch, is splitting it finer better?
  **No, sharply worse** — 10.46 to 27.03 ms, a factor of 2.6.

Both are true. The second is the one a user actually faces, and it is easy to
read F12 as answering it. Recording the distinction is the point of this entry.

### Two mechanisms, one reliable and one only directional

1. **Kernel efficiency loss, reliable.** Compute rises 6826 -> 11028 us (+62%) at
   *constant* FLOPs, purely because `batch` falls 8192 -> 1024. Monotone and
   smooth across all four points.
2. **Collective count, directional only.** `comm` rises 1223 -> 32318 us while
   payload per collective *falls* 8x, so cost is clearly driven by the number of
   collectives rather than the bytes. But per-collective cost comes out at 306,
   967, 602, 1010 us — **not monotone** — so these single profiled runs cannot
   give a scaling law. F11's 66x communication variance is larger than the
   differences here.

**Retracting my own guess from F17.** I proposed ~300 us per collective, then a
quadratic law from two endpoints (wait ratio 66x against a collective ratio of
8x). The intermediate points do not fit either. The endpoints agreed with a
square by coincidence, which is what two points always do.

### The practical consequence, and a real tension for TP x PP

On Piper, TP wants **as few microbatches as possible**: more microbatches
simultaneously shrink the kernels and multiply the synchronization points. PP
wants the opposite — microbatches are what fills the pipeline bubble (F13:
`order` bought nothing precisely because a single stage has no bubble).

So **TP and PP have opposing preferences on the same knob**, and a composed
TP x PP schedule has to trade them off. That is a genuine scheduling question
that Piper's language *can* express, and it is the first one this project has
found where the answer is not obvious from the IR.

**Open question.** Where does the per-collective cost come from? It is not
payload. Candidates, cheapest to separate first: the two ranks each running an
independent Python dispatch loop, so every collective re-synchronizes them and
pays whatever jitter accumulated since the last one; NCCL launch on a fresh
kernel per node; the cross-stream event waits around each comm node. A CUDA-graph
capture of the step would remove the first and third at once and is the sharpest
available test, since the DAG's shapes and order are entirely static.

**Next experiment.** Quantify `comm` properly: interleaved repetitions with a
minimum, not one run per point. Then the scaling law is worth fitting, and only
then is it worth attributing.

---

## 2026-09-10 — F19: CUDA graphs capture a Piper-shaped TP step fine, and buy 2%. The cost is NCCL latency, not dispatch

**Tested.** `experiments/probe_cudagraph.py` under `torchrun --nproc_per_node=2`,
2xB200. Deliberately independent of Piper: it reproduces the *shape* of a TP step
(compute on the default stream, all-reduce on a second stream, joined by events)
with no Ray, no DAG executor, no per-step Python bookkeeping. 16 collectives per
step, batch 512, dim 4096, hidden_local 8192, bf16.

Staged so a failure would localize. All three capture stages passed:

```
stage 1  captured compute + autograd
stage 2  captured NCCL all-reduce on one stream
stage 3  captured two-stream step with events
stage 4  eager  min 1685.9 us   per collective 105.4 us
         graph  min 1651.1 us   per collective 103.2 us
         payload bound 186.4 us  per collective  11.7 us
         -> graph 1.02x faster; eager 9.0x above payload, graph 8.9x
```

### The hypothesis is dead

F18 proposed that per-collective cost comes from each rank running its own Python
dispatch loop, so every collective re-synchronizes two independently drifting
launch streams. A CUDA graph replaces the entire step with one launch and removes
that drift completely. It bought **2%**. So dispatch drift is not the cost.

What remains is NCCL itself: a 4.2 MB two-rank all-reduce at ~105 us against an
11.7 us bandwidth bound is **latency-bound, not bandwidth-bound**, plus whatever
the other tenants are doing to the interconnect (F11).

Worth stating plainly because it inverts the direction I gave: **Piper's runtime
overhead is not TP's bottleneck.** Dispatch cost, Ray fan-out and rank skew are
all real and all measured, and none of them is what makes TP slow here. The cost
is `number of collectives x NCCL latency`.

### What that changes

- **CUDA graphs are not worth doing for this.** Feasible — which is itself worth
  knowing, since the DAG is fully static — but the payoff is noise. Recorded so
  nobody spends a week on it.
- **The lever is the collective *count*.** F18 already measured that lever
  end-to-end: at constant total work, 4 collectives beat 16 by 35% (10.46 vs
  16.15 ms). No new mechanism needed to exploit it on a single stage — just use
  fewer microbatches.
- **A fusion pass is therefore premature.** Collectives from different
  microbatches are independent and could be coalesced, cutting 16 to 4. But that
  only matters when something *else* forces many microbatches, i.e. filling a
  pipeline bubble under TP x PP (F13, F18's opposing-preference tension) — and
  measuring that needs four idle GPUs, which this host has not had. Building it
  now would be unfalsifiable.
- **Revised target.** The goal I stated as "make TP cost track communication
  volume" is unreachable on this machine: NCCL's own latency puts the floor at 9x
  the payload bound. The reachable goal is "minimize collective count for a given
  parallel configuration".

**Open question.** Does the 9x latency gap close at larger payloads? At mb=1
(67 MB per collective) F18 measured 306 us against a 186 us bound — only 1.6x. So
the gap is a small-message effect, which is exactly what fusion would exploit,
and it sets the condition under which fusion becomes worth building: enough
microbatches that per-collective payload falls into the latency-bound regime.

---

## 2026-09-20 — F20: two boundary-comm directives on one region silently drop one of them

**Tested.** `shard` (EP) and `shard_tensor` (TP) applied to a region matched by
both, on CPU.

| directive order | result |
|---|---|
| `shard` then `shard_tensor` | 4 `A2A_COMM`, **0 `TP_COMM`** — TP dropped entirely |
| `shard_tensor` then `shard` | 2 `A2A_COMM`, 2 `TP_COMM` — both incomplete |

Both passes rewrite a matched node's activation edges to route through their own
comm node. Whichever runs second finds `dst` is a comm node rather than compute,
fails its own boundary predicate, and inserts nothing — silently. **The semantics
depend on the order the directives appear in the JSON, and one order is simply
wrong arithmetic with no diagnostic.**

My own gap: `shard_tensor` already refuses to compose with `replicate` (F6), and
I never checked `shard`. Fixed with one check that runs *before* either pass, so
it cannot itself be order-dependent, plus a test for both orders and one that
disjoint regions still compose.

### Why this is more interesting than the bug

This is the second silent, order-dependent failure in the directive layer, after
F14 (a `pp>1` schedule without `order` fails with a message about device sets).
Both have the same shape: **the directive language has no model of how directives
interact.** Each pass validates its own preconditions against the DAG, and
nothing validates the composition. The failure mode is silence, because a pass
that finds no matching edge does nothing rather than complaining.

That is a real risk for a system whose thesis is user-programmable scheduling —
and the checks that would catch it are static properties of the lowered DAG:

- every activation edge crossing a stream boundary has a consumer arm that waits
  on the producer's event (I hit the functional version of this in Stage C: the
  consumer arms match on exact `task_type`, so an unregistered comm kind is
  skipped and the node silently reads the wrong inputs);
- every compute region is claimed by at most one boundary-comm directive (this
  entry);
- every non-last pipeline stage's forward reaches its own backward (F14);
- no buffer is released before its last reader.

**Open question / next direction.** Is a static checker over the lowered
`TrainingDAG` worth building? It needs no GPU, which matters given this host, and
it targets exactly the class of bug this project kept finding by accident. The
first three items above are each a few lines over `dag.nodes`/`dag.edges`; the
interesting one is the first, because it requires a machine-readable statement of
which `task_type`s each executor arm waits on — i.e. the executor's
happens-before contract would have to be declared rather than implied.

---

## 2026-09-20 — F21: CP's correctness fits Piper; ring attention's two key properties do not

**Tested.** Traced a ring-attention-shaped module (`cp_degree=2`, one
`annotate("CP")` per ring step) and read the segmentation. CPU only.

```
3 segments (2 ring steps + output)
  seg0 tag={'CP': 0}  inputs=3  boundary_after=tensor_idx=3
  seg1 tag={'CP': 1}  inputs=4  boundary_after=tensor_idx=0
  seg2 tag={'CP': 2}  inputs=1  boundary_after=None
```

**Structurally CP does fit.** Writing each ring step as its own annotated region
puts the K/V exchange on a *region boundary*, which is exactly where Piper can
insert communication — the same property that let TP in (F6). So the
boundary-comm machinery `shard_tensor` established is reusable; CP would need a
new collective kind (an intra-group P2P ring, which is neither `all_reduce` nor
`all_to_all` nor the existing cross-device-set `SEND`/`RECV`), not a new
mechanism.

**Two things do not fit, and one of them is architectural.**

1. **A boundary carries exactly one tensor.** `a2a_boundary_after` is
   `{"tensor_idx": <int>}`, chosen by `_select_boundary_tensor_idx`'s heuristic
   (floating-point +2, requires_grad +1, take the best). Every existing insertion
   reads that single index — `_insert_shard_a2a_comm_nodes` does, and so does my
   `_insert_tp_all_reduce_comm_nodes`. Ring attention must move **K and V**, two
   tensors, every step. TP needs one (the region output) and EP needs one (the
   token tensor), so the assumption held until now. **Fixable**: make it a list.
2. **Communication between regions cannot overlap the region it belongs to.** A
   boundary comm node sits `seg_i -> comm -> seg_{i+1}`, so it cannot start until
   `seg_i` finishes. Ring attention's whole point is sending block *i+1* while
   computing block *i*. In Piper that overlap is only available across
   *microbatches* (F12), never within a region. **Architectural**: the region is
   the smallest schedulable unit, so intra-region overlap has no expressible form.

So Piper could run CP correctly and would lose the optimization that motivates
ring attention. That is a sharper statement of the limit than TP produced: TP was
*fully* expressible once the collectives were in place (F10, F15), CP is not.

**Also found:** an `order` filter group may not span device sets — its
`ORDER_DUMMY` would have no single device to sit on, so
`_validate_order_edge` rejects it. That is why `build_1f1b_schedule` emits one
`order` directive *per pipeline rank* rather than one global ordering, and why a
hand-written GPipe order has to be written per stage too.

**Open question for #15.** Is (2) worth changing? Allowing a comm node to be
scheduled *concurrently with* the region that produces its input would mean the
region is no longer atomic to the scheduler — a real change to what a
`TrainingDAG` node means. The cheaper alternative is to accept region-granular
overlap and write ring steps finely enough that it suffices, which is what the
probe above does, and to measure whether that recovers most of the benefit.

---

## 2026-09-20 — F22: the executor's happens-before contract, stated rather than checked

**Decision, with the reasoning, because the scope shrank deliberately.**

F20 listed four static properties of a lowered `TrainingDAG` worth checking. On
inspection three already have a mechanism:

| property | status |
|---|---|
| one boundary-comm directive per region | enforced, F20 |
| non-last stage's forward reaches its backward | diagnosed in the error, F14 |
| no buffer released before its last reader | `BufferStore.refcounts` counts `data_succs` and the dispatch order is topological, so a consumer cannot run after the release |
| cross-stream edges are awaited | **this entry** |

The fourth is the one that bit this project: `DagExecutor.run` dispatches on exact
`task_type` *and finds predecessors the same way*, so a comm kind that has a
dispatch arm but is missing from the four predecessor lookups is skipped in
silence and the consumer takes its fallback branch — reading the wrong inputs with
no error. `TP_COMM` was in exactly that state mid-Stage-C.

**I did not build a general checker.** A generic "every cross-stream edge is
awaited" pass would have to recover the contract from the `match` arms, which
means either AST analysis of the executor or a hand-written table that silently
rots when the executor changes. Both are more machinery than the problem
justifies: new comm kinds arrive roughly once per parallelism dimension (EP,
ZeRO, TP — three times in the project's life).

Instead `test/test_executor_contract.py` makes the contract **explicit and
non-defaultable**. Every `TaskType` must be classified as feeding COMPUTE or not;
everything that feeds COMPUTE must appear in the consumer lookups; every type must
have a dispatch arm. Adding a kind fails these until it is classified and wired.
Three assertions, no framework, and it catches the exact bug that occurred.

**What a real checker would need**, if someone wants one later: the executor's
wait-set per task type has to be declared data rather than expressed as control
flow. That is a refactor of `DagExecutor.run` — turning each arm's predecessor
lookup into a table the arm consumes — and it would make the happens-before
relation of a schedule machine-checkable against the DAG. Worth doing if the set
of comm kinds grows (CP would add at least one, F21), not before.

---

## 2026-09-20 — F23: a TP region with two outputs is silently half-reduced; split backward is fine

Two combinations `shard_tensor` had never been run against, both checked on CPU.

**Split backward composes correctly.** Zero-bubble and DualPipeV schedules split
`BWD` into `BWD_I` and `BWD_W`, and that happens *before* `shard_tensor` in
`apply_schedule_directives`, so the TP pass sees the split nodes. It does the
right thing: 4 `TP_COMM` for 2 microbatches, the backward ones anchored on
`BWD_I` (which carries the activation gradient) and none on `BWD_W` (weight
gradients only, shard-local under TP, no collective needed). Locked by a test;
no change required.

**A multi-output TP region is a real hole.** Traced a region emitting two tensors
— a row-parallel partial sum and a replicated side output:

```
seg1 tag={'PP': 0, 'TP': 0}  n_outputs=2  boundary_tensor_idx=0
```

`a2a_boundary_after` records **one** index, chosen by
`_select_boundary_tensor_idx`'s heuristic (floating point +2, requires_grad +1,
highest score wins). So exactly one of the two gets all-reduced, and which one
depends on the score. Either the replicated tensor is summed across ranks or the
partial sum is never reduced — wrong numerics, no diagnostic, and the choice can
change when the model does.

Refused rather than guessed, following F20's precedent: `shard_tensor` now rejects
a matched region whose boundary carries more than one tensor, and says to split
the region so the tensor needing the collective leaves alone.

**This is the same root as F21.** The boundary abstraction assumes *one
interesting tensor per region boundary*. That held for EP (the token tensor) and
for TP as I built it (the region output). It fails for a TP region with a side
output, and it fails for CP, which must move K and V every ring step. Three
directives now depend on a single-tensor assumption that nothing states.

**Open question.** Making `tensor_idx` a list is mechanical in `fx.py`, but every
insertion pass reads it as a scalar and each has a different notion of what to do
with several — EP would all-to-all each, TP would need to know *which* are partial
sums, CP would pair them. So the list is not the hard part; the hard part is that
"which tensors need a collective" is model knowledge the IR cannot represent
(F1 again).

---

## 2026-09-20 — F24: `devices` is symbolic; topology-aware placement cannot be expressed

**Tested.** A schedule naming GPUs 41 and 99 on a 7-GPU box compiles cleanly:

```
place  filter={"PP": 0}  devices=[41, 99]
-> schedule: pp_degree=1 dp_degree=2   rank 0 devices=[(41, 99)]   2 comm nodes
```

The runtime binds with `cuda:{global_rank % torch.cuda.device_count()}`
(`actor.py:285`) and nothing anywhere validates a device id against the machine —
`len(devices)` is the only part ever read (`schedule.py:59`).

So the integers in `devices` carry exactly two pieces of meaning:

1. **how many** there are → `dp_degree`, i.e. the size of the SPMD group;
2. **whether two `place` directives name the same set** → `pp_degree`, and hence
   where `_insert_send_recv_comm_nodes` cuts.

The values themselves never reach a GPU. Actual placement comes from
`CUDA_VISIBLE_DEVICES` and Ray's placement group. The README describes the field
as "Non-empty list of CUDA device IDs for the placement group", which reads as
physical and is not.

**Why this matters rather than being a doc nit.** F11 measured that interconnect
topology dominates TP communication: SMs can be held exclusively but
NVLink/NVSwitch is machine-wide, and on this host GPUs 0–3 and 4–6 sit on
different NUMA nodes. "Put this TP group on cards that share a NUMA node" is
exactly the kind of decision a placement language should express, and it is the
one decision with a measured effect on TP. Piper's placement language looks like
it expresses it and does not.

**Two honest fixes, and they point opposite ways.** Either make `devices`
physical — bind `cuda:{devices[dp_rank]}` instead of `global_rank % count`, which
gives the language real placement power — or rename the field to something
symbolic (`group`, `mesh`) and document that topology is the launcher's job. The
first is a small change with a real behavioural consequence; the second is honest
about what the IR currently models, which is a device *group*, never a device
(F1's shape again).

Not changed here: this is upstream's design call, not a bug to patch under a TP
branch.

---

## 2026-09-20 — F25: TP composes with ZeRO-3 on separate regions, and the dp/TP overload is an asset there

**First, a bug I introduced.** Pushing the loss function once (S1, F17) caches it
on `piper_metadata.installed_loss_fn` and skips the push while the cached object
is identical. `piper_setup` never cleared it, so a second setup in the same
process builds new actors that never receive it and the loss node gets `None`.
Not hit in practice — the harness runs each configuration in a fresh process —
but it is a state leak of my own making. The per-run reset is now
`_reset_run_state()`, with a test.

**TP + ZeRO-3 on different regions works.** `replicate(shard_params=True)` on
`PP=0` and `shard_tensor` on a `TP`-tagged region under `PP=1`:

```
ALL_GATHER_COMM 2   REDUCE_SCATTER_COMM 1   (PP=0, the ZeRO region)
TP_COMM 2                                    (PP=1/TP=0, the TP region)
zero metadata only on s0.seg0 / s0.seg0.bwd
```

No cross-contamination: the ZeRO lifetime flags land only on the ZeRO region's
nodes, and `_derive_dag_bucket_modes` classifies buckets by the presence of
AG/RS nodes, so the TP bucket is correctly not zero-managed.

**The interesting part is why this is meaningful rather than merely non-broken.**
ZeRO shards parameters across `dp_degree` members, and `dp_degree` here *is* the
TP group size — the overload F4 recorded as a limitation. But for a region that
is **replicated** across the TP group, every member holds the same logical
parameters, so sharding them across that group is exactly what ZeRO is for. The
composition is Megatron's distributed optimizer applied along the TP axis, and
Piper gets it for free precisely because it does not distinguish the axes.

So the same overload that blocks TP x DP (F4, F14) *enables* TP + ZeRO. Worth
recording because I had been treating it purely as a limitation.

**Caveat, untested.** This is a CPU check of the lowered DAG only. Whether the
ZeRO all-gather on `dp_group` and the TP all-reduce on `ep_group` interleave
correctly at runtime — they are separate communicators over the same ranks, which
is deliberate (`_join_dp_process_group` makes two so the op types do not share a
proxy stream) — has not been run. It needs two idle GPUs and belongs with the
other pending measurements.

---

## 2026-09-20 — F26: with a pipeline bubble to fill, `order` finally pays — 1F1B beats GPipe by 16% under TP

**Tested.** TP x PP=2 on four GPUs (`0,1,3,5`, claimed by a waiter that polls for
idle cards and preempts nobody), `dim 4096, hidden 16384, batch 2048/microbatch,
stages 2, tp 2, 4 microbatches, bf16, 12 iterations`, three interleaved
repetitions. Same base schedule, same collectives, **only the `order` directive
differs**.

| | rep1 | rep2 | rep3 | min |
|---|---|---|---|---|
| 1F1B (harness-generated) | 12.34 | 12.20 | 11.60 | **11.60 ms** |
| GPipe (all forwards, then all backwards) | 13.85 | 14.95 | 14.42 | **13.85 ms** |

**1F1B is 16% faster**, 3/3 repetitions, and the two ranges do not overlap
(11.60–12.34 against 13.85–14.95) — the signal is larger than this host's noise
even though the claimed cards were merely idle rather than exclusive.

### This closes F13's question, and confirms its reasoning

F13 found `order` bought *nothing* on a single stage — a hand-written 1F1B was
3.7% **slower** than the default ordering, because with one stage there is no
pipeline bubble to fill and the directive only constrains an ordering that was
already good. The inference was that `order` should pay where it pays for
anything: across pipeline stages. With a bubble present, it does.

So the two results are one statement: **`order`'s value is proportional to the
bubble it can fill, and TP alone creates none.** A TP-only schedule gets nothing
from Piper's scheduling language beyond what the default topological order
already does (F12, F13); a TP x PP schedule gets 16%.

### Why it matters for the project's thesis

This is the first measurement where TP and Piper's central claim — programmable
scheduling — produce value *together* rather than TP merely riding along. Every
prior performance result was negative or retracted: `stream` is inert at one
microbatch (F12), `order` hurts on one stage (F13), the cost model ranked
backwards (F16, retracted in F17), more microbatches hurt at constant work (F18),
CUDA graphs buy 2% (F19). This one is positive, reproducible, and attributable to
exactly one directive.

**Caveat.** The comparison is 1F1B against GPipe, both hand-or-harness-generated,
not against an optimum. It says the ordering choice is worth 16% here, not that
1F1B is the best available ordering. `zerobubble` and `dualpipev` are also
generatable and untested under TP — and F23 confirmed the split-backward path
those need does compose with `shard_tensor` correctly, so they are runnable.

---

## 2026-09-20 — F27: zerobubble does not beat 1F1B at this size

**Tested.** Same four-GPU window as F26 (cards `0,1,5,6`), TP x PP=2, mb=4,
three interleaved repetitions.

| | rep1 | rep2 | rep3 | min |
|---|---|---|---|---|
| 1F1B | 13.49 | 13.27 | 15.73 | **13.27 ms** |
| zerobubble | 13.40 | 37.58 | 14.70 | **13.40 ms** |

**No difference** — 1% apart at the minimum, well inside this host's noise. The
37.58 ms sample is contention, not the schedule.

Expected, on reflection: zerobubble's gain comes from deferring `BWD_W` to fill
bubble that 1F1B leaves, and at `pp=2, mb=4` there is little such bubble. It also
costs more nodes (53 per rank against 1F1B's 45, the extra being `BWD_W`), so at
this size the split-backward machinery is pure overhead.

So F26's 16% is specifically **1F1B over GPipe**, not "any better order wins".
The ordering choice that matters here is interleaving forward and backward at
all; refining *how* they interleave does nothing yet. Note also that F26's 1F1B
was 11.60 ms and this window's was 13.27 ms — **timings are only comparable
within one waiter window**, never across.

---

## 2026-09-20 — F28: my refactor did not change EP's lowering, verified byte-for-byte

Lifting `_boundary_info_for_edge` out of `_insert_shard_a2a_comm_nodes` (so the
TP pass could reuse the backward-producer rule instead of re-deriving it) touched
the only code path EP's collectives go through. EP has **no GPU test upstream**,
and the shipped Qwen MoE example had never been run in this project, so the
refactor was unverified against the thing it could break.

`experiments/compare_ep_lowering.py` lowers the shipped Qwen MoE example
(`create_qwen3_config("9M")`, `pp2_dp2_ep2.json` plus a 2-microbatch split)
through the full `apply_schedule_directives` pipeline and prints the DAG's shape.
Run in a `git worktree` of `upstream/main` and in this branch, the outputs are
**identical**: same segment count, node and edge counts, node-kind histogram, and
all 16 `A2A_COMM` nodes with the same direction, `tensor_idx` and anchor.

It uses only APIs that exist on `upstream/main`, which is what lets the same
script run in both checkouts — worth keeping for any future change to the shared
boundary machinery.

**What this does not cover.** It is the lowered DAG, not execution. EP has never
been run on GPUs here, so if `A2A_COMM` were mis-executed the comparison would
not see it — but that code I did not touch. The refactor's blast radius is the
lowering, and the lowering is unchanged.

---

## 2026-09-20 — F29: a single-device `shard_tensor` was accepted and would crash in the executor

`shard_tensor` with `devices: [0]` — the obvious thing to write when debugging on
one GPU — was accepted and inserted two `TP_COMM` nodes. At runtime a one-device
group leaves `dp_degree` and `pp_degree` both at 1, so
`actor.py:_join_process_groups` skips `init_process_group` entirely and
`runtime.ep_group` stays `None`. `all_reduce_activation` would then call
`dist.all_reduce` on an uninitialized default group and fail deep inside the
dispatch loop, with a `torch.distributed` message that says nothing about the
schedule.

Rejected at compile time now, pointing at the fix (remove the directive to run
the region unsharded). Repeated ids (`[0, 0]`) are rejected too, since that is
two entries but one device.

**The same hole exists upstream for `shard`.** `shard` with `devices: [0]` would
insert `A2A_COMM` nodes and hit the identical `ep_group is None` path. Not patched
here — it is upstream's directive and outside what a TP branch should touch — but
it is the same one-line check and worth mentioning alongside the `shard_tensor`
work.

This is the fourth silent-failure-made-loud in this branch (F14, F20, F23, F29),
all the same shape: a precondition the code depends on, checked nowhere, failing
somewhere that does not name it.

---

## 2026-09-20 — F30: both shipped examples still run, on real inputs for the first time

**Tested.** Four GPUs (`0,1,3,4`), `--schedule 1f1b --ranks 2 --mbs 4`, via a
waiter that polls for idle cards.

| example | configuration | result |
|---|---|---|
| `test_qwen.py` + `pp2_dp2_ep2.json` | MoE, EP, ZeRO, PP x DP | **OK**, 54.99 ms/iter |
| `test_llama.py` + `llama_pp2_dp2.json` | dense, PP x DP | **OK**, 37.13 ms/iter |

This closes the regression question this branch has been carrying. The changes
that could plausibly have broken them:

- **the zero-input fix (F9)** changed *every* example's behaviour — until now they
  consumed zeros, and nothing here had run since. If anything downstream depended
  on zero activations, or if real data produced overflow that zeros hid, this is
  where it would surface. Neither happened.
- **the loss plumbing (F8)** added an append per loss-computing backward and
  changed `_update`'s return shape on both paths.
- **the `_boundary_info_for_edge` extraction** touched EP's only code path;
  F28 already showed the lowering is byte-identical, and this shows it executes.
- **four new compile-time rejections** (F20, F23, F29, and the `replicate`
  refusal from F6) could have rejected a legitimate shipped schedule. They did
  not.

The LLaMA schedule is new — `test_llama.py` defaults to `pp2.json`, which does
not exist in the repository — so `llama_pp2_dp2.json` is added: PP x DP with
`replicate` per stage and no EP tag, which is what a dense model needs.

**Caveat.** "Runs" is what is verified, not "computes the right thing". There is
no numerical reference for either example, and building one would need the
parameter-override path (F10) extended to their model constructors. For TP that
comparison exists (F10, F15); for the shipped examples it does not, upstream or
here.

---

## 2026-09-11 — F31: TP scales 2→4 (30% faster), and the latency/bandwidth crossover is real

**Tested.** Four cards at 0% utilization (`0,3,4,6`, other users' processes
resident but idle), `dim 4096, hidden 16384, global batch 8192, stages 1, mb 1,
bf16`, three interleaved repetitions, minimum. Constant global batch and constant
total hidden: TP only changes how much of the MLP each card owns, and the
all-reduce payload is `[batch, dim]` = 67 MB either way.

| tp | min iter | compute | comm (min) | collectives/iter |
|---|---|---|---|---|
| 2 | 6.92 ms | 4247 us | 545 us | 4 |
| 4 | **4.85 ms** | 2957 us | 920 us | 4 |
| ratio | **1.43x** | 1.44x | 1.69x | — |

### Compute matches the roofline

`pre` and `post` are replicated and `up`/`down` are sharded, so the predicted
ratio is `(2·4096² + 2·4096·8192) / (2·4096² + 2·4096·4096)` = **1.50x**, against
**1.44x** measured. The small shortfall is the smaller per-card kernels. The
compute half of F16's model, which survived every retraction, holds again here.

### Communication is bandwidth-bound at this payload, which answers F19

Ring all-reduce moves `2(N-1)/N · S`: **1.0S** at N=2 and **1.5S** at N=4, so the
predicted growth is **1.5x** against **1.69x** measured. Close — growth tracks the
*bandwidth* term, not a per-collective latency that would scale with the number of
ring steps.

F19 left this open: it measured a 105 us collective against an 11.7 us bandwidth
bound — **9x** — on a 4.2 MB payload, and asked whether the gap closes at larger
payloads. It does. At 67 MB the collective sits at 545 us against a 186 us bound,
**2.9x**, and its growth with group size follows the ring model. So the two
measurements are one picture:

| payload | measured / bandwidth bound | regime |
|---|---|---|
| 4.2 MB (F19) | 9.0x | latency-bound |
| 16.8 MB (F18, mb=4) | ~3.6x | crossing |
| 67 MB (here, F18 mb=1) | 2.9x | bandwidth-bound |

**This is the quantitative condition F19 asked for.** Collective fusion — merging
several microbatches' all-reduces into one — only pays in the latency-bound
regime, i.e. when per-collective payload is small enough. At 67 MB there is
nothing to win; at 4.2 MB there is 9x of overhead that fusion would attack. And
F18 already showed the practical route to the bandwidth-bound regime: use fewer
microbatches.

### TP is worth scaling here

TP=4 is **30% faster** than TP=2 end to end. Compute saves 1290 us while
communication costs 375 us more, so the collective is not eating the gain — at
this size TP scales close to how the compute does.

Second positive performance result in the project, after F26. Both came from
running on cards that were actually idle and reducing three interleaved
repetitions by minimum; every earlier attempt that skipped either step produced
a number I later had to retract (F16 → F17).

---

## 2026-09-11 — F32: the cost of a TP all-reduce barely depends on its payload. **F31's "bandwidth-bound" reading is wrong**

**Tested.** TP=2 on two idle cards (`0,3`), `dim 4096, hidden 16384, stages 1,
mb 1, bf16`, batch swept 256→8192 so the per-collective payload sweeps 2→64 MB.
Three interleaved repetitions, minimum. Bound computed at the 360 GB/s measured
in F11.

| batch | payload | bound/coll | **measured/coll** | ratio |
|---|---|---|---|---|
| 256 | 2.0 MB | 5.8 us | **138.6 us** | 23.8x |
| 512 | 4.0 MB | 11.7 | **313.1** | 26.9x |
| 1024 | 8.0 MB | 23.3 | **227.1** | 9.7x |
| 2048 | 16.0 MB | 46.6 | **223.1** | 4.8x |
| 4096 | 32.0 MB | 93.2 | **284.5** | 3.1x |
| 8192 | 64.0 MB | 186.4 | **322.0** | 1.7x |

### Retraction

**F31 concluded that a 67 MB collective is bandwidth-bound.** Read the measured
column: a **32x** increase in payload produces a **2.3x** increase in time
(138.6 → 322.0 us, non-monotone in between). The cost is close to a **constant
~220–320 us regardless of payload** over the whole range tested. The ratio falling
from 23.8x to 1.7x is the *denominator* growing while the numerator barely moves —
not the collective entering a bandwidth-bound regime.

I read a falling ratio as a change of regime. It is arithmetic. The regime never
changed: it is fixed-cost-dominated everywhere from 2 MB to 64 MB, and 64 MB is
merely where the bandwidth bound finally catches up with the fixed cost.

F31's other results stand — compute scaling at 1.44x against a predicted 1.50x,
and TP=4 being 30% faster than TP=2. Only the bandwidth interpretation is
withdrawn. (Fourth self-correction: F16→F17, F17→F19, F18→F19, F31→here.)

### What the fixed cost is not, and what follows

Not dispatch: F19 captured the whole step in a CUDA graph and gained 2%. Not
bandwidth: this sweep. What is left is NCCL's own per-collective cost — kernel
launch, the ring's per-step synchronization, and the two ranks' arrival
difference at each collective, which F11 measured directly at the step level and
which recurs at every collective.

**This makes collective fusion worth building, where F19 judged it premature.**
If cost is ~constant per collective, merging K collectives into one saves
`(K-1) x ~250 us` at *any* payload, not only in a latency-bound corner. Against
F18's numbers that is large: at mb=4 with 2 stages there are 16 collectives per
step, so fusing across microbatches could remove on the order of 3 ms from a
~16 ms step.

The trade-off F18 and F12 identified is unchanged and now quantifiable: fusing
across microbatches forces the later microbatches' compute to wait, so it trades
overlap (worth 1.06x concurrency, F12) against fixed cost (worth ~250 us per
collective removed). Those are now both measured, so the comparison can be made
before writing the pass rather than after.

**Caveat.** The sweep is noisy — 313 us at 4 MB against 227 us at 8 MB is
non-monotone, and this host is shared. The robust claim is the one that survives
the noise: **time does not scale with payload over a 32x range.** A tighter
constant would need an idle machine.

---

## 2026-09-11 — F33: fusion helps, but F32's mechanism was wrong — the fixed cost is *synchronization between* collectives, not NCCL per-call cost

**Tested.** `experiments/probe_collective_fusion.py`, `torchrun
--nproc_per_node=2` on two idle cards. Outside Piper on purpose (no Ray, no DAG,
no dispatch loop), same method as F19's CUDA-graph probe.

| K | payload/coll | separate | fused | fused+copy | speedup | bound |
|---|---|---|---|---|---|---|
| 4 | 16 MB | 228 us | 162 | 207 | 1.10x | 186 |
| 8 | 16 MB | 450 | 292 | 387 | 1.16x | 373 |
| 16 | 8 MB | 713 | 291 | 406 | 1.75x | 373 |
| 8 | 4 MB | 312 | 95 | 128 | 2.43x | 93 |

### The number that matters is in the `separate` column

F32 measured an *isolated* TP all-reduce in Piper's DAG at **220–320 us**
regardless of payload, and inferred a per-collective fixed cost. But four
back-to-back all-reduces of 16 MB here total **228 us — 57 us each**, four to five
times cheaper than the same collective measured inside the DAG.

**So the fixed cost is not NCCL's.** Back-to-back collectives do not pay it. What
differs inside Piper is that the collectives have *compute between them*, and the
two ranks' compute does not finish in lockstep — so every collective re-pays an
arrival difference, which is exactly what F11 measured at step granularity and
what F16 wrongly generalized into a per-step constant. It is per-*collective*,
and it exists because of what sits between them.

Fifth correction to this model: F16→F17, F17→F19, F18→F19, F31→F32, F32→here.

### Fusion helps, less than F32 predicted, and the probe is a lower bound

F32 predicted `(K-1) x ~250 us`. Measured is **1.10x–2.43x**, growing with K and
with *smaller* payloads — the shape a fixed-cost model predicts, but with a much
smaller constant, because this probe removed the very thing that creates the cost.

That cuts both ways: **the probe underestimates fusion's value in Piper.** In the
real DAG each fused collective also removes a re-synchronization point, which the
probe cannot show. The honest statement is that fusion is worth between 1.1x on
the collectives alone and something larger in situ, and only a pass in Piper
would measure the difference.

Two practical numbers for whoever writes it:

- **the copies are not free**: `cat` + scatter-back costs 45 us at K=4 and 115 us
  at K=16, i.e. 20–30% of the fused collective. A real pass should reduce into a
  pre-allocated flat buffer that the compute writes into directly, not cat after
  the fact.
- **360 GB/s (F11) is an underestimate**: fused times come in *below* the bound
  computed from it (162 us against 186 us at K=4), so F11's figure was depressed
  by contention. Any bandwidth-derived bound in this log is conservative.

### And fusion conflicts with `order`

Fusing microbatches' collectives forces all of them to reach the region before
any collective runs. That is compatible with GPipe and **incompatible with
1F1B**, which interleaves microbatch *i*'s backward with *i+1*'s forward — and
F26 measured 1F1B beating GPipe by 16%. So fusion's real competitor is not "no
fusion", it is 1F1B, and the comparison to run is `GPipe + fusion` against
`1F1B`, not `fusion` against `no fusion`.

**Next experiment.** That comparison, which needs the pass. With F26's 13.85 ms
for GPipe and 11.60 ms for 1F1B, fusion has to find ~2.3 ms in GPipe's 16
collectives to break even — plausible at 1.1x–2.4x on ~4 ms of collectives, but
not obviously so. Worth writing the pass to find out; not worth assuming.

---

## 2026-09-11 — F34: `fuse_collectives` works, is worth 12–19% where communication matters, and does *not* cost overlap

**Design, as specified:** an independent directive, and **`order` wins**.

```json
{ "op": "fuse_collectives", "filter": {"TP": "*"} }
```

**Implementation note.** The nodes are *not* merged into one. Each microbatch's
`TP_COMM` keeps its own successors, because they consume different microbatches'
activations; merging them would hand every successor the same buffer. Instead one
member of each group is the **leader** that issues the combined collective, and
the others read their slice of the result. The DAG keeps its shape; only the
runtime behaviour changes.

**`order` priority** is enforced by running the pass *after* `order` and refusing
any group whose fusion edges would contradict an existing path. 1F1B interleaves
microbatch *i*'s backward with *i+2*'s forward, which contradicts making every
member's producer complete before any member runs — so under 1F1B **nothing
fuses**, verified by test rather than asserted.

### Correctness

Fused and unfused produce **bit-identical losses** across three optimizer steps,
both ranks: `5.467978 / 2.951428 / 1.893221`.

### Gain, and where it comes from

| config | comm share | events/iter | comm (min) | iter (min) | iter (median) |
|---|---|---|---|---|---|
| mb=4, batch 2048 (16 MB/coll) | ~14% | 16 → 4 | 1125 → 874 us (1.29x) | 8.06 → 7.53 | — |
| mb=8, batch 1024 (8 MB/coll) | ~75% | 32 → 4 | **790 → 336 us (2.35x)** | **13.93 → 12.28** | **19.95 → 16.15** |

At mb=8, seven interleaved repetitions each: **12% faster at the minimum, 19% at
the median**, with communication 2.35x cheaper. At mb=4 the collective gain is
real but the step gain is inside the noise, because communication is only ~14% of
the step there. Both match F33's prediction that fusion pays more as
per-collective payload shrinks.

### My overlap hypothesis was wrong

I expected fusion to cost overlap: it forces every microbatch's producer to
finish before any collective runs, which should undo F12's microbatch-level
hiding. Measured concurrency (kernel-sum / wall-span) says otherwise — **0.47x
unfused against 0.53x fused**, i.e. slightly *better*.

Both are well below 1.0, and that is the explanation: at batch 1024 the GPU
timeline is ~50% idle (F12 found the same at this size, and attributed it to
per-node dispatch). There was no overlap to lose, because communication was not
being hidden behind compute in the first place — it was waiting, as F33's
arrival-difference mechanism says. Removing synchronization points helps and
costs nothing here.

**The hypothesis is not refuted in general**, only at this size. Where the GPU is
saturated and F12's 1.06x concurrency is real (batch 8192, mb=4), fusion should
trade against genuine overlap, and that configuration has ~14% communication
share where the gain is smallest. So the honest scope is: **fusion pays when
communication share is high, which is exactly when the GPU is least saturated and
overlap is least available.** The two effects do not compete as directly as I
assumed.

Sixth correction in this log (F16→F17, F17→F19, F18→F19, F31→F32, F32→F33, and
this one).

### Still open

`GPipe + fusion` against `1F1B`, which F33 named as the real comparison, needs
four GPUs and the pipeline schedules; fusion cannot combine with 1F1B by
construction, so that is the choice a user actually faces.

---

## 2026-09-11 — F35: `GPipe + fusion` and `1F1B` are indistinguishable. Fusion helps GPipe; it does not decide the schedule

**The question**, from F33: fusion cannot combine with 1F1B by construction
(`order` wins, F34), so a user picks between them. Does `GPipe + fusion` beat
`1F1B`?

**Criterion declared before the data.** Nine interleaved repetitions per arm, TP x
PP=2 on four idle cards, `dim 4096, hidden 16384, batch 2048/microbatch, stages 2,
mb 4, bf16`. If the ranges overlap, the verdict is *indistinguishable* — not
"pick the better minimum".

| arm | n | min | p25 | median | max |
|---|---|---|---|---|---|
| 1F1B | 9 | 10.46 | 13.15 | 13.75 | 14.82 |
| GPipe | 9 | 13.07 | 13.34 | 13.76 | 15.06 |
| GPipe + fusion | 9 | 11.25 | **12.82** | **13.07** | 14.57 |

**Verdict: indistinguishable.** `1F1B` spans 10.46–14.82 and `GPipe + fusion`
spans 11.25–14.57; the ranges overlap almost entirely.

**What *is* visible:** fusion improves GPipe on all three order statistics —
min 13.07→11.25, p25 13.34→12.82, median 13.76→13.07, about 5%. That is
consistent with F34's mechanism and with fusion's 2.35x on the collectives
themselves; it is simply not enough to separate the schedules.

### The process matters more than the number here

The first three repetitions gave **GPipe faster than 1F1B**, the opposite of
F26's 16% the other way. Stopping there and reporting it would have been
reporting the window, not the effect. Running to nine and declaring the criterion
first is what turns that into a real answer, and the answer is a tie.

F26's result stands in its own window — there the ranges genuinely did not
overlap (1F1B 11.60–12.34 against GPipe 13.85–14.95). Both are true: the
schedules differ by more than noise on a quiet machine and by less than noise on
a busy one, which is F27's "timings compare only within one window" stated
sharply.

### What this settles

- **Fusion is worth having** — 2.35x on collectives (F34), 12–19% end to end
  where communication dominates (F34), ~5% on GPipe here.
- **Fusion is not a reason to abandon 1F1B.** Its cost is being locked out of
  interleaved schedules, and it does not buy back enough to make that worthwhile
  at this size.
- **The useful configuration is therefore `1F1B` without fusion, or fusion
  wherever `order` is not filling a bubble** — i.e. single-stage TP, where F34
  measured the 12–19%, and where F13 already showed `order` buys nothing.

That is a coherent recommendation and it fell out of two directives interacting,
which is the thing Piper's scheduling language is for.

---

## 2026-09-11 — F36: "three cards with no processes" is not a quiet machine, and `median/min` on communication says how contaminated a window is

**Tested.** Repeated F34's fusion measurement on GPUs 1 and 2 at a moment when
they, and GPU 6, held **no processes at all** — the cleanest the host had been in
this project. Nine interleaved repetitions per arm.

| arm | n | min | p25 | median | max | comm-min | comm-med |
|---|---|---|---|---|---|---|---|
| unfused | 9 | 14.60 | 18.54 | 21.69 | 25.13 | 4750 us | 7379 us |
| fused | 9 | 13.40 | 17.44 | 19.22 | 20.90 | **1654** | **7351** |

**This window was worse than F34's**, not better: unfused median 21.69 ms against
F34's 19.95 ms. During the run the other four cards all went to ~100%.

### The diagnostic

Look at communication. The **minimum** separates the arms by 2.87x, consistent
with F34's 2.35x. The **median** does not separate them at all — 7379 against
7351 us, a 0.4% difference between a schedule with 32 collectives and one with 4.
A median that cannot tell those apart is not measuring the schedule.

That gives a cheap, per-window contamination check: **`comm median / comm min`**.

| arm | median/min | reading |
|---|---|---|
| unfused | 1.55 | tolerable |
| fused | **4.46** | the window is dominated by other traffic |

When that ratio is large, only the minimum carries signal, and any statistic that
averages is reporting the machine's other users. This is F11's rule ("report the
minimum") turned into something checkable *before* trusting a number, rather than
a principle to remember.

### Why owning the SMs did not help

F11 established it and this is the concrete instance: SMs are per-GPU and were
exclusively mine; NVLink/NVSwitch is machine-wide and four other cards were
saturating it. For a TP measurement the interconnect is the contended resource,
so "no processes on my cards" is the wrong thing to wait for. The waiter in
`experiments/` polls exactly that wrong condition — it is the best signal
available from `nvidia-smi` without inspecting other users' traffic, and its
limitation is now recorded rather than implied.

**F34's numbers stand**, measured in a better window. Substituting this window's
would be choosing the data that suits the conclusion; both are in the log.

## 2026-09-13 — F37: a collective's issue point is an accident of topological level; ZeRO-3 gathers every layer before the first forward runs, and `cp-design.md` F-c is retracted

### Tested

`notes/cp-design.md` §7 P4 asked whether the ordering pass pins a
dependency-free collective next to its consumer. The cheapest discriminator is
ZeRO-3's `ALL_GATHER_COMM`, which `directives.py:867-895` creates with no
incoming edge. Three stages, one device group, `replicate(shard_params=true)`,
mb=1 and mb=2 (`examples/base-schedules/zero3_s3_mb{1,2}.json`), lowered on CPU
and printed with `experiments/dump_dag.py`. Edge lists confirmed directly.

### Unexpected

The prediction was wrong in both of its parts.

1. The anchor logic does not apply. `ALL_GATHER_COMM` is on `default_stream`;
   `reduce_stream` governs only the reduce-scatter. `resolve_total_order_per_stream`
   never sees it.
2. It is not issued late. It is issued **first**. All nine forward all-gathers
   occupy dispatch slots 0–8, before any compute node. `_serial_topological_order`
   sorts by `(topo_level, priority, topo_idx)`; a root has level 0. That is the
   entire mechanism — there is no policy, the position falls out of the sort key.

The backward all-gathers are the same story one step later: each has exactly one
predecessor, a *temporal* edge from its layer's forward compute (free-after-forward,
re-gather-any-time-after), so `all_gather.9` for `s0.seg0.bwd` is issued at slot 10
and consumed at slot 42.

### Consequence for the shipped system

Full parameters are released by `defer_free_full_params` *after* the consuming
compute (`executors.py:837-1120`). By the dispatch order, when `s0.seg0` runs every
layer's full parameters have already been gathered and none has been freed.
**Peak parameter memory under this schedule is the unsharded total** — the quantity
ZeRO-3 exists to avoid. Correctness is unaffected; the memory saving is. This is
stated from the dispatch order and the free point; a GPU measurement of
`max_memory_allocated` against `sum(full params)` is queued behind the same
load-gate as F36's experiments and will be recorded when the machine allows.

With mb=2 the artifact is visible in another form: `all_gather.26` (MB1, last stage)
is dispatched at slot 3, before MB0's first forward.

### IR-runtime assumption discovered

The design document claimed two mechanisms conspire to serialize a ring exchange:
edge-splicing at insertion (F-a) and a late-pinning anchor at ordering (F-c).
F-a stands (`directives.py:1143,1168`). F-c is **retracted** — the anchor rule is
real but irrelevant to this collective, and the actual behaviour is the opposite
sign.

What replaces it is narrower and, I think, more useful:

> Piper has no notion of *where* a collective is issued. Position is a function of
> graph structure only. An edge-spliced collective is issued when its source region
> completes (too late for CP, whose payload was ready at region start); a root
> collective is issued at topological level 0 (too early for ZeRO-3, whose memory
> budget wanted it one layer ahead). Both are the same missing concept — an
> earliest-ready point and an issue budget between it and the consumer — seen from
> opposite ends.

That reframing makes the `distance` argument in cp-design §4 G2 the *missing*
knob rather than an optimization: today the only two available values are 0 (for
spliced collectives) and ∞ (for roots).

### Retraction discipline

F-c was written from reading `ordering.py:115-159` and `directives.py:867-895`
together and inferring the interaction without lowering a DAG. The inference was
plausible and wrong. It cost one CPU probe to test; the rule from F16/F17 applies
to reading code as much as to reading timings: a mechanism inferred from two
sites is a hypothesis until the lowered artifact is looked at.

### Open question

Whether the executor *actually* holds all gathered buffers concurrently, or whether
`ParamStore` aliases them in a way the dispatch order does not show. The dispatch
order says yes; the measurement decides.

### Next experiment

`test/test_zero3_dispatch_order.py` pins the three facts above as a
characterization test so the eventual issue-policy change has a red test to turn
green. Then G-1 (multi-tensor boundaries) proceeds unchanged — F37 strengthens its
motivation rather than altering its shape.

## 2026-09-13 — F38: `replicate` emits a gradient all-reduce for regions with no parameters; the runtime silently discards it

### Tested

Composing `replicate` (projection weights) with `ring_exchange` (CP regions) on the
ring-attention model, lowered on CPU. Expected `REDUCE_COMM` on the prologue and
epilogue backward nodes only, since the CP regions are pure tensor arithmetic on
q/k/v/o/m/l and carry `param_idxs == []`.

### Unexpected

Every CP region's backward node got a `REDUCE_COMM` too (`reduce.1`, `reduce.2`).
`_insert_reduce_comm_nodes` (`directives.py:771-800`) filters on backward-weight
subkind, tag match and device, and never asks whether the region has trainable
parameters. `_insert_all_gather_comm_nodes` in the same file does
(`_node_has_trainable_params`, `directives.py:850`). The two passes disagree about
the same question.

### Consequence

Not a wrong answer and not a stray NCCL call: `all_reduce_grads`
(`executors.py:164-167`) begins with
`if not self.has_trainable_params_for_collective(...): return 0`. The node is
dispatched, waits on its predecessor's event, records its own, releases the
backward buffer, and does nothing in between. One dead node per parameter-free
region per microbatch, issued on `reduce_stream`.

The guard is in the wrong layer. The compiler emits a collective it can tell has
no payload; the runtime knows to skip it. Nothing shipped could expose this:
every region in the LLaMA, Qwen and MLP examples owns parameters. Attention-only
ring steps are the first parameter-free regions Piper has seen.

### IR-runtime assumption discovered

Adds to F19-F24's list of unstated invariants: **a comm node may be emitted
whose payload is statically empty, and correctness then depends on the executor
recognizing that**. Same family as the silent-skip failures — a pass that has
nothing to do says nothing — but in the opposite direction: a pass that has
nothing to do emits a node anyway.

### Evidence pinned

`test/test_ring_directive.py::test_replicate_attaches_reductions_to_parameter_free_regions`
asserts the current count (three dead reductions for three ring steps). It is a
characterization: the two-line guard makes it fail, which is when it gets
rewritten. Not fixed now so the queued G-2 run exercises upstream behaviour;
recorded so the fix has evidence to cite.

### Next experiment

Queued (`experiments/run_cp_gate.sh`, chained behind the F37 memory job on the
same card gate): the out-of-band CP gate under torchrun, the in-Piper CP=2 vs
dense CP=1 equivalence across optimizer steps with a no-ring negative control,
and P3 on four cards when four are free.

## 2026-09-13 — F39: one temporal edge gives ZeRO-3 its prefetch budget; the knob derived for CP needed no new mechanism

### Tested

`replicate(prefetch_distance=1)` on the three-stage ZeRO-3 schedule from F37,
lowered on CPU. `_bound_all_gather_issue` adds, for each `ALL_GATHER_COMM`, a
temporal edge from the compute node `distance` steps before its consumer along
data edges (walking through comm nodes).

### Result

Dispatch order goes from F37's

    AG_0 AG_1 ... AG_8 | compute_0 compute_1 ...

to

    AG_0 compute_0 AG_1 compute_1 AG_2 compute_2 ...

Backward gathers, which F37 showed issued during the forward pass ~30 slots
ahead of their consumer, now sit 1–3 slots ahead. The first layer keeps no edge
and starts the pipeline. Default behaviour is untouched and still pinned by
`test_zero3_dispatch_order.py`; the new behaviour by
`test_zero3_prefetch_budget.py`. 74 CPU tests.

### Why this is the design's central claim, and what it does not yet show

The `distance` argument was derived in `cp-design.md` for the ring exchange,
where without it a hoisted ring reproduces F37 on the feature meant to hold
`1/n` of K/V. Applied unchanged to the collective F37 was found on, it produces
the schedule every hand-written ZeRO-3 implementation uses. One mechanism —
"a collective's issue point is an edge, not an accident of topological level" —
covers both. That is the evidence the abstraction is at the right level; a
placement type system would have said nothing about either.

What the CPU cannot show: the memory. The dispatch order says peak parameter
memory drops from the unsharded total to one layer's full parameters plus the
shard; the slope experiment queued in `experiments/measure_zero3_peak.sh` now
has a third arm predicting 2.0× stage-bytes against the shipped 3.0× and plain
DP's 4.0×. Also not shown: overlap. `ALL_GATHER_COMM` sits on `default_stream`
unless `gather_stream` is set, so at `distance=1` on one stream the prefetch is
serial — it bounds memory, it does not yet hide latency. That needs a second
stream and is the same measurement as the ring's.

### Next experiment

The queued slope run. Then the GPU G-2/G-3 chain.

## 2026-09-13 — F40: ring attention matches dense attention out of band, and the backward ring is constrained

### Tested

`torchrun --nproc_per_node=2 test/test_cp_equivalence.py` on two B200s shared
with other tenants (numerics are indifferent to that). Each rank holds one
sequence chunk of Q/K/V; K/V travel the ring via an autograd Function whose
forward hands the chunk to the next rank and whose backward hands the chunk's
gradient back to the previous one. Reference: gather the chunks, run dense
`scaled_dot_product_attention`, slice this rank's rows.

### Result

    out 4.2e-07   dQ 1.2e-06   dK 2.9e-06   dV 1.9e-06     (fp32, max abs err)

Both controls broke as required: with no rotation the output differs; with a
forward-only rotation the output *matches* and dK/dV differ. The second control
is the one a loss curve cannot provide — it is the only thing in this project
that constrains the backward ring's placement, and it fails the moment the
gradient stops travelling back.

### What this does and does not establish

Establishes: the math in `examples/models/ring_attn.py::ring_step`, the
forward/backward rotation directions (+1/−1) that `ring_exchange` lowers to,
and that the accumulation of a chunk's gradient across ranks is correct when
done by the receiving rank's autograd (no explicit reduction). These are the
three things G-1b's executor arms encode.

Does not establish: that Piper's executor does the same. That is the in-Piper
half of G-2, which had not run at this point (a script bug; rerun queued).

### Next experiment

In-Piper CP=2 vs dense CP=1 across optimizer steps with the no-ring control.

## 2026-09-13 — F41: `ring_exchange` is numerically correct inside Piper across optimizer steps, spliced and hoisted alike

### Tested

`experiments/check_cp_equivalence.py --cp 2`: CP=2 (`cp2_ring_dp`: place on two
ranks, `replicate` on the projection weights, `ring_exchange` on the CP regions)
against dense CP=1 on one rank, same global seed for data and weights, three
optimizer steps, fp32, Inductor off. Then the same with `cp2_ring_dp_hoist`
(`hoist=true, distance=1`). Two shared B200s.

### Result

    CP=1 dense            [1.002299, 0.998082, 0.994083]
    CP=2 rank0            [0.998436, 0.994217, 0.990225]
    CP=2 rank1            [1.006162, 1.001945, 0.997937]
    CP=2 mean over ranks  [1.002299, 0.998081, 0.994081]   worst |diff| 1.7e-06
    no-ring control differs by 3.2e-03

The hoisted run's per-rank losses are **identical to the last printed digit** to
the spliced run's. The consumer merge, the ring node reading a ring predecessor,
the shared-event rule and `record_stream` compute the same thing as the spliced
path.

### What it constrains

Iteration 0 matching blames nothing; iterations 1–2 matching means the backward
ring delivered each chunk's gradient to its owner and the projection-weight
all-reduce composed with it — otherwise the two ranks' weights would have
diverged from step 1 (as they did for TP in F10 when the backward collective was
missing). Per-rank losses differ (different sequence halves) and their mean
equals the dense loss, which is the arithmetic of equal chunks.

With F40 this closes G-2 and the numerics half of G-3. The overlap half of G-3
needs four quiet cards and is queued.

## 2026-09-13 — F42: `prefetch_distance` changed the dispatch order and not the peak memory; the budget edge bounds issue on the stream, allocation happens on the host

### Tested

`experiments/measure_zero3_peak.sh`: TPMlp, dim 4096, hidden 16384 (stage =
pre+up+down+post = 0.625 GiB fp32), two ranks, mb=1, stages ∈ {2,4,8}, three
arms. Peak = `max_memory_allocated` over the timed iterations, max over ranks.
Slope fitted against stages × stage-bytes.

| arm | predicted | measured | intercept |
|---|---|---|---|
| plain DP | 4.0 | **4.00** | 0.03 GiB |
| ZeRO-3 as shipped | 3.0 | **3.61** | 0.72 |
| ZeRO-3, `prefetch_distance=1` | 2.0 | **3.99** | −0.78 |

Plain DP landing on 4.00 says the accounting (params + grads + 2× Adam) is
right and the measurement is clean. The `prefetch_distance` arm is the failure:
at 2 and 4 stages it is lower than shipped (4.28 vs 4.85, 9.09 vs 10.34 GiB), at
8 stages it is *higher* (19.22 vs 18.59). Shipped ZeRO-3 shows a rank asymmetry
plain DP does not (17.41 vs 18.59 GiB at 8 stages).

### Retraction

F37's consequence paragraph — "by the dispatch order and the free point, peak
parameter memory is the unsharded total" — inferred memory from dispatch order.
The dispatch order did change exactly as `test_zero3_prefetch_budget.py` says
(CPU-verified). Peak memory did not follow. The inference was wrong, and the
mechanism is in the runtime, not the DAG:

- `alloc_full_params` runs **on the host, at dispatch time** (`runtime.py:334`);
- the free is deferred to a **background thread** that `evt.synchronize()`s and
  then frees (`runtime.py:285-300`, `defer_free_full_params`);
- the host dispatch loop (`executors.py:664`) does not block on anything.

So the host allocates every layer's full-parameter buffer while the GPU is still
on layer 0, in *both* arms. The temporal edge orders the gather's stream work
after the previous compute; it cannot delay a host-side `alloc`. Peak is then
set by the race between host dispatch and GPU completion — which is why the
gain is non-monotone in depth and why ranks differ under ZeRO-3 and not under
DP (DP has no deferred frees).

This is stated as a mechanism *hypothesis*, consistent with every number above
but not yet observed directly.

### Pre-registered prediction

If the cause is allocation-at-dispatch, a host-side `synchronize()` on the
budget predecessor's event before `alloc_full_params` (an experiment knob, not
a fix — it stalls dispatch) brings the `prefetch_distance=1` slope to ≈2.0. If
the slope stays ≈4, the cause is elsewhere and the hypothesis is withdrawn. A
per-node memory trace (allocated bytes after each dispatched node) should show
the full-parameter buffers accumulating during the first layer's dispatch.

### Consequence for the design

An issue budget on the DAG bounds when a collective *executes*; bounding what
it *holds* needs the runtime to allocate in step with the budget — a bounded
pool of `distance+1` full-parameter buffers reused round-robin, with reuse
ordered by the free event on the stream rather than by a host wait. That is the
actual shape of the ZeRO-3 fix, and it was invisible from the DAG. The ring's
`distance` has the same exposure in principle: recv buffers are allocated at
dispatch too. With two ranks there is one rotation and nothing to accumulate;
the four-card run is where it would show.

## 2026-09-13 — F43: the peak is a host-ahead-of-GPU race over deferred frees; a DAG budget cannot reach it, and the pre-registered knob proved why

### Tested

`experiments/measure_zero3_mechanism.sh`, same model and arms as F42, plus two
knobs added for this experiment and off by default: `PIPER_MEM_TRACE` (allocated
bytes sampled on the host before every dispatched node) and
`PIPER_AG_HOST_SYNC=1` (block the host on the budget predecessor's event before
`alloc_full_params`). `PIPER_*` variables are now forwarded to Ray workers via
`runtime_env`; they were not reliably inherited before.

### Result 1 — the pre-registered prediction held

| arm | slope (× stage-bytes) | peak @8 stages |
|---|---|---|
| plain DP | 4.00 | 20.03 GiB |
| ZeRO-3 shipped | 3.61 | 18.59 |
| `prefetch_distance=1`, DAG edge only (F42) | 3.99 | 19.22 |
| `prefetch_distance=1` + host blocked on the budget predecessor | **2.01** | **11.63** |

F42 predicted ≈2.0 if allocation-at-dispatch was the cause and ≈4 if not. The
rank asymmetry F42 noted under ZeRO-3 (up to 1.3 GiB) is gone under host-sync:
both ranks report the same byte count at every depth. It was the race.

### Result 2 — what the trace shows, and a correction to F42's own reading

Shipped ZeRO-3, rank 0, before the first forward: twelve `all_gather` nodes
dispatched back to back, allocated bytes climbing 2.52 → 4.95 GiB in steps of
64 MB (pre/post) and 0.5 GiB (up/down). That is F37's dispatch order turned into
bytes, and it is exactly what `prefetch_distance=1` removes: the same point
reads 2.58 GiB, and the forward stays between 3.1 and 3.7 GiB throughout.

But the step's peak is not there. In *both* arms the sampled maximum is at the
last node of the backward (`reduce_scatter.0`, 8.15 GiB, identical), reached by
a monotone climb through the backward — full gradients allocated ahead of
their deferred frees (`defer_free_full_grads`, same background-thread pattern
as the parameters). The forward budget shaved 0.57 GiB off a region that was
below the peak. F42 attributed the whole effect to parameter buffers; the
parameter half is real and the gradient half sets the number.

Host-sync lowers the peak anyway because it throttles dispatch *globally*: with
the host unable to run ahead, deferred frees of every kind keep pace. It works
by accident of coarseness, which is why it is a knob and not a fix.

### What this establishes

An issue budget is two things. In the DAG it is an edge that bounds when a
collective's stream work may begin — `prefetch_distance` and `ring_exchange`'s
`distance` do that, and the CPU tests show it. In the runtime it is an
allocation policy that bounds what may be *held* — and Piper's runtime allocates
at host dispatch and frees on a background thread, so the second half does not
exist and the first half alone is measurably nothing (slope 3.99 vs 3.61).

The fix is in the runtime and invisible from the DAG: bounded pools for
full-parameter and full-gradient buffers, `distance+1` slots, reuse ordered by
the free event on the stream rather than by a host wait. That preserves the
asynchronous dispatch the host-sync knob sacrifices. Not built here; the
argument and the measurement that motivates it are.

### Timing in these runs is not a result

The memory gate deliberately shares cards with tenants at 100% utilization.
Iteration times swing from 0.066 s (shipped, 8 stages) to 0.33 s (pf1, 8
stages) with no schedule reason; F36's contamination signature. The overlap
questions stay with the four-card chains.

## 2026-09-13 — F44: a bounded buffer pool is the runtime half of the budget; with the DAG edge it gives ZeRO-3 slope 2.01, and both predictions held

### Tested

`PIPER_BUFFER_POOL=1` (`src/runtime.py::_BufferPool`, off by default): a
released full-parameter or full-gradient buffer goes back to a pool with the
event after which the GPU is done with it, and the next taker waits on that
event on its own stream. The host never blocks and the background free thread
is bypassed. Same model, depths and metric as F42/F43; predictions written in
the runner before it ran.

| arm | predicted | measured | peaks 2/4/8 stages (GiB) |
|---|---|---|---|
| `prefetch_distance=1` + pool | ≈2.0 | **2.01** | 4.16 / 6.67 / 11.69 |
| shipped ZeRO-3 + pool | ≈3.0 | **3.01** | 4.85 / 8.61 / 16.13 |

For reference from F42/F43: plain DP 4.00, shipped 3.61, DAG edge alone 3.99,
DAG edge + host blocked 2.01 (20.0 / 18.6 / 19.2 / 11.6 GiB at 8 stages).

Ranks report identical bytes in every pooled run; the F42 asymmetry was the
host/GPU race and the pool removes the race. Losses are identical to the
corresponding non-pool arms at every depth (28347.9492…, 1.58e9…, 1.48e18…), so
the pool computes the same thing. Those loss magnitudes are the `--init random`
scale through un-normalized MLP stages (per-stage gain ≈10.7, eight stages
≈1.7e8, squared), not a defect; memory is shape-driven and indifferent to it.

### What the two rows say together

The 3.01 row isolates the gradient half: with no DAG budget the forward still
gathers every layer (F37), the pool bounds the backward's gradient buffers, and
the peak returns to the forward gather-all at exactly the 3.0 the F42 accounting
predicted for shipped ZeRO-3 before the gradient race pushed it to 3.61. The 2.01
row is both halves: the DAG edge bounds the gathers' issue, the pool bounds what
is held, and the host runs free.

This closes the loop opened in F37. An issue budget is an edge in the DAG and an
allocation policy in the runtime. Piper shipped with neither; F39 added the edge
and F42 showed it alone is nothing; F43 showed why; this entry adds the policy
and measures the pair.

### What it took to get right

The first GPU run failed on the first reuse: release had detached the bucket
tensor by `set_` to shape `(0,)`, so the next alloc sized itself from
`numel()==0`. The fix — keep the shape, empty the storage — failed a CPU test I
wrote for the invariant: `set_` bounds-checks, and `resize_(0)` reaches that
state only because it does not. Under the pool the tensor object is rebuilt on
alloc and replaced on release, with every parameter view pointed at an empty
tensor so a stale use still fails loudly. Two rounds, both caught before or at
the first GPU attempt; the CPU test stays.

### Default

Off. The pool changes when memory is reclaimed, and the shipped examples lower
and run exactly as before with it off. Turning it on is the recommendation; the
measurement above is the argument.

### Timing hint, and what is still pending

Interleaved in one window at 8 stages, min iteration time: pool 0.0965 /
0.0949 s, host-sync 0.109 / 0.176 s. Cards were shared, so this is a hint: the
pool is not slower than blocking the host, and blocking the host is what it
costs. The 4-card overlap and P3 runs remain queued.

## 2026-09-14 — F45: the per-collective skew tax is a property of global synchronization, not of NCCL — and a ring confines it to one hop until the message outgrows the eager path

### Tested

`experiments/probe_skew_topology.py` on four H200s (ZJU host, cards 1/2/3/5 all
idle; this is the four-quiet-card window catalyst-fleet1 never produced in ~20
hours of waiting). Four ranks; rank 2 is delayed by a busy-wait kernel of
0/100/500/2000 us before each collective; every rank times its own call. Two
arms with identical payload: `all_reduce` (global) and a one-hop ring
(`batch_isend_irecv`, send to `r+1`, recv from `r-1`). 40 iterations, medians,
payloads 4/16/64 MiB.

Roles relative to the delayed rank 2: rank 1 is **upstream** (sends *into* it),
rank 3 is **downstream** (receives *from* it), rank 0 is **opposite**.

### Result — median cost of a 2000 us skew, over the unskewed baseline

| payload | collective | delayed | upstream | downstream | opposite |
|---|---|---|---|---|---|
| 4 MiB | all-reduce | −35 us | **+1869** | **+1876** | **+1873** |
| 4 MiB | ring | −61 | +4 | **+1852** | −4 |
| 16 MiB | all-reduce | −33 | **+1881** | **+1630** | **+1880** |
| 16 MiB | ring | −72 | +4 | **+1638** | +2 |
| 64 MiB | all-reduce | −25 | **+1880** | **+1880** | **+1872** |
| 64 MiB | ring | −67 | **+1669** | **+1849** | −7 |

Under an all-reduce, *every* rank that was not delayed pays the full delay, at
every payload. Under a ring at 4 and 16 MiB, exactly **one** rank pays — the one
receiving from the delayed rank — and the rank on the opposite side of the ring
pays nothing measurable (−4 to +2 us). The delayed rank itself always pays
negative: it arrives last and waits for no one.

### The 64 MiB exception, and why it is the useful part

At 64 MiB the **upstream** rank starts paying too (+1669 us), which it does not
at 4 or 16 MiB. A send into a rank that has not reached its receive can complete
into an eager buffer while the message is small; past the eager threshold it
needs the receiver, so the delay propagates one hop *backwards* as well. So the
locality is not a property of P2P as such — it is a property of P2P **below the
eager threshold**. That is a schedulable fact: it says the chunk size at which a
ring stops localizing skew is a tunable of the transport, not of the algorithm.

### What this settles

Stage E/F left this open. F19 established that each collective in a Piper step
re-pays a rank arrival difference of 157–2831 us, and F33 attributed the cost to
computation *between* collectives desynchronizing the ranks rather than to NCCL
per call. F45 completes it: the tax is levied by **global synchronization**.
Replacing the collective with a neighbour-coupled one does not make the delay
disappear — the downstream rank still pays it — but it stops charging it to
every rank. With three non-delayed ranks, an all-reduce burns 3x the delay in
aggregate wait and a small-payload ring burns 1x.

For CP this is the favourable half of the ring's structure, and it is
independent of whether the hoist wins: ring attention's exchange is exactly this
one-hop pattern, so a straggler costs one neighbour per step rather than the
whole group. It also bounds the claim: at chunk sizes past the eager threshold
(64 MiB here) the advantage halves, and a long-sequence CP run is squarely in
that regime.

### Measurement notes

The delayed rank's own numbers being *lower* than its unskewed baseline is the
signature that the injection worked: it is the one rank that never waits.
Contamination on this shared host shows in a few med/min ratios (6.46 for
all-reduce rank 3 at 500 us; 3.35 for ring rank 3 at 2000 us) — those are the
F36 signature and the reason the table reports medians of 40 iterations rather
than single samples. The deltas are 1–2 orders of magnitude above that noise.

### Environment

Second host, so this is also a portability check: the out-of-band CP gate
(F40) reproduced **bit-identically** on H200 + CUDA 12.8 (`out
4.172325134277344e-07`, `dQ 1.1920928955078125e-06`, `dK 2.86102294921875e-06`,
`dV 1.9073486328125e-06`), and all 78 CPU tests pass. The tree there is a clone
of `feat/tp-sharding`; commits are still made only on catalyst-fleet1.

## 2026-09-14 — F46: P1 falsified — a ring needs no issue budget, because its own data dependency is one; and at s_local=2048 the exchange is 1.4% of the step, so P2 is not testable there

### Tested

Four H200s. CP=4 ring attention, global seq 8192 (s_local 2048), dim 512, 8
heads, batch 2, five optimizer steps, three schedules: spliced,
`hoist distance=1`, `hoist distance=0`.

### Result 1 — all three are correct and indistinguishable

Every schedule matches dense CP=1 to 1.29e-05 across five steps, and the four
per-rank loss sequences are identical across schedules to every printed digit.
Peak memory is identical **to the byte** (3,099,330,048 on every rank in every
arm) and minimum iteration time is 0.0241–0.0260 s in all three.

### Result 2 — why, and what P1 got wrong

`cp-design.md` G3′ claimed that without a budget "every rotation completes
before step 0 finishes and every K/V chunk is resident — F37's pattern on the
feature meant to hold 1/n of K/V". Lowering the three DAGs shows the hoist
applied exactly as designed (`hoisted=True`, ring node dispatched *before* its
source region in both hoisted arms) and the budget edge doing nothing:

    spliced   : seg1 #1  ring0 #2  seg2 #3  ring1 #4  seg3 #5  ring2 #6
    hoist d=1 : ring0 #1  seg1 #2  ring1 #3  seg2 #4  ring2 #5  seg3 #6
    hoist d=0 : ring0 #1  seg1 #2  ring1 #3  seg2 #4  ring2 #5  seg3 #6

d=1 and d=0 are the same order. The reason is that **rotation `i` consumes the
chunk rotation `i-1` produced**: the ring nodes form a data chain, so at most
one rotation is ever in flight no matter what the budget says. F37's pattern
required the opposite — ZeRO-3's gathers are mutually *independent* DAG roots,
which is precisely why all nine fired at topological level 0.

That gives the criterion the design was missing:

> An issue budget is needed exactly when the collectives it governs are
> **mutually independent**. For a chain of collectives the data dependency is
> already the budget, and adding one is a no-op.

So the `distance` argument is not wrong for `ring_exchange`, it is *redundant*
there — and it is load-bearing for `replicate(prefetch_distance)`, which is
where F39/F44 measured it. The two features share the mechanism but not the
need. This is the first time the design's "one knob, two features" claim has
been qualified rather than confirmed.

### Result 3 — P2 is not testable at this problem size

K/V per rank is 2 x 2 x 8 x 2048 x 64 x 4 B = 16 MiB; from F45 a 16 MiB ring
exchange costs ~115 us, so the three exchanges are ~345 us of a 25 ms
iteration: **1.4%**. Perfect overlap would save 1.4%, an order of magnitude
below the run-to-run spread on this shared host. The identical timings are
therefore not evidence that the hoist fails to overlap; they are evidence that
the experiment had no resolution.

Ring attention's own scaling says where to look. Per step, communication is
proportional to `b·h·s_local·d` and compute to `b·h·s_local²·d`, so the
communication fraction goes as **1/s_local**: it rises as the local chunk
shrinks, i.e. exactly as CP degree rises at fixed global sequence. s_local=2048
is the regime where nobody needs CP. A sweep down to s_local=256 is queued.

### Correction recorded

`cp-design.md` G3′'s second paragraph is retracted; the criterion above
replaces it. The paragraph was written from the shape of the ZeRO-3 failure
without lowering a ring DAG at four steps — the same error as F37, one level
down, and caught the same way.

## 2026-09-14 — F47: replicated on different hardware, the race-free arms reproduce exactly and the racing arms do not — so without the pool, ZeRO-3's peak memory is a property of the machine, not of the program

### Tested

The F42/F44 memory-slope experiment re-run unchanged on two H200s (ZJU host,
CUDA 12.8, torch 2.10.0+cu128) against the original two B200s
(catalyst-fleet1). Identical model, depths, metric and code; different GPU,
different driver, different host load.

### Result

| arm | B200 slope | H200 slope | depends on the host/GPU race? |
|---|---|---|---|
| plain DP | 4.00 | **4.00** | no — no deferred frees |
| ZeRO-3 as shipped | 3.61 | **3.06** | **yes** |
| `prefetch_distance=1`, DAG edge only | 3.99 | **2.85** | **yes** |
| `prefetch_distance=1` + pool | 2.01 | **2.01** | no — pool removes it |
| shipped ZeRO-3 + pool | 3.01 | **3.01** | no — pool removes it |

Three arms reproduce to two decimals across different silicon. The two that do
not are exactly the two F43 identified as governed by how far the host runs
ahead of the GPU before the background free thread catches up. A faster GPU
keeps up better, fewer buffers are live at once, and the slope falls.

The ordering even flips: on B200, `prefetch_distance=1` was *worse* than shipped
(3.99 vs 3.61); on H200 it is *better* (2.85 vs 3.06). Same code, same
schedule, opposite conclusion about whether the directive helps.

### What this establishes

F43 argued from a memory trace that the peak is set by a host-ahead-of-GPU race
over deferred frees, and demonstrated it by blocking the host. This is the
stronger form of the same claim, and it needed a second machine to make:

> Under Piper's shipped ZeRO-3, **peak memory is not a function of the program.**
> It is a function of the ratio between host dispatch rate and GPU execution
> rate. The same schedule on the same code gives 3.61 or 3.06 stage-bytes
> depending on the card, and two schedules can swap places.

That is a correctness-of-reasoning problem before it is a performance problem: a
user who measures a memory saving from a scheduling change on one machine has
not measured a property of their schedule. The pool restores the property —
2.01 and 3.01 on both machines, to two decimals — which is a better argument
for it than the 11.6-vs-19.2 GiB headline in F44.

It also retroactively explains the rank asymmetry in F42 (up to 1.3 GiB between
two ranks of the same run) that vanished under both the host-sync knob and the
pool: two ranks of one job are two instances of the same race.

### Note on what was *not* replicated

The absolute peaks differ between hosts even in the reproducing arms (e.g. plain
DP 8 stages: 20.03 GiB on B200, 20.1 GiB on H200), which is allocator rounding
and context size, not signal. The slope is the invariant; the intercept is not.

### Environment

All 78 CPU tests pass on the H200 host and the out-of-band CP gate reproduces
bit-identically (F45). The tree there is a clone of `feat/tp-sharding` used as a
run site; commits are made only on catalyst-fleet1.

## 2026-09-14 — F48: with the ranks synchronized, collective cost *is* affine in bytes (421 GB/s, residual <= 11 us). F31/F32 measured the skew term, not the transport term — and symmetric GPU work does not produce that skew

### Tested

`experiments/probe_collective_law.py` on four idle B200s (catalyst-fleet1 went
quiet for the first time in this project: five cards at 0%). Payload swept
0.5–256 MiB, two collectives (`all_reduce`; a one-hop ring over
`batch_isend_irecv`), three regimes at each point, 30 iterations, minimum per
rank, group cost = slowest rank:

* **synced** — `barrier` + device sync immediately before each call;
* **natural** — a ~2 ms matmul between calls, no barrier, as inside a real step;
* **skewed** — rank 2 delayed 2000 us before each call.

The hypothesis under test, assembled from F19/F31/F32/F33/F45:

    cost  ~  max( transport(bytes),  arrival skew )

### Result 1 — the transport term exists and is clean

| MiB | AR synced | AR natural | AR skewed | ring synced | ring skewed |
|---|---|---|---|---|---|
| 1 | 32.6 | 31.1 | 1983 | 57.3 | 1929 |
| 8 | 59.2 | 59.7 | 1973 | 95.6 | 1996 |
| 32 | 132.3 | 134.4 | 2046 | 243.3 | 874 |
| 64 | 197.1 | 198.2 | 2101 | 437.2 | 2337 |
| 256 | 678.8 | 679.2 | 2549 | 840.5 | 2108 |

Affine fit over 4–256 MiB, synced:

    all_reduce:  41.7 us + 2.49 us/MiB   ->  421 GB/s,  max |residual| 11 us
    ring      : 139.8 us + 3.31 us/MiB   ->  317 GB/s,  max |residual| 264 us

**`all_reduce` is a textbook bandwidth model.** Eleven microseconds of residual
across a 64x payload range is as clean as this kind of measurement gets.

F31 and F32 reported that a 32x payload sweep changed cost by only 2.3x and
concluded that cost is near-independent of volume. That conclusion is now
qualified: it was measured with a real Piper DAG between collectives, where
F19's 157–2831 us of arrival skew dwarfs the 60–200 us of transport that TP
payloads actually need. **They measured the skew term.** The transport term was
there all along, below it.

The skewed column is the same law from the other side: flat at ~2000 us while
the injected skew dominates, and rising above it only once transport becomes
comparable (256 MiB: 2549 ~ 2000 + 549). For `all_reduce` the crossover is
`2.49 * MiB + 41.7 = skew`; at F19's typical few-hundred-us skew that is
~100 MiB, and at 2000 us it is ~800 MiB.

### Result 2 — the ring is *not* affine, on the same hardware in the same sweep

The ring's residual is 264 us, 24x the all-reduce's, and the misfit is
structured: 128 MiB costs 827 us and 256 MiB costs 840 us — 2x the bytes for
1.6% more time. NCCL switches protocol/channel configuration under the ring,
and the cost is piecewise. So "a bandwidth model is valid" is a claim about a
*collective*, not about a machine: it holds for `all_reduce` here and fails for
raw neighbour P2P beside it.

### Result 3 — the skew is not made by symmetric GPU work

`natural` and `synced` agree to a median of **2.2 us** across all twenty
(payload, collective) pairs, against the 2000 us the injection produces. Four
identical GPUs running identical matmuls do not drift apart.

This contradicts the mechanism F33 proposed — "computation between collectives
desynchronizes the ranks" — in its general form, and narrows it to something
sharper. The arrival skew inside a Piper step cannot come from symmetric device
work. What is left is **host-side asynchrony**: each rank runs its own Python
dispatch loop in its own Ray process, and those loops are not coupled to each
other by anything except the collectives themselves.

That is the same mechanism F43 and F47 found on the memory side: the host runs
ahead of the GPU, independently per rank. One asynchrony, two symptoms —
unbounded buffer residency and arrival skew. F47 showed the memory symptom is
machine-dependent because the race is; the skew should be too, which the H200
run of this same sweep will say.

### Result 4 — the ring's localization threshold is between 32 and 64 MiB here

Per-rank cost of the 2000 us skew, over each rank's own synced baseline
(rank 2 delayed; rank 1 sends into it, rank 3 receives from it):

| MiB | collective | r0 (opposite) | r1 (upstream) | r2 (delayed) | r3 (downstream) |
|---|---|---|---|---|---|
| 4 | all_reduce | +1021 | +1458 | +8 | +1928 |
| 4 | ring | −0 | −1 | −32 | **+1909** |
| 32 | ring | −2 | +2 | −33 | **+638** |
| 64 | ring | −6 | **+1696** | −41 | **+1903** |
| 256 | ring | −7 | **+1277** | −54 | **+1277** |

F45 found on H200 that the upstream sender begins paying at 64 MiB but not at
16. B200 agrees: clean localization at 4 and 32 MiB, upstream paying from 64.
Two different machines put the knee in the same octave, which suggests a
protocol constant rather than a bandwidth effect. A fine sweep between 16 and
96 MiB on both hosts is queued to pin it.

### Consequence for the project's thesis

The paper's limitation paragraph said automatic TP selection must model the
runtime rather than the hardware, "because communication cost is not a function
of communication volume". The conclusion stands; its reason was wrong and the
correct reason is more useful. Cost *is* a function of volume. The roofline
failed because it was missing a second term — arrival skew — which at TP
payloads is the larger of the two, and which belongs to the runtime because the
runtime is what lets the ranks drift.

That converts a dead end into a specification: a usable cost model for Piper is
`max(a + b*bytes, skew)` with `a, b` measurable in one sweep like this one and
`skew` a property of the dispatch loop. Whether `skew` is predictable at all is
the next question, and it is a runtime question.

### Correction to F48, same day, from its own data

F48 above states the law as `cost ~ max(transport(bytes), arrival skew)`. Its
own table falsifies that form. Subtracting the synced arm from the skewed arm
at each payload:

| MiB | synced | skewed | difference |
|---|---|---|---|
| 4 | 47.8 | 1973.6 | 1925.8 |
| 32 | 132.3 | 2045.9 | 1913.6 |
| 64 | 197.1 | 2100.9 | 1903.8 |
| 128 | 360.5 | 2268.1 | 1907.6 |
| 256 | 678.8 | 2549.2 | 1870.4 |

The difference is constant to within 3% across a 64x payload range. Under a
`max` law it would have collapsed toward zero as transport approached the
skew; it does not. The terms **add**:

    cost(waiting rank) = transport(bytes) + skew
    cost(late rank)    = transport(bytes)

which is also what the mechanism requires: the group cannot begin until the
last rank arrives, so every other rank measures its own wait plus the transfer,
while the late rank measures the transfer alone. The per-rank table in Result 4
already showed the late rank paying ~0 and is consistent with this.

The distinction matters for the thing F48 is about. Under `max`, once skew
dominates, reducing communication volume buys nothing — which is how F31/F32's
"payload barely matters" invited being read. Under addition, cutting bytes
always buys its full transport time even while skew is the larger term, so
`fuse_collectives` and payload reductions keep their value in exactly the
regime where the earlier reading said they had none.

The `max` form was written from the two endpoints of the skewed row without
differencing it against the synced row sitting beside it — the F17 error
(fitting a curve through two points) in a new place. `probe_skew_transfer.py`
sweeps skew over 0-3200 us at three payloads to test the form rather than infer
it; it predicts `(waiting - base)/skew = 1.0` everywhere, against a `max` law's
prediction that the ratio falls below 1 whenever base > skew.

## 2026-09-14 — F49: the additive law, proved without reference to the injected value; arrival differences below ~100 us are partly absorbed; and the ring's localization ends in a step between 32 and 40 MiB

### The law, in measured quantities only

`experiments/probe_skew_transfer.py`, four idle B200s, all-reduce, skew swept
0–3200 us at 4 / 32 / 128 MiB. The quantity below uses no assumption about how
much delay was actually injected: it is the **waiting rank's cost minus the late
rank's cost** in the same iteration. The late rank waits for nobody, so its cost
*is* the transport term, measured rather than modelled.

| skew req. | 4 MiB | 32 MiB | 128 MiB | spread |
|---|---|---|---|---|
| 25 | 14.5 | 18.3 | 8.6 | 70% |
| 50 | 34.9 | 35.6 | 24.8 | 34% |
| 100 | 71.7 | 73.3 | 76.2 | **6.2%** |
| 200 | 160.7 | 161.8 | 150.8 | **7.0%** |
| 400 | 335.6 | 337.7 | 326.8 | **3.3%** |
| 800 | 660.7 | 682.0 | 666.2 | **3.2%** |
| 1600 | 1371.0 | 1373.5 | 1368.5 | **0.4%** |
| 3200 | 2758.1 | 2752.1 | 2749.4 | **0.3%** |

Transport at these payloads is 37 / 120 / 350 us, so the 32x payload range
changes the transport term by 9.5x while leaving this difference constant. That
is the additive law:

    cost(waiting rank) = transport(bytes) + arrival difference
    cost(late rank)    = transport(bytes)

**`max` is refuted directly, not by inference.** At 128 MiB the transport is
350 us, so `max(transport, skew)` predicts that a skew of 25/50/100/200 us costs
nothing at all. Measured: +9 / +25 / +76 / +151 us. The F48 body's `max` form is
retracted; the correction appended to F48 stands and this is its test.

### Arrival differences below ~100 us are partly absorbed

The difference tracks the injected skew with a slope of 0.86 and an offset that
is proportional rather than constant (25 us of skew buys 8.6–18.3 us of cost;
3200 buys 2749–2758). Two contributions, separated:

* `experiments/probe_sleep_cal.py` shows the injection itself is faithful —
  delivered/requested is 1.000 at 3200 us and 1.26 at 25 us (it *over*-delivers
  when small). The in-probe calibration was cold, one spin with no warmup, and
  read 1853.7 cycles/us against a careful 1963.8, so it under-injected by 5.6%.
* The remaining ~9% is real: a waiting rank's own NCCL setup overlaps the head
  of its wait, so a fraction of any arrival difference is hidden.

The practical form of this: **an arrival difference under about 40 us is free,
and the absorbed fraction is roughly a tenth thereafter.** Piper's measured
skews (F19: 157–2831 us) are an order of magnitude above the free region.

### What this settles about `fuse_collectives`

F34 measured fusion worth 2.35x on communication time and explained it by byte
count and NCCL call overhead. The additive law gives the real account: each
collective pays its own arrival difference, so fusing `n` collectives into one
saves `(n-1)` skew payments, not `(n-1)` launch overheads. In the fused
schedules that is 28 skew payments of 157–2831 us each against transport terms
of order 100 us. The mechanism was mis-attributed and the measurement stands.

It also answers the question F31/F32 raised and could not: cutting bytes is
*always* worth its full transport time, because the terms add. A schedule that
both fuses and shrinks payload collects both.

### The ring's localization ends in a step, not a slope

Fine sweep, 16–96 MiB, per-rank cost of a 2000 us skew over each rank's own
synced baseline (rank 2 late; rank 1 upstream; rank 3 downstream):

| MiB | opposite | **upstream** | late | downstream |
|---|---|---|---|---|
| 16 | 3 | **3** | −32 | 1860 |
| 24 | 0 | **1** | −34 | 1895 |
| 32 | −27 | **0** | −34 | 1913 |
| 40 | 5 | **1699** | −37 | 1895 |
| 48 | 0 | **1684** | −36 | 1177 |
| 64 | −3 | **1700** | −40 | 1892 |
| 96 | 2 | **1689** | −38 | 1902 |

Between 32 and 40 MiB the upstream sender goes from paying nothing to paying
1699 us. A **step**, not a transition: 0 at 32, 1699 at 40, and flat either
side. That is a buffer or protocol constant, not a bandwidth effect — which the
smooth affine cost curve over the same range (F48) already implied, since the
transport term shows no feature there at all.

So F45's statement is sharpened: a ring localizes a straggler to one hop for
messages under ~36 MiB on this hardware and to two hops above it, and the
boundary is sharp enough to be designed around. For CP this is a chunk-size
rule: K/V per step under the knee keeps a straggler's cost on one neighbour.
Whether the constant is the same on H200 is the point of the paired sweep.

### Method note

Every number above is a minimum over 25 iterations on four cards whose
utilization was 0% throughout, on a host that was contended for the whole
project until this window. The payload-independence is what carries the
argument, and it is a ratio between two quantities measured in the same
iteration on the same ranks, so it survives whatever the machine was doing.

## 2026-09-14 — F50: the runtime's share of a Piper iteration is a ~8 ms constant, so the GPU does 12% of the work at the sizes these experiments use and 84% at 68x the work

### Tested

`experiments/probe_host_bound.py` runs one CP rank's forward and backward —
the same arithmetic, including every ring step — in one process on one GPU with
no Ray, no DAG and no dispatch loop, and omitting only the collectives. Paired
against Piper's own minimum iteration time for the identical configuration
(CP=4, four B200s, `cp4_ring_dp`, dim 512, 8 heads, batch 8, 10 timed
iterations).

### Result

| seq | s_local | GPU math | Piper iteration | **non-arithmetic** | GPU share |
|---|---|---|---|---|---|
| 1024 | 256 | 1.35 ms | 11.08 ms | **9.73 ms** | 12.2% |
| 2048 | 512 | 5.89 | 12.64 | **6.75** | 46.6% |
| 4096 | 1024 | 24.01 | 32.04 | **8.03** | 75.0% |
| 8192 | 2048 | 91.57 | 109.70 | **18.13** | 83.5% |

The arithmetic spans 68x. The non-arithmetic remainder is 6.8–9.7 ms over the
first three points — flat — and rises only at the largest, where K/V is 32 MiB
per tensor and communication is itself material (and, per F49, above the ring's
32–40 MiB localization knee).

So a Piper iteration is, to first order:

    iteration  ~  GPU arithmetic  +  ~8 ms of runtime  +  collectives

and the ~8 ms does not depend on the model. At these experiment sizes it is the
*dominant* term: at s_local=256 the GPU is doing the model's arithmetic for
12% of the iteration and something else for 88%.

### Why this matters more than the CP result it came from

It supplies the criterion this project has been missing for which negative
results are informative. A scheduling change that removes `x` of communication
changes the iteration by `x / (math + 8 ms + comm)`. At s_local=256 the three
ring exchanges are ~345 us (F45/F49) against an 11 ms iteration: 3%, of which
overlap could recover part. That is below the run-to-run spread, and no amount
of repetition fixes it — the experiment has no resolution *by construction*.

This retroactively explains a family of earlier non-results as one thing:
`stream` having no effect at one microbatch, the hoist being unmeasurable
(F46), and the 1.06x ceiling on stream at four microbatches. None of them were
statements about the directive. They were statements about a denominator.

**The rule**: before measuring a scheduling knob, measure the GPU's share. Below
about 50% the knob is being tested against the runtime, not against the
schedule.

### Relation to F43 / F47 / F48

This is the fourth symptom of one cause. The host runs its own dispatch loop per
rank, ahead of the GPU and uncoupled from the other ranks:

* F43 — it allocates every full-parameter buffer before the GPU needs them;
* F47 — so peak memory is a property of the machine, not of the program;
* F48 — so ranks drift, and each collective pays the drift additively;
* F50 — and it takes ~8 ms per iteration whatever the model does.

The first three were diagnosed from the inside, through the DAG and the
collectives. This one is the same asynchrony measured from the outside, by
subtraction against the bare arithmetic, and it is the largest of the four.

### Caveat, stated rather than smoothed

The bare-arithmetic reference runs on one card while Piper runs on four, so
clock and cache state differ; a standalone run of the same reference on an idle
card gave 2.51 ms at s_local=256 against the 1.35 ms measured here. Taking the
larger figure the GPU share at s_local=256 is 23%, not 12%, and every conclusion
above is unchanged. The reference is also generous to Piper: it includes all the
ring arithmetic and excludes only the collectives.

### Next

The same sweep on H200 — if the ~8 ms constant is a property of Ray and the
Python dispatch loop it should reproduce; if it is a property of the host it
should not. That is the F47 test applied to the fourth symptom.

## 2026-09-14 — F51: run on two machines, every structural claim reproduces and every constant lands where its mechanism says it should

### Method

The F48/F49/F50 sweeps re-run unchanged on four H200s (ZJU host, NVLink 4,
CUDA 12.8, a box with 94 logged-in users and load ~8) against four B200s
(catalyst-fleet1, NVLink 5, CUDA 12.8, idle). Same code, same payloads, same
iteration counts. This is the F47 test applied to everything Stage G claims:
a structural claim should reproduce, a hardware constant should not, and a
software constant should.

### Result

| quantity | B200 | H200 | reproduces? | what it therefore is |
|---|---|---|---|---|
| all-reduce intercept `a` | 41.7 us | **44.5 us** | **yes, 7%** | launch + protocol — software |
| all-reduce slope `b` | 2.49 us/MiB | **4.51 us/MiB** | no, 1.81x | fabric bandwidth — hardware |
| affine form, max residual | 11.0 us | **7.6 us** | **yes** | the law itself holds |
| ring is *not* affine, residual | 264 us | **254 us** | **yes** | NCCL protocol piecewise |
| ring localization knee | 32 -> 40 MiB | **32 -> 40 MiB** | **yes, exactly** | a P2P buffer constant |
| `natural - synced` | ~2 us | **~3 us** | **yes** | symmetric work never drifts ranks |
| additive law (`max` refuted) | confirmed | **confirmed** | **yes** | queueing, not hardware |
| runtime overhead, form | flat over 18x work | **flat over 3.8x** | **yes** | a per-iteration host constant |
| runtime overhead, value | 6.8–9.7 ms | **10.3–10.9 ms** | no | host CPU and load |

Effective bandwidths (421 vs 232 GB/s) are in the right ratio for NVLink 5
against NVLink 4, so the slope is measuring what it should.

### The two entries that carry the most

**The intercept is software and the slope is hardware, and the sweep separates
them.** 41.7 vs 44.5 us of fixed cost on machines whose bandwidths differ by
1.8x is not a coincidence; it is the affine form being the right
decomposition. A cost model for Piper can therefore be *calibrated per machine
in one sweep* and its structure carried across — which is precisely what a
scheduling search needs and what F31/F32's "cost is not a function of volume"
appeared to deny.

**The ring's knee is at the same place on both machines.** Two GPU generations,
two NVLink generations, two hosts: the upstream sender pays 0 us at 32 MiB and
~1650–1700 us at 40 MiB on both. A bandwidth effect could not do that. It is a
protocol/buffer constant, and it turns F45's observation into a design rule
that transfers: **keep a ring's per-hop message under 32 MiB and a straggler
costs one neighbour; above 40 MiB it costs two.** For CP that is a chunk-size
constraint, `batch * heads * s_local * head_dim * 4 < 32 MiB`, and it is
checkable at compile time from the traced shapes.

### The one that is most uncomfortable

The runtime overhead is flat on both machines and *larger on the busier host*
(10.6 ms on a box with 94 users against 8.2 ms on an idle one). That is
consistent with it being host-side work — which is what F50 concluded — and it
means the denominator every scheduling measurement in this project divides by
is set by who else is logged into the machine. It does not invalidate the
comparisons, which were interleaved A/B within one window, but it does mean a
number like "`order` is worth 16%" is a number about one machine at one moment
in a way that "collective cost is `a + b*bytes`" is not.

### What is now established well enough to hand over

Structure, on two machines: collective cost is affine in bytes for an
all-reduce and piecewise for raw P2P; skew and transport add rather than
dominate; symmetric GPU work does not create skew; a ring localizes a straggler
below a 32-40 MiB protocol knee; the runtime costs a model-independent constant
per iteration. Constants: per machine, one sweep each, scripts in
`experiments/probe_collective_law.py`, `probe_skew_transfer.py`,
`probe_host_bound.py`.

Not established: whether the ~8-11 ms is Ray, the Python dispatch loop, or the
actor round trip — a breakdown needs host-side instrumentation of `run_dag`,
which is the obvious next measurement and the one that would say what to fix.

## 2026-09-14 — F52: the per-iteration constant is not Ray and not the scheduling loop — it is the segmented execution path, which costs 4.6x the host launch time of the same arithmetic run straight

### Tested

`PIPER_TIME_TRACE` records a host timestamp before every dispatched node, plus
markers after the last enqueue and after the drain, so an iteration splits into
host dispatch / GPU drain / everything else. CP=4 on four B200s, ring model,
s_local=256, 10 timed iterations. Paired with `probe_host_bound.py`, now
reporting the bare reference's **host** launch time as well as its GPU time.

### Result 1 — the iteration *is* the dispatch loop

| | steady-state |
|---|---|
| host dispatch (first node to last enqueue) | 10.3–11.7 ms |
| GPU drain after the last enqueue | **0.10–0.13 ms** |
| driver-side iteration | 11.85 ms |
| Ray + driver remainder | **~0 ms** |

By the time the host finishes issuing 21 nodes the GPU has finished executing
them. F50's open question — Ray, the actor round trip, or the loop — is
answered: **it is the loop, and the Ray round trip is not measurable against
it.**

### Result 2 — but the loop's cost is in the compute nodes, not the scheduling

| task | n | total | mean |
|---|---|---|---|
| backward | 6 | 5.93 ms | **989 us** |
| forward | 6 | 1.69 | 282 |
| update | 1 | 1.46 | 1464 |
| backward_ring_exchange | 3 | 0.88 | 295 |
| forward_ring_exchange | 3 | 0.72 | 239 |
| all_reduce | 2 | 0.35 | 177 |
| **total** | **21** | **11.05 ms** | |

Compute nodes are **82%**; every communication node together is 18%. So this is
not scheduling overhead in the sense of "the DAG machinery is expensive".

### Result 3 — it is the *segmented* execution path, and that is measurable

The bare reference runs the identical arithmetic as one straight Python
function. Its host launch time:

| s_local | bare GPU | **bare host launch** | Piper host dispatch |
|---|---|---|---|
| 256 | 2.39 ms | **2.40 ms** | **11.05 ms** |
| 512 | 3.67 | **2.81** | ~11 |
| 1024 | 11.81 | **2.90** | ~11 |

The same math costs **2.4 ms** of host time launched straight and **11.05 ms**
launched through Piper — **4.6x** — and the bare figure barely moves with
problem size, so the gap is per-node, not per-flop. Restricted to the compute
nodes, 9.08 ms against 2.4 ms is **3.8x**.

What the segmented path does that the straight one does not: interpret twelve
FX GraphModules instead of running one function; detach, clone and re-attach at
every segment boundary; and call autograd once per segment rather than once per
step. `backward` at 989 us per node against `forward` at 282 us is consistent
with the last of these being the largest term.

### Correction to F50

F50 called the ~8 ms "the runtime's share" and grouped it with F43/F47/F48 as a
symptom of host-side asynchrony. The grouping was too quick. The asynchrony
finding stands for the memory race and the arrival skew, which are about ranks
drifting *relative to each other*. This constant is a different thing: a
single-rank, per-node cost of the execution model, present even with one rank
and no collectives. F50's practical criterion is unaffected — the denominator
is what it is — but its attribution is corrected here.

### The obvious mitigation works for the shipped example and fails for this one

`--use-inductor` compiles each GraphModule, which should collapse the
interpretation cost. It runs clean on the shipped `test_tp_mlp` example
(rc=0) and fails on the ring model with
`InductorError: SubprocException ... RuntimeError: PassManager::run failed`.
So the mitigation exists, is off by default in both examples, and does not
currently apply to the model this project added. Whether the failure is the
online-softmax `-inf` initialisation or the segment shape is not yet known and
is worth one bisect, because it decides whether "turn on Inductor" is a general
answer to Result 3 or only a sometimes-answer.

### Method note

Per-node cost is the gap between consecutive host timestamps, so node `i` is
charged with its own dispatch plus any blocking that node `i+1` does before its
timestamp. With the drain at 0.1 ms the GPU is never the thing being waited on,
so the attribution is sound here; it would not be in a GPU-bound regime.

## 2026-09-14 — F53: the 4.6x host-side tax decomposed — segmentation 1.3-1.7x, FX interpretation another 1.2-1.6x, and about 1.8x that is Piper's own per-node machinery

### Tested

`experiments/probe_segmentation_cost.py`: the same ring arithmetic three ways,
plain PyTorch, no Ray and no DAG, host launch time as the metric (the GPU is
starved throughout, and its time tracks the host's in every variant).

* **A straight** — one function, one backward over the whole thing.
* **B detached** — a `detach().clone().requires_grad_()` boundary between ring
  steps and `torch.autograd.grad` called once per segment with explicit
  `grad_outputs`. This is what Piper's segment boundaries do.
* **C detached + fx** — B with each segment run through `torch.fx.Interpreter`,
  which is how Piper executes a segment's GraphModule.

Inductor, the obvious way to ask the same question, fails on both hosts for
unrelated toolchain reasons (B200: Triton's NVIDIA backend, `make_ttgir` →
`PassManager::run failed`; H200: `gcc` on Triton's `cuda_utils.c`), while the
model's own math compiles standalone on B200. So the cost was decomposed
directly instead.

### Result

| variant | B200 host | vs A | H200 host | vs A |
|---|---|---|---|---|
| A straight | 1.97 ms | 1.00x | 3.47 ms | 1.00x |
| B detached | 3.43 | **1.74x** | 4.39 | **1.26x** |
| C detached + fx | 4.19 | **2.13x** | 6.83 | **1.97x** |

Both machines order the three the same way and land within 8% of each other on
the combined factor (2.13x, 1.97x), which by the F51 criterion makes this a
structural property rather than a machine one. FX interpretation on top of
segmentation is 1.22x (B200) and 1.56x (H200).

### The accounting

Piper's compute nodes cost 9.08 ms of host time (F52) against 2.4 ms for the
same arithmetic written straight — 3.8x. Of that:

| component | factor | measured by |
|---|---|---|
| detach/clone boundaries + per-segment autograd | 1.3–1.7x | B/A |
| FX interpretation of each segment | 1.2–1.6x | C/B |
| **the two together** | **2.0–2.1x** | C/A |
| everything else Piper does per node | **~1.8x** | 3.8x / C/A |

The remainder is not mysterious, it is just not isolated here: a CUDA event
created and recorded per node, the buffer store's dict traffic and refcounting,
the `match task_type` dispatch with its predecessor lookups, the
`requires_grad_`/contiguity handling at each boundary — and the fact that
Piper's DAG has twelve compute nodes for this model where the probe has four
segments, so per-segment costs are counted three times as often.

### Why this is the useful form of F50

F50 said the runtime costs a model-independent ~8-11 ms and that below about
50% GPU share a scheduling knob is being tested against the runtime. F52 said
the cost is the loop and 82% of it is compute nodes. This says what inside the
compute nodes, in components that can be attacked separately:

* the FX term is what `--use-inductor` is for, and it is *off by default in
  every shipped example*, so every number in the Piper paper and in this one is
  measured with it on the table;
* the segmentation term is the price of the IR itself — cutting autograd at
  every boundary is what makes boundary collectives expressible (§Background),
  so it is not removable without changing what Piper is;
* the per-node term is ordinary engineering, and the event-per-node is the
  first thing to look at.

That is a different conclusion from "Piper has overhead". One of the three
components is load-bearing for the abstraction, one is a compile flag that does
not currently work on either of our hosts for this model, and one is
implementation.

### Caveat

The probe's four segments are not Piper's twelve compute nodes, so B and C
under-count relative to Piper by roughly the segment-count ratio; the 1.8x
remainder is an upper bound on Piper-specific machinery and would shrink if the
probe matched the node count. The direction and the ordering are what the
two-machine agreement supports, not the third decimal.

## 2026-09-14 — F54: F53's decomposition retracted — Piper runs codegen'd GraphModules, not an Interpreter, and with the variants interleaved only the segmentation term is resolvable

### Two errors in F53, found by checking the code and then the method

**It measured the wrong thing.** F53's variant C ran each segment through
`torch.fx.Interpreter` on the stated ground that this "is how Piper executes a
segment's GraphModule". It is not. `actor.py:385` calls `module.recompile()`
and `bucket.forward_fn` is a closure that calls the module directly, so Piper
runs the **codegen'd** `GraphModule.forward` — ordinary generated Python, not a
node-by-node interpreter. The gap between those two is roughly the whole term
F53 attributed to FX.

**It was measured in blocks.** Re-running variant by variant put B (detached)
*above* C (detached + GraphModule) on B200 — 4.87 against 3.17 ms — which is
impossible by construction, C being B plus work. That is the signature of
drift, and the project has had a rule for it since F11: interleave, and take
the minimum of each arm. That rule had been applied to every GPU timing in this
project and to none of the host timings.

### Re-measured, interleaved, 25 rounds, with the spread reported

| variant | B200 min | vs A | H200 min | vs A |
|---|---|---|---|---|
| A straight | 2.05 ms | 1.00x | 2.99 ms | 1.00x |
| B detached | 4.19 | **2.05x** | 4.43 | **1.48x** |
| C detached + codegen'd gm | 3.74 | 1.83x | 4.53 | 1.51x |
| D + name-keyed arg marshalling | 3.97 | 1.94x | 4.36 | 1.46x |

min-to-median spread: B200 0.28 / 0.75 / 1.06 / 0.90 ms; H200 0.70 / 1.58 /
1.32 / 1.12 ms.

**A against the rest is resolvable** — a 1.4–2.1 ms gap with A's own spread at
0.28 and 0.70. **B, C and D are not**: they differ by 0.2–0.45 ms against
spreads of 0.9–1.6 ms, and their order is not even consistent between the two
machines.

### What survives

> Cutting autograd at every segment boundary — `detach().clone().requires_grad_()`
> on the way in and `torch.autograd.grad` per segment instead of one backward —
> costs **1.5–2.0x** the host launch time of the same arithmetic written
> straight. Running the segment as a codegen'd GraphModule, and marshalling its
> arguments by placeholder name, are below the noise floor of this measurement
> and are not separately priced.

F53's three-component table is withdrawn. The one term that was load-bearing
for the argument is the one that survives: the boundary cost is the price of
the IR itself, since cutting autograd at boundaries is what makes boundary
collectives expressible at all. F53's remark that `--use-inductor` addresses
the FX term also goes, since there is no measured FX term; what Inductor would
address is the kernel count inside each segment, which is a different claim and
untested (it fails on both hosts for this model, F52).

Piper's compute nodes cost 3.8x the straight arithmetic (F52). Segmentation
accounts for 1.5–2.0x of that. The remaining ~2x is not decomposed, and this
method will not decompose it: an external mock cannot reproduce the executor's
per-node work. Instrumenting inside `DagExecutor.run` at sub-node granularity
would, and is the way to do it if it matters.

### Method note, recorded because it cost two wrong tables

Host-side Python timing on a shared machine drifts as much as GPU timing does,
and for the same reason. Every host measurement from here on is interleaved and
reports its spread, and no difference is claimed that does not exceed it. The
first two attempts at this table each produced a plausible, publishable-looking
decomposition that was an artifact.

## 2026-09-14 — F55: the executor's wrapper is thin; the cost is the segmented arithmetic plus twelve nodes the reference does not have — and a comm node costs ~9x its own transport in host time

### Tested

`PIPER_TIME_TRACE` now also times the `compute.forward` / `compute.backward`
call inside each compute node, so a node splits into the call and the executor
around it. CP=4, four B200s, ring model, s_local=256, 12 iterations, steady
state only (last six).

### Result

| task | n | total us | inside compute.* | executor | inside % |
|---|---|---|---|---|---|
| backward | 6 | 762 | 656 | **106** | **86%** |
| forward | 6 | 543 | 477 | **66** | **88%** |
| update | 1 | 1974 | — | 1974 | — |
| backward ring exchange | 3 | 402 | — | 402 | — |
| forward ring exchange | 3 | 399 | — | 399 | — |
| all-reduce | 2 | 193 | — | 193 | — |

Per iteration: **12.59 ms total, 6.80 ms inside `compute.*` (54%), 5.79 ms
executor (46%)**.

### Three things this settles

**The executor's per-node wrapper is not the problem.** Around a compute node
it is 66–106 us — event creation, buffer-store traffic, the `match` dispatch,
predecessor lookups, boundary `requires_grad_` handling, all of it — against a
543–762 us node. F54 left "~2x unattributed" and suggested per-node machinery;
that guess is now excluded for compute nodes.

**The 3.8x comparison was against a reference missing a third of the work.**
F52 compared Piper's compute-node time to the same arithmetic written straight.
But Piper's DAG for this model has *twelve* non-compute nodes — six ring
exchanges, two all-reduces, an update, and the rest — costing 5.79 ms, and the
straight reference has no counterpart to any of them. The honest split is:

* segmented arithmetic: 6.80 ms against 2.05 ms straight (F54) — **3.3x**, and
  F54 priced 1.5–2.0x of that as the boundary cost;
* everything the reference does not do at all: **5.79 ms**, of which the
  optimizer step is 1.97 ms and is real work by any measure.

"Piper costs 4.6x" was therefore comparing unlike things. The defensible
statement is the first bullet.

**A communication node costs about nine times its own transport, in host time.**
Each ring exchange moves 1 MiB per tensor; by F48's affine fit that is
$41.7 + 2.49 \approx 44$ us on the wire. The node costs **399–402 us of host
time**. The all-reduce node, which touches one tensor instead of two, costs
193 us — so this is ordinary per-tensor, per-op Python cost (a
`detach().clone()`, an `empty_like`, `P2POp` objects, `batch_isend_irecv`,
`wait`, an event, buffer dict copies), scaling with the tensors touched, not one
pathological call. Roughly **100–200 us of host time per tensor a collective
touches.**

### The law, as it applies inside Piper

F48/F49 measured `cost = transport(bytes) + skew` in a tight loop with no
dispatch. Inside Piper a third term sits in front of it:

    cost of a collective in a Piper step
        ~ host_dispatch(~100-200 us per tensor) + transport(bytes) + skew

and at TP and CP payloads the host term is comparable to or larger than the
transport term (400 us against 44 us here). That is the third and best account
of `fuse_collectives`: F34 explained its 2.35x by bytes, F49 corrected that to
`(n-1)` saved skew payments, and this adds `(n-1)` saved host dispatches of
100–200 us per tensor. Fusing 32 collectives into 4 saves 28 of each.

It also explains why fusion helped on single-stage TP and was indistinguishable
under 1F1B (F34): under 1F1B the host has other nodes to dispatch in the gap, so
the saved host time is absorbed; on a single stage it is on the critical path.

### What is not resolved

Why the segmented arithmetic costs 3.3x when F54 priced the boundary at
1.5–2.0x. The gap is inside `fwd_fn(fwd_args)` and the per-segment
`torch.autograd.grad`, and F54's mock had four segments where Piper has six of
each. Matching the segment count in the mock is the cheap next step; if the
factors then agree, the accounting is closed.

### Addendum, same day: F55's open item, partly closed

F55 asked why the segmented arithmetic costs 3.3x when F54 priced the boundary
at 1.5–2.0x, and named the likely cause: F54's mock cut four boundaries where
Piper's DAG has six compute nodes per pass. Making the segment count an axis:

| machine | segments | A straight | B detached | C +gm | D +marshalling |
|---|---|---|---|---|---|
| B200 | 4 | 2.09 ms | 4.06 (**1.94x**) | 3.99 (1.91x) | 3.51 (1.68x) |
| B200 | 6 | 1.90 | 4.86 (**2.56x**) | 4.21 (2.22x) | 4.53 (2.38x) |
| H200 | 4 | 3.00 | 4.01 (**1.34x**) | 4.09 (1.36x) | 4.07 (1.36x) |
| H200 | 6 | 2.90 | 4.04 (**1.40x**) | 4.08 (1.41x) | 4.05 (1.40x) |

On B200 the factor rises with the boundary count as it should — 1.9x at four,
2.2–2.6x at six — and at Piper's count it reaches two-thirds to three-quarters
of Piper's 3.3x. That much of the gap is now attributed.

On H200 it does not: 1.34x at four boundaries and 1.40x at six, an increment
inside the spread. The absolute boundary cost is ~1.15 ms there against ~2.5 ms
on B200, which is the wrong direction for a Python-side cost on a slower host
(H200's straight variant is 2.90 ms against B200's 1.90). H200's straight arm is
also unusually stable (min-to-median 0.09–0.17 ms against the segmented arms'
0.23–1.65), which suggests it is sitting on a floor set by something other than
the work — most likely launch queue behaviour — that the segmented arms then do
not add to proportionally. Not explained; recorded.

So: **the boundary cost is resolvable against the straight arithmetic on both
machines and accounts for most of Piper's segmented-arithmetic gap on B200, it
does not scale with boundary count on H200, and the B/C/D distinctions stay
below the noise floor everywhere.** The `--segments` axis is in the probe for
whoever picks this up.

## 2026-09-14 — F56: the update node is not the optimizer — it is a full `torch.cuda.synchronize()` that absorbs the whole iteration's cross-rank wait, and it forbids overlap between iterations

### Tested

`_update` (`executors.py`) split under `PIPER_TIME_TRACE` into its parts:
draining deferred frees, the per-bucket `optimizer.step()` loop, and the
`torch.cuda.synchronize()` that ends it. Ring model, four B200s, steady state.

### Result

| part | CP=4, four ranks | CP=1, one rank |
|---|---|---|
| drain deferred frees | 2–3 us | 2–3 us |
| per-bucket loop incl. event waits | 254–274 us | 292–432 us |
| **`optimizer.step()`, 2 buckets** | **191–212 us** | **226–350 us** |
| **final `torch.cuda.synchronize()`** | **2788–16934 us** | **2257–2490 us** |

The optimizer is a fifth of a millisecond. The node costs 11.5 ms because it
ends with a full device synchronize.

### Per-rank, same step

| step | r0 | r1 | r2 | r3 | spread |
|---|---|---|---|---|---|
| 8 | 6786 | 8836 | 7178 | 9860 | 3074 us |
| 10 | 11176 | 14533 | 16771 | 11896 | 5595 |
| 11 | 16523 | 15907 | 10702 | 16934 | 6232 |
| 13 | 16798 | 15649 | 11555 | 16404 | 5243 |

Ranks in the *same iteration* differ by 2.9–6.2 ms, and the value swings by 6x
across iterations, while the optimizer next to it is stable to 10%. With one
rank and no collectives the sync is 2.3–2.5 ms and steady. That is the
signature of waiting on peers: the collectives' NCCL kernels cannot retire
until the other ranks arrive, and the device sync is where that bill is
presented.

### What it corrects

**F52's headline.** F52 measured 0.10–0.13 ms of GPU drain after the last
enqueue and concluded that by the time the host has issued every node the GPU
has finished. The drain is small because it already happened — inside the
update node, which is not last in dispatch order. The correct statement is that
the iteration is host dispatch *up to* the update node, then a blocking device
sync that absorbs everything outstanding, then a short tail.

**F55's reading of the update node.** F55 recorded "update 1974 us ... real work
by any measure" and left it. It is 210 us of real work and the rest is a wait.

### What it means

`torch.cuda.synchronize()` is device-wide: it waits on every stream, not just
the one carrying the losses it is there to materialize. Three consequences,
in increasing order of interest:

1. **The update node's cost is not attributable to the update.** Any profile
   that reads the DAG's per-node times will mis-attribute the iteration's
   cross-rank wait to the optimizer.
2. **It is where the additive law's skew term is actually paid.** F48/F49
   showed each collective pays `transport + skew`; in Piper most of that skew
   is deferred to this one point, because the collectives are enqueued and only
   the sync forces them to retire.
3. **Piper cannot overlap anything across iterations.** A device-wide sync at
   the end of every step means iteration `n+1`'s first kernel cannot be issued
   until iteration `n`'s last one has retired on every stream. The multi-stream
   design the `stream` directive exists to exploit is confined within one
   iteration by construction. That is a plausible part of why `stream` measured
   1.00x at one microbatch and 1.06x at four.

### The fix, and why it is not free

The sync exists so `_drain_losses` can read loss values on the host. Waiting on
an event recorded after the loss-producing backward, rather than on the device,
would materialize the same values without stalling other streams; returning the
loss tensors and synchronizing once in the driver would do better still. Neither
is a one-liner: `drain_pending_frees` and the ZeRO path also rely on the sync's
ordering, and the second `torch.cuda.synchronize()` in
`step_zero_shard_optimizers` would need the same treatment. Not attempted here;
the measurement is what this entry contributes.

### Method note

This was found by splitting a node that three previous entries had treated as
atomic, after F55 flagged it as the largest single node and moved on. The
project's own rule — measure the parts before naming the whole — had not been
applied to the one node whose name made its cost sound self-explanatory.

## 2026-09-14 — F57: `backward` costs 79 us per call plus 21 us per op, which predicts the node times from the op counts — so there is no separate Piper tax inside the compute nodes

### Tested

Two measurements that meet in the middle.

**The engine's cost function.** `experiments/probe_autograd_fixed.py` backwards
chains of 1–32 elementwise ops, interleaved, 30 rounds, spread reported. On
H200 (the clean host this round; B200's spreads were 333–421 us and its fit is
not usable):

    backward(n ops) = 79 us + 20.9 us/op

Validated against the thing Piper actually does — the same 32 ops backwarded in
one call versus in K calls of 32/K, predicted as `79*K + 20.9*32`:

| segments | predicted | measured |
|---|---|---|
| 1 | 749 us | 748 |
| 2 | 828 | 852 |
| 4 | 987 | 1008 |
| 8 | 1305 | 1288 |

Within 3% at every point.

**The op counts.** Dumping each segment's GraphModule for this model:

| segment | fx ops | boundary outputs |
|---|---|---|
| s0.seg0 prologue | 10 | 6 |
| s0.seg1–4, one per CP step | 17 each | 6 |
| s0.seg5 epilogue | 5 | 1 |
| **total forward** | **83** | |

### The two meet

A 17-op forward segment has roughly 30–35 backward ops (each `matmul` becomes
two, `exp`/`mul`/`sub` one each, plus the accumulations). At the measured rate
that is `79 + 21*35 ≈ 814 us`. **Piper's measured backward node is 822 us.**
The forward node at 249 us over 17 ops is 14.6 us/op, the eager launch rate for
these shapes.

So the compute nodes cost what their op counts say they should. Combined with
F55 (the executor's wrapper around a compute node is 66–106 us) and F56 (the
update node is a device sync, not the optimizer), **every large item in a Piper
iteration is now accounted for by a measured constant times a counted
quantity**, and none of the accounts needs a term for "Piper overhead" inside a
compute node.

### Which also kills the tidiest available explanation

The obvious story after F54 was that Piper pays the autograd engine's fixed cost
once per segment instead of once per step. It does, and that cost is 79 us, so
six segments cost `5 * 79 = 395 us` per iteration — real, and roughly 3% of the
iteration. Not the segmentation tax. The tax F54 measured at 1.5–2.0x is
elsewhere, and F57's arithmetic says where it is not.

### What is left, stated precisely

Piper's compute nodes total 6.80 ms (F55) for arithmetic the straight reference
does in 2.05 ms (F54). The op counts differ by much less than 3.3x — 83 forward
ops against roughly 65 for the reference — so most of the gap is a **per-op
rate** difference, about 24 us/op against 12. Candidates, none isolated: the
`with torch.cuda.stream(...)` entered per node, the argument marshalling through
the bucket closure, the detach at every boundary output (six tensors per
boundary in this model, of which three are values the segment only forwards),
and saved-tensor lifetimes that cross a detached boundary. Distinguishing them
needs a profiler at the operator level, not more arithmetic.

One of those candidates is cheap to test and worth recording as a suggestion:
three of the six tensors crossing each boundary here are *forwarded*, not
produced — the flag G-1a already records — and a forwarded tensor does not need
its own detach-and-reattach, because it is the same value on both sides. That
is 15 of the model's 30 boundary detaches, and their backwards.

### Method note

The two halves of this entry were measured independently — an engine microbenchmark
on one host and an FX op count with no GPU at all — and they agree to 1% on a
number neither was fitted to. That is the only kind of agreement in this project
that was not arranged.

## 2026-09-14 — F58: forwarded boundary outputs no longer get a shadow leaf — correct, verified six ways, and with no measurable benefit

### The change

`ComputeExecutor.forward` detached every boundary output that required grad. For
an output the segment *forwards* rather than produces — the value is the very
object the segment received — that builds a shadow leaf whose only job is to
receive the downstream gradient and hand it to the real leaf one autograd call
later. `boundary_outputs` (extracted to module level for testing) now emits the
input itself in that case, decided by object identity so it cannot drift from
what the GraphModule actually returned.

The change is only correct in pairs. The plumbing is: the successor's
`inp_grads` are assigned positionally onto the producer's
`detached_outs[j].grad`, and `compute.backward` drives
`(pre_detach_outs[j], detached_outs[j].grad)`. With the shadow leaf gone,
`pre_detach_outs[j] is detached_outs[j]` is the segment's own input leaf, and
driving that pair would accumulate the downstream gradient **a second time** on
a tensor that already holds it. So `backward` now skips pairs where `p is d`;
the value is already where `inp_grads` reads it, and the produced outputs'
backward accumulates the in-segment contribution on top. Same sum, one tensor
and one graph entry fewer per forwarded value.

For this model that is 3 of the 6 tensors crossing each of 5 boundaries: 15
fewer detaches and 15 fewer backward pairs per iteration.

### Verification

Every check has a negative control, and the in-Piper ones span optimizer steps
on purpose: a gradient counted once too often or too few leaves iteration 0
correct and diverges from iteration 1 (the F10 pattern).

| check | result | control |
|---|---|---|
| out-of-band TP math (torchrun) | output 1.49e-08, input grad 2.21e-09 | dropping the collectives changes it |
| out-of-band CP math (torchrun) | out 4.17e-07, dQ/dK/dV 1.2–2.9e-06 | no-rotation and forward-only-rotation both break |
| in-Piper TP=2 vs TP=1, 3 steps | worst 7.15e-07, both ranks agree | no directive: 1.103 |
| in-Piper CP=2 vs dense, 3 steps | worst 1.729e-06 | no ring: 3.220e-03 |
| in-Piper CP=4 spliced, 5 steps | worst 1.293e-05 | — |
| in-Piper CP=4 hoisted, 5 steps | worst 1.293e-05, identical to spliced | — |
| CPU suite | 82 passed (78 before, +4 for the new rule) | — |

All of it re-run after the helper was extracted, because moving code is exactly
the kind of "should be identical" this project keeps being punished for.

### The measurement, and it is negative

Interleaved before/after, five reps each, four H200s at under 25% utilization,
swapping the executor between harness launches:

| | before min | after min | delta | within-arm range |
|---|---|---|---|---|
| iteration | 13.09 ms | 12.80 ms | −0.29 | 13.1–20.8 / 12.8–18.0 |
| forward node | 338 us | **454 us** | +116 | — |
| backward node | 913 us | 769 us | −144 | — |

F57 predicted 0.2–0.5 ms per iteration from 15 fewer detaches at ~15 us and 15
fewer backward entries at ~21 us. The measured |delta| is at most 0.35 ms
against a within-arm range of 5–8 ms — a twentieth of the noise. And the forward
node moved *up* while the backward moved down, which a change that removes work
from both cannot do. **Not resolvable.** The prediction is neither confirmed nor
refuted; the experiment lacks the resolution, on the quietest cards available.

### Kept, and why that is a judgement not a measurement

It removes work that is provably redundant, it is verified, and the saving
scales with boundaries times forwarded tensors — a 32-layer model under CP=8
crosses far more of both than this one. Against that: it adds an invariant to a
correctness-critical path, which is why the rule is now a module-level function
with four CPU tests including one that checks a gradient is counted exactly once
end to end. If the invariant is ever broken the tests fail rather than the loss
curve diverging on the second optimizer step.

It would be equally defensible to revert it. The honest summary is: correct,
free of measured benefit, kept for a scaling argument that has not been tested
at scale.

### Found on the way: the out-of-band TP gate had rotted

It failed with `KeyError: 'pre'` and does not import `src` at all, so the change
could not have caused it. `tp_mlp.global_weights` grew an `n_stages` parameter
and moved to `blocks.N.*` naming; the gate kept indexing short names. It carries
the `gpu` mark, so the CPU suite deselects it, and every chain since had run the
*in-Piper* TP check instead — so it broke silently and stayed broken. Fixed by
deriving short names from the module path with an assertion that fails loudly if
a second block ever makes them ambiguous.

That is the same failure mode this project has been documenting in Piper —
something with no observer stops being true — occurring in this project's own
test suite. The lesson recorded: a gpu-marked test that nothing in the default
path runs needs a scheduled run, or it is documentation.

## 2026-09-14 — F59: an iteration is 38% host dispatch, 44% blocked in the device sync, 10% idle waiting for the driver — so neither the sync nor Ray is the single binding constraint

### Method

No new run. Every `PIPER_TIME_TRACE` file records absolute `perf_counter`
values, so step *n*'s `__end__` and step *n+1*'s first node give the gap in
which the actor is doing nothing, and F56's `__upd_final_sync__` gives the part
of the in-actor span where the host is blocked rather than dispatching. Median
over four ranks and fourteen steps (the max column is the warm-up compile and is
why the median is the statistic).

### Result, on the instrumented four-rank run

| | ms | share |
|---|---|---|
| host actually dispatching | **8.58** | 38% |
| host blocked in the update node's `torch.cuda.synchronize()` | **9.98** | 44% |
| actor idle between `run_dag` calls — Ray round trip and driver | **2.35** | 10% |
| iteration | **22.55** | |

The idle gap is 1.43–4.32 ms across all ranks and steps; it never goes away.

### This corrects two earlier entries

**F52 said the Ray round trip "is not measurable against" the dispatch loop.**
It is 1.4–2.4 ms per iteration, about a tenth of the step. F52 computed it as
`driver_iteration - (dispatch + drain)` using the *minimum* driver iteration
against a *per-step* dispatch, so the subtraction went negative and was read as
zero. Measuring the gap between consecutive traces, rather than subtracting two
statistics taken differently, gives the number directly.

**F56 called the device sync "the consequence worth acting on" for
cross-iteration overlap.** Reading `piper_exec_dag` settles it: the driver does
`run_refs = [actor.run_dag.remote(...) for ...]` then `ray.get(run_refs)`, one
synchronous round trip per step. So even with the sync gone the actor would
return and sit idle until the driver issued the next step. The sync forbids
overlap, but it is not the binding constraint by itself — **both** it and the
driver's synchronous step loop have to change before the host can dispatch
iteration *n+1* while the GPU finishes *n*.

That is why the change F56 pointed at was not attempted: on inspection it would
have bought nothing on its own, and the measurement above is the reason rather
than the excuse.

### The ceiling, stated properly

With both removed, an iteration is bounded by `max(host dispatch, GPU work)`.
Host dispatch is 8.58 ms against today's 22.55 — a **2.6x ceiling on a loaded
machine**. On a quiet one the sync term collapses (it is mostly cross-rank
waiting, F56) and the ceiling falls to roughly the 1.4–2.4 ms of idle, about
1.15x. So the value of pipelining is a function of how contended the machine
is, which is the same dependence F47 found for memory and F51 for the runtime
constant, and it means a benchmark of this change would report whatever the
machine was doing that day.

### What would actually be worth building

Not the sync removal alone. The pair: return the loss as a device tensor and
convert it at the top of the next step, and have the driver keep two steps in
flight. That is a change to `piper_exec_dag` and `_update` together, it changes
the observable contract (losses lag by one step), and its benefit is between
1.15x and 2.6x depending on the machine. Recorded as specified rather than
built; the three numbers above are what a decision would need.

## 2026-09-14 — F60: the host-side wait at the end of a step becomes a knob, `narrow` becomes the default, and `defer` is 1.25–1.38x

### What was built

`PIPER_SYNC_MODE`, three ways for the host to wait at the end of `_update`:

* **device** — `torch.cuda.synchronize()`, upstream behaviour, kept so old
  timings reproduce exactly;
* **narrow** — wait only on the events that mark the loss values as ready.
  The optimizer and the trailing collectives stay in flight and the host
  returns early, so Ray's round trip and the next step's dispatch overlap this
  step's GPU tail. **Nothing any caller observes changes.** Now the default.
* **defer** — no host wait at all. The step records its completion as an event
  the next step's streams queue behind, and hands its loss tensors to that step
  to convert. Fastest and steadiest; losses lag one step, so a caller must call
  `piper_flush_losses()` when its loop ends.

F59 specified this as a change to `piper_exec_dag` *and* `_update`, with the
driver keeping two steps in flight. That was over-specified: the driver needs no
change at all. Once the actor stops blocking, `ray.get` returns early and the
round trip overlaps the GPU on its own. The constraint was never "the host must
not wait" but "the host must not wait on *the whole device*".

### Coverage

Every cell is losses compared across the three modes; the TP and CP cells also
carry negative controls (removing the directive moves the loss by 1.103 and
3.2e-03) and span optimizer steps, where a gradient counted wrong leaves
iteration 0 correct and diverges from iteration 1.

| configuration | B200 | H200 |
|---|---|---|
| TP=2 vs TP=1, 3 steps | 7.15e-07, all modes | 5.96e-07, all modes |
| CP=2 vs dense, 3 steps | 1.729e-06, all modes | 1.729e-06, all modes |
| CP=4, 5 steps | 1.293e-05, all modes | 1.292e-05, all modes |
| TP x PP, four cards, 1f1b | 3.81e-05, all modes | 1.56e-04, all modes |
| EP, shipped Qwen, pp2_dp2_ep2 | identical, all modes | identical, all modes |
| ZeRO-3, three stages, shard_params | identical, all modes | identical, all modes |
| out-of-band TP and CP math | unchanged | unchanged |
| CPU suite | 82 passed | 82 passed |

Two of those cells only exist because of holes found while testing, not by
testing:

* **ZeRO-3 bypassed the modes entirely.** `_update` returns inside the
  `has_zero_shard_optimizers()` branch, *before* the sync-mode code, so ZeRO-3
  kept the device-wide block whatever the variable said. Without reading that
  branch the matrix would have shown "ZeRO-3: all modes agree" while ZeRO-3 had
  never run a new path. Now that branch has the three modes too.
* **`narrow` was unsafe on any rank that computes no loss** — every non-final PP
  stage — because its wait is over the loss events and there are none. It would
  have returned immediately and the next step's `clear_param_grads` would free
  buffers the optimizer was still reading. The completion event and the
  next-step stream waits now apply to `narrow` as well as `defer`, and the
  TP x PP cell is there to prove it.

And one hazard was fixed rather than documented: `clear_param_grads` now takes
the in-flight streams and calls `record_stream` on each gradient before dropping
the host's reference. Freeing returns a block to the pool of the stream it was
allocated on, which is only safe to reuse *on that stream*; a reader on any
other stream needs this, or the allocator can hand the block to a new kernel
while the optimizer is still in it.

### Timing, interleaved, four reps, both machines

`final_sync` is the host's own block, measured directly:

| | B200 (six of seven cards busy) | H200 (four cards idle) |
|---|---|---|
| device | 13.14 ms min, sync 9790–20461 us | 17.59 ms min, sync 2241–7431 us |
| narrow | 10.10 ms min, sync **1441–6242 us** | 14.05 ms min, sync **11–254 us** |
| defer | 10.24 ms min, sync **8–9 us** | 14.23 ms min, sync **11–12 us** |

By median: device 15.9 / narrow 14.5 / defer **11.5** on B200; 1.38x. On H200
the ordering holds at 1.25x. `defer` is also far steadier (10.2–12.9 against
`narrow`'s 10.1–17.6).

### A conclusion the second machine reversed

Measured on H200 alone, `narrow` and `defer` were indistinguishable and the
entry would have read "`defer` is over-engineering; `narrow` gets the whole
benefit". B200 shows `narrow` still blocking 1441–6242 us: on a contended
machine the loss is *not* ready when `_update` runs. The quiet machine hid the
difference. Third time in this project that running on both machines reversed a
conclusion (F47, F51, here).

### The default, decided

**`narrow`.** It is contract-identical to `device` — no lag, no flush, no change
to any example — it is verified across six configurations on two machines with
negative controls, and it is faster on both by median and by minimum.
`PIPER_SYNC_MODE=device` restores the old behaviour byte for byte, and
`defer` is there for callers who will take a one-step lag for another ~10% and a
third of the variance.

Choosing `device` as the default would have been the cautious call and it is not
the better one: `narrow` differs from it only in *which* stream the host waits
on, and every observable in the matrix above is identical.

### What the EP cell does and does not prove

It confirms the three modes agree on the shipped Qwen example, but its loss is
bf16 and constant at 7.625 across all six iterations, so it would catch a gross
corruption and not a 1e-3 one. It is the weakest cell in the matrix. Also: that
example reported no loss at all until this entry — `_raw_metrics` carried time
and memory only, and `piper_exec_dag`'s return value was discarded — so there
was nothing to compare two EP runs against. F8's finding, still true in the one
example this project had not touched. It now collects losses and writes
`qwen_metrics_dp*.json` like the other two examples.

## 2026-09-15 — F61: the sync-mode win shrinks with scale, not grows — the 2.83x was a clock that stopped before the GPU did

### The wrong number, and how it was made

The scale sweep first reported `defer` beating `device` by 1.27x, 1.65x and
**2.83x** as the model grew, i.e. the benefit *growing* with problem size. The
opposite was predicted (a bigger model is GPU-bound, so a host-side wait should
matter less), and the prediction was right.

`defer` is defined by the host not waiting for the device. So
`piper_exec_dag` returns while the GPU is still working, the per-iteration timer
stops early, and `min(iter_times_s)` selects whichever iteration the host ran
furthest ahead on. The same run's own numbers give it away: at seq=8192 `defer`
reported 12.91 ms per iteration while the whole timed loop, drained, took
31.82 ms per iteration — the host was about 2.5 steps ahead.

Fixed by measuring the wall time of the entire timed loop with
`torch.cuda.synchronize()` on every rank at the end (`actor.drain()`,
`total_timed_s`). That metric is fair to all three modes because it contains
the GPU tail.

### The honest curve

Drained wall time per iteration, minimum of three interleaved reps, four B200s:

| seq (s_local) | device | narrow | defer | best vs device |
|---|---|---|---|---|
| 1024 (256) | 20.15 ms | **13.43** | 14.61 | 1.50x |
| 2048 (512) | 15.34 | **12.25** | 14.19 | 1.25x |
| 4096 (1024) | 21.34 | 20.83 | **18.75** | 1.14x |
| 8192 (2048) | 34.00 | 35.14 | **31.82** | 1.07x |

The win falls from 1.50x to 1.07x as the step becomes GPU-bound, which is what
F50's criterion says it must: a knob that removes host-side waiting is worth
`waiting / iteration`, and that ratio collapses as the arithmetic grows. The
host block itself (`final_sync`) behaves as designed at every size — 8–24 ms for
`device`, 0.8–18 ms for `narrow`, and 6–9 **us** for `defer` throughout.

`narrow` wins at the small end and `defer` at the large end, which follows from
the same mechanism: `narrow` waits for the loss, and the bigger the model the
later the loss is ready.

### What this changes about F60's claim

F60 reported 1.25x on a quiet machine and 1.38x on a loaded one, both at
s_local=256 — the host-bound end of this curve. That is where the number came
from, and it should be quoted with its condition. The defensible statement is:
**the sync mode is worth up to 1.5x while the step is host-bound and about 1.07x
once it is not.** The default (`narrow`) is unaffected: it is contract-identical
and never slower than `device` by more than noise at any size measured.

### Why this is the same failure this project keeps finding

An instrument that reports a number for a configuration it cannot actually
measure. Piper's shipped examples reported throughput and memory while training
on zero inputs (F9) because nothing they printed depended on the arithmetic. A
per-iteration timer reports a time for a mode whose defining property is that
the iteration is not over when the timer stops. In both cases the output is
well-formed, plausible, and about something other than what it claims.

Two instruments were added rather than one, because the same audit found the
other: the EP cell of F60's matrix compared a **bf16 loss on inputs drawn once
and reused**, which is 7.625 whatever the model computed. Three modes agreeing
on it agreed on nothing. `actor.param_checksum()` now returns an fp64 sum and
sum-of-squares over every trainable parameter — every gradient ever applied is
in there — and all three examples record it.

### Retained

`logs/scale_b200_BOGUS_per_iter.txt` keeps the wrong run rather than deleting
it, per the project's rule on retractions.

## 2026-09-15 — F62: the canonical machine's CUDA runtime failed, and idle utilization was the wrong signal for it

### What happened

`nvidia-smi` on catalyst-fleet1 showed four cards at 0% with 178\,GB free each —
the quiet window this project had waited for since F36, where a strict
whole-machine-quiet waiter ran eleven hours without firing. A clean re-validation
of the paper's headline directive numbers was launched into it and every run
returned `NA`.

The cause is one line of `nvidia-smi` output that was scrolled past:

    Unable to determine the device handle for GPU2: 0000:52:00.0: Unknown Error

With that device in a bad state the whole CUDA runtime is unusable on the host:

    CUDA_VISIBLE_DEVICES=3,4,5,6 → torch.cuda.device_count() == 0
    CUDA_VISIBLE_DEVICES=0,1     → 0
    by UUID, skipping GPU2       → 0
    CUDA_DEVICE_ORDER=PCI_BUS_ID → 0

`_join_process_groups` then divides by `device_count` and raises
`ZeroDivisionError`, which is how it surfaced. Recovering needs a driver reset
and root, which this project does not have.

**The cards were idle because nobody could use them.** Utilization was not just
a weak signal here, it was an inverted one.

### The check that should have been made, and now is

Availability is `torch.cuda.device_count() > 0`, not `utilization == 0`. Every
gate script in `experiments/` polls utilization and free memory; on this host,
in this state, all of them would report a perfect window forever. A one-line
CUDA probe in the gate would have failed in two seconds instead of producing
twenty `NA` rows over sixteen minutes.

### Consequence for the project

The B200 half of the two-machine method (F47, F51, F60) is unavailable until
someone resets that host. Everything it produced is preserved: `results/b200/`
holds the raw measurements, and the repository is byte-identical on the H200
host, which is healthy (`device_count == 8`). Work continues there.

What is lost is the specific thing the window was for: the paper's directive
numbers — `order` worth 16% under PP=2, fusion 12–19% on a single stage,
`stream` 1.06x at four microbatches — were all measured under contention, and
re-validating them on a quiet machine is still owed. H200 does not currently
have four quiet cards either. The claim stands as originally measured, with the
contention noted in the paper, and the re-validation stays open.

## 2026-09-16 — F63: on a quiet machine the `order` claim does not reproduce, and its sign reverses

### Tested

The paper's four directive numbers, re-measured on catalyst-fleet1 during the
first genuinely idle window this project has seen (F36 recorded a strict waiter
running eleven hours without firing; F62 recorded a false window caused by a
dead CUDA runtime). Drained clock (F61), arms interleaved, five reps, minimum
per arm.

| claim as published | re-measured | verdict |
|---|---|---|
| `order`: 1F1B beats GPipe by **16%** (PP=2) | GPipe 11.44\,ms vs 1F1B 13.33 — **GPipe faster by 16.5%** | **reversed** |
| fusion under 1F1B: indistinguishable | 12.93 vs 13.01 | confirmed |
| fusion on a single stage: 12–19% | 8.08 vs 9.19 — 13.7% | confirmed |
| microbatch 1→8 at fixed work: 2.6x worse | 7.45 → 28.34 — 3.8x | same direction, larger |

### Ruling out our own change

Three things differed from the original measurement: the machine's load, the
drained clock, and this project's own change of the default host-sync mode
(F60). The last is the one that would invalidate the comparison, so the
`order` pair was rerun under both sync modes and read by both metrics:

| | 1F1B | GPipe |
|---|---|---|
| `sync=device` (upstream), drained | 14.56\,ms | **12.37** |
| `sync=device`, per-iteration min (the original metric) | 11.57 | **10.28** |
| `sync=narrow` (new default), drained | 12.71 | **12.58** |
| `sync=narrow`, per-iteration min | 10.28 | **9.64** |

GPipe is never slower in any of the four combinations. **The reversal is not
ours.**

### Mechanism, and why it makes the paper's thesis sharper rather than weaker

1F1B exists to fill pipeline bubbles. F50 established that at this model size
the iteration is host-dispatch-bound — the GPU finishes what it is given and
waits — so there are no GPU bubbles to fill, and 1F1B's extra ordering
constraints are pure additional host work. Contention on the original
measurement's machine would have slowed the GPU relative to the host, moving
the step toward GPU-bound, which is the regime where 1F1B's bubble-filling
pays.

So the published claim and this one are both true, of different machines. The
paper argues that a directive's value depends on the structure of the schedule;
this says the same directive, in the same schedule, on the same code, **changes
sign with how busy the machine is**. That is a stronger statement of the same
thesis and an uncomfortable one for any static search over schedules: the
quantity being optimised is not a property of the program.

### What this does to the auto-TP conclusion

The project's limitation section says automatic TP selection must model the
runtime rather than the hardware, and F48 refined the reason: cost is
`a + b·bytes + skew`, with the skew term belonging to the runtime. F63 adds the
term that is not even that: **the sign of a directive's benefit depends on
contention from other tenants**, which no model of this program can see. A
search run on an idle machine would pick GPipe here and be wrong on a busy one.

### Confidence, stated

The window closed before a confirmation run on four *fully* idle cards — one of
the four cards in the reported run was at 100% from another tenant, and the
spread within an arm is 12.4–18.0\,ms. What is solid: GPipe is not slower in any
of four measurement combinations, and the published 16% for 1F1B does not
reproduce. What is not yet solid: the exact magnitude. A waiter is queued to
rerun on four idle cards if a window opens.

### Confirmation, 2026-09-16: six independent comparisons, GPipe never slower

The magnitude F63 left open is still open; the direction is not. Across two
clean-window sessions on different card sets, two sync modes and both metrics:

| session | cards | mode / metric | 1F1B | GPipe |
|---|---|---|---|---|
| revalidation 1 | 0,1,2,4 | drained | 13.33 | **11.44** |
| isolation 1 | 0,1,2,4 | device, drained | 14.56 | **12.37** |
| isolation 1 | 0,1,2,4 | narrow, drained | 12.71 | **12.58** |
| isolation 2 | 0,1,2,3 | device, drained | 11.23 | **10.45** |
| isolation 2 | 0,1,2,3 | narrow, drained | 12.17 | **10.23** |
| revalidation 2 | 0,1,2,3 | drained | 10.95 | **10.80** |

GPipe is not slower in any of the six, by 1.4% to 19%. Read by the metric the
original claim used — the minimum per-iteration time — the same holds: 9.58
against 8.32 under `device`, 10.50 against 8.52 under `narrow`.

**The published 16% for 1F1B did not reproduce in any measurement taken on this
machine over two days.** What remains undetermined is how much faster GPipe is:
the within-arm spread is 11.2–22.3 ms, and in every session another tenant
appeared on one of the four cards partway through. The direction is supported by
six comparisons; the magnitude by none of them individually.

The other three claims, from the same two sessions:

| claim as published | session 1 | session 2 | verdict |
|---|---|---|---|
| fusion under 1F1B: indistinguishable | 13.01 vs 12.93 | 14.56 vs 14.45 | confirmed twice |
| fusion single-stage: 12–19% | 13.7% | 28% | direction confirmed, magnitude wider |
| microbatch 1→8: 2.6x worse | 3.8x | 4.6x | direction confirmed, magnitude larger |

So of four published numbers: one confirmed exactly, two confirmed in direction
with larger magnitudes, and one that does not reproduce and whose sign is
reversed in every measurement.

## 2026-09-16 — F64: the selection prototype ranks correctly once it carries the terms this project measured

### What changed

`experiments/tp_search.py` was written at Stage F with constants fitted from
F11–F13: a peak-FLOPS figure, a single all-reduce bandwidth, an overlap fraction
and a per-node dispatch cost. Its own ranking check reported that it could pick
the fastest configuration but could not order the rest — "treat it as a filter,
not a ranking".

Three days of measurement supplied the terms it was missing. The communication
model is now the one F48/F51 measured rather than a bandwidth:

    per collective:  fixed + per-MiB·bytes + host-dispatch + skew
      fixed          41.7 us (B200) / 44.5 us (H200)      -- software
      per MiB        2.49 us (B200) / 4.51 us (H200)      -- fabric
      host dispatch  ~150 us per tensor touched (F55)
      skew           ~200 us, additive not max (F49)

### Result

| | before | after |
|---|---|---|
| picks the fastest configuration | yes | yes |
| orders the rest | **no** | **yes** |
| predicted order | — | `[one GPU, pp=2, tp=2]` |
| measured order | — | `[one GPU, pp=2, tp=2]` |
| per-configuration error | 1.00x / 1.13x / 0.99x | same |
| calibration check (in-sample) | 2% | 4% |

The out-of-sample ranking went from wrong to right while the in-sample fit got
slightly worse, which is the trade one wants: the old model fitted its
calibration shape by absorbing communication into a bandwidth that did not
generalise.

This closes the question the project opened with. The limitation section said a
roofline "ranked three measured configurations backwards" and concluded that
automatic selection must model the runtime rather than the hardware. It does —
and the runtime terms are three numbers that one sweep measures.

### Why the ranking transfers between machines, and why that is not reassuring

Run with each host's constants, the two produce **identical rankings**. Not
because the model is robust: at TP payloads the fixed, host-dispatch and skew
terms dominate, and those are software constants that differ by 7% between
machines, while the per-byte term that differs by 1.81x barely participates.

So the ranking transfers across hardware for the same reason F51 found the
intercept transfers — the deciding quantities are properties of the software
stack. The one thing that would break it is contention, and that is precisely
what `--limits` now says it cannot see:

> Contention flips signs, not just magnitudes. On a quiet machine GPipe beat
> 1F1B by 16.5% on the very configuration where the contended measurement had
> 1F1B ahead by 16% (F63). Nothing in this program can see how busy the machine
> will be.

### Status against the original goals

The brief asked for four things: understand the IR/compiler/runtime boundaries;
add TP; start from the smallest correct two-GPU example; and only afterwards
explore a very lightweight automatic TP-selection prototype. The fourth existed
from Stage F but carried a cost model that predated everything measured since.
It now carries it, ranks correctly out of sample, and states its own blind spot
rather than leaving it to be found.

---

## 2026-09-23 — F65: a placement lattice with device permutation and device group predicts all 66 communication nodes in four lowerings; plain SPMD misses the ring hops and the stage crossings

### Tested

`experiments/probe_placement.py`, CPU only. Four lowerings (TP=2, CP=2 with DP, ZeRO-3 over
three stages, TP=2 x PP=2 under 1F1B), scored on the per-rank DAGs that execute. Communication
nodes are spliced out first; each remaining compute-to-compute edge gets a prediction from the
tags of its two ends, the forward segment's own boundary record, and what the schedule
declares. The kind of a spliced node is never read. Parameter-side nodes are predicted from the
ZeRO and replication flags, with ZeRO backward gathers conditioned on the forward's
free-after flag.

### Predictions, written in the probe's docstring before the run

TP, DP and ZeRO predicted by plain SPMD; CP rings missed by SPMD and predicted once S carries a
device permutation; stage crossings predicted only once placement carries a device group; ZeRO
backward gathers follow the free-after flag.

### Result

| lattice | TP=2 | CP=2 + DP | ZeRO-3 | TP=2 x PP=2 |
|---|---|---|---|---|
| SPMD (R, S, P) | 2/2 | 2/6, ring hops missed | 26/26 | 16/32, stage crossings missed |
| + device permutation | 2/2 | 6/6 | 26/26 | 16/32 |
| + permutation + device group | 2/2 | 6/6 | 26/26 | 32/32 |

The 45 compute edges carrying no communication are predicted empty under every lattice; no
lattice predicts a collective that is not there.

### Three defects in the first version, fixed before these numbers

The first draft assigned each node's placement from the node's own kind, which makes agreement
circular; it was rewritten to predict from the compute ends only. The TP region was read from
the tag, which the MLP carries even at tp=1; it is now read from the `shard_tensor` directive.
Stage crossings were read from the PP index; three ZeRO stages share one device set and need no
transfer, so the group is now the device set. Scoring the global DAG over-counted one ZeRO
gather that the per-stage lifetime pass prunes, and missed every send/recv, which have no edge
between them; scoring now uses the executed per-rank DAGs with each send paired to its recv.

### What this says about Direction A

The lattice Piper needs is layout x device assignment x device group, with buffer lifetime
supplied by the schedule. With a permutation in the layout, a ring hop is a collective permute
and is determined by placement; what remains a schedule choice is expressing an all-gather as
n-1 permutes and when each issues.

### Boundary

Placements come from region-level rules written from the declared layouts, not from op-level
propagation; the probe shows such placements suffice to predict communication, not that they
can be inferred. EP is not covered (the dump tool cannot trace the Qwen example), tied weights
are not lowered here, and all four models are small.

---

## 2026-09-23 — F66: two independent branches lower as a chain; the frontend threads every crossing value through the segments in between, the text encoder's own input included

### Tested

`experiments/probe_branches.py`, CPU only. Two encoders that share no data feed one decoder;
each is its own PP scope placed on its own device (0, 1, 2). The training DAG is inspected
before the per-stage split.

### Prediction, written in the probe's docstring before the run

Chained: contiguous segments plus threading of crossing values would add a stage 0 -> stage 1
data edge the model does not have.

### Result

| check | result |
|---|---|
| image encoder reaches text encoder along data edges | yes |
| text encoder's own input `x_txt` forwarded through the image encoder's segment | yes |
| image encoder's output `a` forwarded through the text encoder's segment | yes |
| forward sends | stage 0 -> 1 and 1 -> 2 only; none from 0 to 2 |

The text encoder cannot start until the image encoder has finished and forwarded its input.
The ordering is created by segmentation, so no `order` directive can remove it. The per-stage
split then refuses the schedule for want of an `order` directive, as F14 requires for pp > 1;
the chain is already in the data edges before that point.

The prediction held and understated the effect: the independent branch's own input is threaded
too, not only the other branch's output.

### The same rule, seen twice

Threading a crossing value through the segments in between is what makes a CP ring's payload
ready at segment start (F58, the hoist). The rule that lets the ring issue early is the rule
that chains the branches. A DAG-shaped schedule needs the frontend to route a crossing value
from its producer to its consumers directly, which would also change where a forwarded tensor
is visible to a ring.

### Probe bug fixed before the verdict

Reachability first ignored the cross-device edges, because a send and its recv carry no edge
between them; the first run printed "parallel" beside boundary records that showed the chain.
Sends are now paired to recvs by index, as in F65.
