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
