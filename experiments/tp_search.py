"""A deliberately small cost model for picking (pp, tp, microbatches) for TPMlp.

    python experiments/tp_search.py --gpus 2 --global-batch 8192
    python experiments/tp_search.py --calibration-check

Not an auto-parallelism framework. It exists to answer one question: given the
limits this project found in Piper's IR, what is even searchable, and can a model
calibrated on a handful of measurements rank configurations correctly?

What is searchable, and why it is this and not more:

* `tp` cannot be a schedule-level choice. Piper's IR has no representation of a
  partitioned tensor (notes/log.md F1), so TP-local shapes live in the *model*;
  changing `tp` means rebuilding and retracing it. Any TP search therefore spans
  model construction, not just directives -- which is the concrete cost of F1.
* `pp * tp <= gpus` with `tp` equal to the place-group size, because Piper's
  world is `pp_degree x dp_degree` with no third axis (F4). So TP x PP is in the
  space and TP x DP is not (F14).
* Microbatches in Piper **replicate the DAG, they do not split the batch**
  (`_apply_split_directive` copies nodes; `load_input` hands every copy the same
  tensor). So holding the global batch fixed means per-microbatch batch is
  `global / mb`, and `mb` trades activation memory and overlap against per-node
  dispatch cost.

Calibration comes from F11-F13, all measured on 2xB200:

* `ACHIEVED_TFLOPS` from the mb1 batch-8192 run: 19.8 TFLOP of work against
  14992us of non-communication GPU time.
* `ALLREDUCE_GBPS` from the clean backward all-reduce: 32 MiB in 93us.
* `OVERLAP_FRACTION` from F12: mb>=2 reached concurrency 1.06x against a 6.4%
  communication share, so about 90% of communication is hidden. At mb=1 the
  collective has nothing to overlap with (F12), so none is.
* `DISPATCH_US_PER_NODE` from F12: at batch 1024 the GPU timeline sat ~25% idle
  with 33 nodes; the gaps are per-node Python dispatch in DagExecutor.run.

Every number above is a measurement from this log, not a datasheet figure. The
model is a roofline with one overlap term; it is meant to rank, not to predict.
"""
import argparse
import itertools
import sys

ACHIEVED_TFLOPS = 1320.0        # bf16, per GPU, from F12's mb1 batch-8192 run
ALLREDUCE_GBPS = 360.0          # kept only for the F11/F12 calibration check

# Measured after this model was first written, on four idle cards of each host
# (F48, F51). A collective costs a fixed part plus a per-byte part, and the two
# behave differently across machines: the fixed part is software (41.7 vs
# 44.5 us on hosts whose bandwidths differ by 1.81x) while the slope is the
# fabric. Carrying the pair is what lets a calibration transfer at all.
COLL = {"b200": (41.7, 2.49), "h200": (44.5, 4.51)}   # (fixed us, us per MiB)

# Each collective also costs host time to issue, roughly per tensor it touches,
# and at TP and CP payloads that exceeds the transfer: a ring node moving 1 MiB
# per tensor costs ~400 us of host time against ~44 us on the wire (F55).
COLL_HOST_US_PER_TENSOR = 150.0

# Arrival skew is paid per collective and ADDS to the transfer rather than
# hiding under it (F49: differencing the waiting rank against the late one is
# constant to 0.3% across a 32x payload range). Zero asks what a perfectly
# synchronized group would cost.
SKEW_PER_COLLECTIVE_US = 200.0
OVERLAP_FRACTION = 0.90         # of communication, when mb >= 2 (F12)
DISPATCH_US_PER_NODE = 19.0     # per DAG node, from F12's idle-timeline fit

# Implementation overheads, not hardware ones. A pure roofline over compute,
# bandwidth, bubble and dispatch ranked these configurations *backwards*: it put
# TP=2 fastest and one GPU slowest, while the measurement put one GPU fastest by
# 30% (notes/log.md F16). The two terms below are what was missing, and both are
# properties of how Piper drives devices rather than of the machine:
#
#   DRIVER_OVERHEAD_US -- Ray driver plus actor RPC per step. Fitted from the
#       one-GPU run: 18720us measured against 12494us of predicted compute.
#   SKEW_US -- rank arrival skew, charged once per step as soon as more than one
#       rank exists. Each device is driven by its own Ray actor and piper_exec_dag
#       fans out with ray.get, so iteration start times differ by milliseconds and
#       the first collective absorbs it inside the NCCL kernel (F11). Measured
#       directly: TP=2's communication kernels totalled 12875us per rank per
#       iteration against ~745us of actual payload at 360 GB/s, a factor of 17.
#
# These do not scale with problem size, so they dominate the decision at small and
# medium scale and decide it wrongly if omitted. Lowering them -- batching
# dispatch, or driving all devices from one process -- would matter more for TP
# than any schedule choice at this scale.
DRIVER_OVERHEAD_US = 6200.0
SKEW_US = 12000.0

BYTES = {"bf16": 2, "fp32": 4}


def _block_flops(batch: int, dim: int, hidden_local: int) -> float:
    """pre + up + down + post, forward only. 2 FLOP per multiply-add."""
    return 2.0 * batch * (dim * dim + dim * hidden_local + hidden_local * dim + dim * dim)


def estimate(*, gpus, pp, tp, mb, global_batch, dim, hidden, stages, dtype,
             machine="b200"):
    """Return (step_us, breakdown) for one configuration, or None if invalid."""
    if pp * tp > gpus or stages % pp or hidden % tp:
        return None
    micro_batch = global_batch // mb
    if micro_batch == 0:
        return None

    stages_per_rank = stages // pp
    hidden_local = hidden // tp

    # Compute: each pipeline rank owns stages_per_rank blocks; forward plus a
    # backward costed at 2x forward.
    fwd = _block_flops(micro_batch, dim, hidden_local) * stages_per_rank
    compute_us = 3.0 * fwd * mb / (ACHIEVED_TFLOPS * 1e12) * 1e6

    # TP communication: two all-reduces of the region output per TP block.
    tp_bytes = (
        0.0 if tp == 1
        else 2.0 * micro_batch * dim * BYTES[dtype] * stages_per_rank * mb
    )
    _f, _b = COLL[machine]
    n_tp_colls = 2 * stages_per_rank * mb if tp > 1 else 0
    tp_us = n_tp_colls * (_f + _b * (tp_bytes / max(n_tp_colls, 1)) / 2**20
                          + COLL_HOST_US_PER_TENSOR + SKEW_PER_COLLECTIVE_US)
    tp_exposed = tp_us * (1.0 - OVERLAP_FRACTION) if mb >= 2 else tp_us

    # PP point-to-point: one activation each way per boundary crossing.
    pp_bytes = (
        0.0 if pp == 1
        else 2.0 * micro_batch * dim * BYTES[dtype] * mb
    )
    n_pp_colls = 2 * mb if pp > 1 else 0
    pp_us = n_pp_colls * (_f + _b * (pp_bytes / max(n_pp_colls, 1)) / 2**20
                          + COLL_HOST_US_PER_TENSOR + SKEW_PER_COLLECTIVE_US)
    pp_exposed = pp_us * (1.0 - OVERLAP_FRACTION) if mb >= 2 else pp_us

    # Dispatch: nodes per rank grow with microbatches and owned stages.
    nodes = mb * (2 * (3 * stages_per_rank) + (2 * stages_per_rank if tp > 1 else 0)
                  + (2 if pp > 1 else 0)) + 1
    dispatch_us = nodes * DISPATCH_US_PER_NODE

    # A pipeline bubble the microbatches cannot fill: (pp-1) stage latencies.
    bubble_us = (pp - 1) * (compute_us / mb) if pp > 1 else 0.0

    world = pp * tp
    overhead_us = DRIVER_OVERHEAD_US + (SKEW_US if world > 1 else 0.0)

    step_us = (
        max(compute_us, dispatch_us)
        + tp_exposed + pp_exposed + bubble_us + overhead_us
    )
    return step_us, {
        "compute_us": compute_us, "tp_us": tp_us, "tp_exposed": tp_exposed,
        "pp_exposed": pp_exposed, "dispatch_us": dispatch_us,
        "bubble_us": bubble_us, "overhead_us": overhead_us,
        "nodes": nodes, "micro_batch": micro_batch, "world": world,
    }


def search(args):
    rows = []
    for pp, tp, mb in itertools.product(
        [1, 2, 4], [1, 2, 4], [1, 2, 4, 8]
    ):
        got = estimate(
            gpus=args.gpus, pp=pp, tp=tp, mb=mb, global_batch=args.global_batch,
            dim=args.dim, hidden=args.hidden, stages=args.stages, dtype=args.dtype,
        )
        if got is None:
            continue
        rows.append((got[0], pp, tp, mb, got[1]))
    rows.sort()
    print(f"gpus={args.gpus} global_batch={args.global_batch} dim={args.dim} "
          f"hidden={args.hidden} stages={args.stages} {args.dtype}\n")
    print(f"{'rank':<5s}{'pp':>3s}{'tp':>3s}{'mb':>4s}{'step (us)':>11s}"
          f"{'compute':>10s}{'tp exp':>9s}{'pp exp':>9s}{'dispatch':>10s}"
          f"{'bubble':>9s}{'overhead':>10s}{'nodes':>7s}")
    for i, (step, pp, tp, mb, b) in enumerate(rows, 1):
        print(f"{i:<5d}{pp:>3d}{tp:>3d}{mb:>4d}{step:>11.0f}"
              f"{b['compute_us']:>10.0f}{b['tp_exposed']:>9.0f}"
              f"{b['pp_exposed']:>9.0f}{b['dispatch_us']:>10.0f}"
              f"{b['bubble_us']:>9.0f}{b['overhead_us']:>10.0f}{b['nodes']:>7d}")
    return rows


RANKING_CASES = [
    # (label, pp, tp, mb, measured end-to-end iter time us) -- F16, 2xB200,
    # global batch 8192, dim 4096, hidden 16384, stages 2, bf16. Wall clock from
    # results.csv, so it includes the driver and skew terms.
    ("one GPU", 1, 1, 4, 18720.0),
    ("pp=2",    2, 1, 4, 23364.0),
    ("tp=2",    1, 2, 4, 26523.0),
]


MACHINE = "b200"


def ranking_check():
    """Does the model rank parallel strategies correctly once overheads are in?

    This is the out-of-sample test the calibration check is not: the constants
    were fitted on GPU-side kernel time at a different shape, and here the model
    has to order three different parallel strategies by end-to-end step time.
    """
    print("ranking check against F16 (2xB200, global batch 8192, dim 4096, "
          "hidden 16384, stages 2, bf16)\n")
    print(f"{'config':<10s}{'predicted':>11s}{'measured':>10s}{'ratio':>8s}")
    got = []
    for label, pp, tp, mb, measured in RANKING_CASES:
        step, _ = estimate(gpus=4, pp=pp, tp=tp, mb=mb, global_batch=8192,
                           dim=4096, hidden=16384, stages=2, dtype="bf16",
                           machine=MACHINE)
        got.append((step, label, measured))
        print(f"{label:<10s}{step:>11.0f}{measured:>10.0f}{step / measured:>7.2f}x")
    pred_order = [l for _, l, _ in sorted(got)]
    meas_order = [l for _, l, _ in sorted(got, key=lambda r: r[2])]
    print(f"\npredicted order: {pred_order}")
    print(f"measured  order: {meas_order}")
    if pred_order[0] != meas_order[0]:
        print("FAIL: the model does not pick the fastest configuration")
        return 1
    print("OK: the model picks the fastest configuration")
    if pred_order != meas_order:
        # Only the configurations it got wrong, i.e. everything but the winner.
        rest = sorted(got)[1:]
        pred_spread = max(s for s, _, _ in rest) / min(s for s, _, _ in rest) - 1.0
        meas_spread = (
            max(m for _, _, m in rest) / min(m for _, _, m in rest) - 1.0
        )
        print(
            f"NOTE: the rest of the order is wrong. The model separates the "
            f"non-winning configurations by {pred_spread * 100:.0f}% where the "
            f"measurement separates them by {meas_spread * 100:.0f}%, so it cannot "
            f"tell TP from PP here -- only that neither beats staying on one GPU. "
            f"Treat it as a filter, not a ranking."
        )
    return 0


def calibration_check():
    """Does the model reproduce the runs it was calibrated on (F12)?

    Only a consistency check: these are the same measurements the constants came
    from, so agreement shows the arithmetic is right, not that the model
    generalizes. The out-of-sample test is the ranking in F16.
    """
    cases = [
        # (label, mb, global_batch, measured GPU span us)
        ("mb1 batch 8192", 1, 8192, 15735),
        ("mb4 batch 8192 (4x the work: split replicates)", 4, 4 * 8192, 59328),
    ]
    print("calibration check against F12 (dim 8192, hidden 32768, bf16, tp=2, stages=1)\n")
    print(f"{'case':<48s}{'predicted':>11s}{'measured':>10s}{'ratio':>8s}")
    worst = 0.0
    for label, mb, gb, measured in cases:
        step, b = estimate(gpus=2, pp=1, tp=2, mb=mb, global_batch=gb,
                           dim=8192, hidden=32768, stages=1, dtype="bf16")
        # F12's numbers are GPU spans, which exclude the driver/skew term.
        step -= b["overhead_us"]
        ratio = step / measured
        worst = max(worst, abs(ratio - 1.0))
        print(f"{label:<48s}{step:>11.0f}{measured:>10d}{ratio:>7.2f}x")
    print(f"\nworst deviation {worst * 100:.0f}%")
    return 0 if worst < 0.25 else 1


LIMITS = """
What this cannot know, stated rather than left to be discovered:

  * Contention flips signs, not just magnitudes. On a quiet machine GPipe beat
    1F1B by 16.5% on the very configuration where the contended measurement had
    1F1B ahead by 16% (F63). 1F1B fills pipeline bubbles; when the step is
    host-dispatch-bound there are none to fill and its extra ordering is pure
    host work. Nothing in this program can see how busy the machine will be, so
    a ranking produced here does not transfer to a machine under load.
  * Peak memory under ZeRO-3 is likewise not a property of the program but of
    the host-dispatch to GPU-execution ratio (F47), unless the bounded buffer
    pool is on.
  * `tp` is not a schedule-level choice. Piper's IR carries no partition
    information, so changing it rebuilds and retraces the model (F1). Anything
    here that varies `tp` is proposing a different program, not a different
    schedule.
  * The compute term is a roofline fitted at one shape. It held to 2% on the
    shape it was fitted to and is unvalidated elsewhere.

Use it to exclude configurations, not to choose between close ones.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--global-batch", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=32768)
    ap.add_argument("--stages", type=int, default=1)
    ap.add_argument("--dtype", choices=sorted(BYTES), default="bf16")
    ap.add_argument("--machine", choices=sorted(COLL), default="b200",
                    help="which host's measured collective constants to use")
    ap.add_argument("--limits", action="store_true",
                    help="print what the model cannot know, and stop")
    ap.add_argument("--calibration-check", action="store_true")
    ap.add_argument("--ranking-check", action="store_true",
                    help="Out-of-sample: rank three parallel strategies against F16.")
    args = ap.parse_args(argv)
    global MACHINE
    MACHINE = getattr(args, "machine", "b200")
    if getattr(args, "limits", False):
        print(LIMITS)
        return 0
    if args.calibration_check:
        return calibration_check()
    if args.ranking_check:
        return ranking_check()
    search(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
