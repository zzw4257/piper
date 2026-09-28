"""F94: choose SmolVLM's placement on four GPUs with the pipeline model, then check against measurements.

    python experiments/select_vlm.py PROFILE.json

48 image-caption pairs per step, m microbatches of 48/m. Candidates: one GPU; 2 stages (vision |
decoder); 4 stages (vision in three stages | decoder); heterogeneous (vision in three chunks side by
side on GPUs 0-2, decoder on GPU 3, log F84). Durations from single-GPU stage profiles
(experiments/profile_stages.py); no communication or host cost, as in F78.
"""
import json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pipeline_sim import simulate

B = 48
# measured on 4 healthy B200s (GPU 1 excluded, log F94), 48 pairs per step: ms
MEAS = {("one GPU", 1, "gpipe"): 776.4, ("one GPU", 2, "gpipe"): 814.7, ("one GPU", 4, "gpipe"): 900.4,
        ("hetero", 1, "gpipe"): 347.0, ("hetero", 2, "gpipe"): 301.5, ("hetero", 4, "gpipe"): 291.5,
        ("hetero", 2, "1f1b"): 263.0, ("hetero", 4, "1f1b"): 274.5,
        ("4 stages", 1, "gpipe"): 742.1, ("4 stages", 2, "1f1b"): 455.0, ("4 stages", 4, "1f1b"): 352.1}


def main():
    p = json.load(open(sys.argv[1]))
    t = lambda s, b: p[s][str(b)]  # noqa: E731  [fwd, bwd] ms
    rows = []
    for m in (1, 2, 4):
        b = B // m
        for order in ("gpipe", "1f1b"):
            if m == 1 and order == "1f1b":
                continue
            if order == "gpipe" and b % 3 == 0:
                # the measured one-GPU runs chunk the vision tower in three, as the heterogeneous ones do
                one = m * (3 * sum(t("vision", b // 3)) + sum(t("decoder", b)))
                rows.append(("one GPU", m, order, one))
            st = ["v", "d"]
            rows.append(("2 stages", m, order, simulate(st, {"v": [], "d": ["v"]},
                         {"v": t("vision", b)[0], "d": t("decoder", b)[0]}, {"v": t("vision", b)[1], "d": t("decoder", b)[1]}, m, order)))
            st = ["v1", "v2", "v3", "d"]
            f = {"v1": t("vision_1", b)[0], "v2": t("vision_2", b)[0], "v3": t("vision_3", b)[0], "d": t("decoder", b)[0]}
            g = {"v1": t("vision_1", b)[1], "v2": t("vision_2", b)[1], "v3": t("vision_3", b)[1], "d": t("decoder", b)[1]}
            rows.append(("4 stages", m, order, simulate(st, {"v1": [], "v2": ["v1"], "v3": ["v2"], "d": ["v3"]}, f, g, m, order)))
            if b % 3 == 0:
                c = b // 3
                f = {"c0": t("vision", c)[0], "c1": t("vision", c)[0], "c2": t("vision", c)[0], "d": t("decoder", b)[0]}
                g = {"c0": t("vision", c)[1], "c1": t("vision", c)[1], "c2": t("vision", c)[1], "d": t("decoder", b)[1]}
                rows.append(("hetero", m, order, simulate(["c0", "c1", "c2", "d"], {"c0": [], "c1": [], "c2": [], "d": ["c0", "c1", "c2"]}, f, g, m, order)))
    rows.sort(key=lambda r: r[3])
    print(f"{'rank':>4} {'placement':10s} {'m':>2} {'order':6s} {'predicted':>10} {'measured':>9} {'error':>7}")
    for i, (name, m, order, pred) in enumerate(rows, 1):
        meas = MEAS.get((name, m, order))
        print(f"{i:4d} {name:10s} {m:2d} {order:6s} {pred:10.1f} {meas if meas else '—':>9} "
              f"{(f'{pred / meas - 1:+.0%}' if meas else ''):>7}")
    measured = [(r, MEAS[(r[0], r[1], r[2])]) for r in rows if (r[0], r[1], r[2]) in MEAS]
    best_meas = min(measured, key=lambda x: x[1])[0]
    print(f"model's pick: {rows[0][:3]}; fastest measured: {best_meas[:3]}; "
          f"rank of the fastest measured in the model's order: {rows.index(best_meas) + 1}")
    import itertools
    pairs = list(itertools.combinations(measured, 2))
    agree = sum((a[0][3] < b[0][3]) == (a[1] < b[1]) for a, b in pairs)
    print(f"pairwise order agreement on measured plans: {agree}/{len(pairs)}")


if __name__ == "__main__":
    main()
