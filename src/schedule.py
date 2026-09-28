import json
import os
from typing import Any


def load_schedule_directives(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        directives = json.load(f)
    if not isinstance(directives, list):
        raise ValueError(f"schedule directives file must contain a list, got: {type(directives)}")

    for i, directive in enumerate(directives):
        if not isinstance(directive, dict):
            raise ValueError(f"schedule directive[{i}] must be an object, got: {type(directive)}")
        _validate_directive_shape(directive, i)
        if directive.get("op") == "split" and directive.get("num_microbatches") == "__MBS__":
            raise ValueError(
                "split.num_microbatches must be encoded in the schedule JSON; "
                "'__MBS__' is no longer supported"
            )
    return directives


def load_schedule_info(path: str) -> dict[str, Any]:
    return derive_schedule_info(load_schedule_directives(path), path)


def derive_schedule_info(directives: list[dict], schedule_path: str) -> dict[str, Any]:
    pp_to_devices: dict[int, list[int]] = {}
    num_microbatches = None
    mesh = None

    for directive in directives:
        if not isinstance(directive, dict):
            continue
        op = directive.get("op")
        if op == "place":
            pp_idx = _filter_value(directive.get("filter"), "PP")
            devices = directive.get("devices", directive.get("device"))
            if pp_idx is None or not isinstance(devices, list) or not devices:
                raise ValueError(f"place directive must include PP filter and non-empty devices: {directive}")
            pp_to_devices[int(pp_idx)] = [int(d) for d in devices]
        elif op == "mesh":
            mesh = [[str(a), int(n)] for a, n in directive["axes"]]
        elif op == "split":
            n = int(directive.get("num_microbatches", 0))
            if n <= 0:
                raise ValueError(f"split directive requires num_microbatches > 0: {directive}")
            if num_microbatches is not None and num_microbatches != n:
                raise ValueError(
                    f"multiple split directives disagree on num_microbatches: "
                    f"{num_microbatches} vs {n}"
                )
            num_microbatches = n

    if not pp_to_devices:
        raise ValueError("schedule JSON must include place directives with PP filters")
    pp_indices = sorted(pp_to_devices)
    expected_pp = list(range(len(pp_indices)))
    if pp_indices != expected_pp:
        raise ValueError(f"PP indices must be contiguous from 0, got {pp_indices}")
    device_counts = {len(devices) for devices in pp_to_devices.values()}
    if len(device_counts) != 1:
        raise ValueError(f"all PP place directives must use the same device count, got {pp_to_devices}")
    device_keys = sorted({tuple(devices) for devices in pp_to_devices.values()})
    if num_microbatches is None:
        raise ValueError("schedule JSON must include a split directive with num_microbatches")
    group = next(iter(device_counts))
    if mesh is not None:
        import math
        if math.prod(n for _, n in mesh) != group:
            raise ValueError(f"mesh axes {mesh} multiply to {math.prod(n for _, n in mesh)}, "
                             f"but each stage's device group has {group} devices")

    return {
        "name": os.path.splitext(os.path.basename(schedule_path))[0],
        "path": schedule_path,
        "num_stages": len(pp_indices),
        "pp_degree": len(device_keys),
        "dp_degree": next(iter(device_counts)),
        "num_microbatches": num_microbatches,
        # Named axes over each stage's device group, outermost first (log F82); absent: one
        # unnamed axis, the group itself, as before.
        **({"mesh": mesh} if mesh is not None else {}),
    }


def _validate_directive_shape(directive: dict, idx: int) -> None:
    op = directive.get("op")
    if op in {"place", "replicate", "shard", "shard_tensor", "ring_exchange", "fuse_collectives", "split"}:
        if not isinstance(directive.get("filter"), dict):
            raise ValueError(f"{op} directive[{idx}] requires object field 'filter': {directive}")
        if "filters" in directive:
            raise ValueError(f"{op} directive[{idx}] does not accept field 'filters': {directive}")
    elif op == "mesh":
        axes = directive.get("axes")
        if (not isinstance(axes, list) or not axes
                or not all(isinstance(a, list) and len(a) == 2 and isinstance(a[0], str) and int(a[1]) > 0 for a in axes)
                or len({a[0] for a in axes}) != len(axes)):
            raise ValueError(f"mesh directive[{idx}] needs axes: [[name, size], ...] with distinct names: {directive}")
    elif op == "route":
        if directive.get("mode", "thread") not in ("thread", "consumers"):
            raise ValueError(f"route directive[{idx}] mode must be 'thread' or 'consumers': {directive}")
    elif op == "order":
        filters = directive.get("filters")
        if not isinstance(filters, list) or not filters:
            raise ValueError(f"order directive[{idx}] requires non-empty list field 'filters': {directive}")
        for group_idx, group in enumerate(filters):
            if not isinstance(group, list) or not group:
                raise ValueError(
                    f"order directive[{idx}] group[{group_idx}] must be a non-empty list: {directive}"
                )
            for filter_idx, flt in enumerate(group):
                if not isinstance(flt, dict):
                    raise ValueError(
                        f"order directive[{idx}] group[{group_idx}][{filter_idx}] "
                        f"must be a filter object: {directive}"
                    )
    else:
        raise ValueError(f"schedule directive[{idx}] has unsupported op: {directive}")


def _filter_value(filter_spec, key: str):
    if isinstance(filter_spec, dict):
        return filter_spec.get(key)
    return None


def mesh_coords(info: dict, index: int) -> dict[str, int]:
    """Coordinates of a rank's place in its stage group on the schedule's mesh (log F82).

    ``index`` is that place (``PIPER_DP_RANK``); axes are read row-major, the first
    outermost, as the runtime builds its axis groups. {} when the schedule has no mesh.
    """
    import math
    mesh = info.get("mesh") or []
    sizes = [int(n) for _, n in mesh]
    return {a: (index // math.prod(sizes[k + 1:])) % sizes[k] for k, (a, _) in enumerate(mesh)}
