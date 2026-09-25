from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import os
import time
import torch
import torch.distributed as dist
from torch.autograd.graph import GradientEdge, Node
from torch.nn import Parameter

from .backward import construct_reverse_graph, get_param_groups, _get_grad_fn_or_grad_acc
from .runtime import BufferStore, EventStore, ParamStorage, RuntimeState, StageStore
from .tasks import TaskType

# An A2A and a TP all-reduce are the same structural class of node: a comm that
# sits on an activation boundary, replaces one tensor of its producer's buffer and
# passes the rest through. Consumer arms therefore have to treat them alike, or a
# compute node silently falls through to the wrong inputs.
_FWD_BOUNDARY_COMM_TASKS = (TaskType.FWD_A2A, TaskType.FWD_TP_ALL_REDUCE, TaskType.FWD_RING_EXCHANGE)
_BWD_BOUNDARY_COMM_TASKS = (TaskType.BWD_A2A, TaskType.BWD_TP_ALL_REDUCE, TaskType.BWD_RING_EXCHANGE)


def flush_pending_losses(executor) -> list[float]:
    """Convert whatever `defer` mode is still holding. Called at the end of a run."""
    pending = getattr(executor, "_pending_losses", None)
    if not pending:
        return []
    evt = getattr(executor, "_step_done", None)
    if evt is not None:
        evt.synchronize()
    executor._pending_losses = []
    return _drain_losses(pending)


def _drain_losses(loss_buffer: list) -> list[float]:
    """Empty the loss buffer into plain floats. Call only after a synchronize."""
    losses = [
        float(item.item() if isinstance(item, torch.Tensor) else item)
        for item in loss_buffer
    ]
    loss_buffer.clear()
    return losses


@dataclass
class CommunicationExecutor:
    """Actor-local communication operations used by the DAG dispatcher."""

    runtime: RuntimeState
    stages: StageStore
    logger: Any

    def send(self, send_data: Any, peer_pp_rank: int, stream: torch.cuda.Stream) -> None:
        global_dst_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        with torch.cuda.stream(stream):
            tensors = send_data if isinstance(send_data, (list, tuple)) else [send_data]
            use_lo_hi = global_dst_rank > self.runtime.global_rank
            pp_group = self.runtime.pp_lo_hi if use_lo_hi else self.runtime.pp_hi_lo
            for tensor in tensors:
                dist.send(tensor, dst=global_dst_rank, group=pp_group)

    def recv_fwd(self, recv_ubid: Any, peer_pp_rank: int, stream: torch.cuda.Stream) -> list:
        global_src_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        buf = [
            torch.empty(
                shape,
                dtype=dtype,
                requires_grad=requires_grad,
                device=self.runtime.device,
            )
            for shape, dtype, requires_grad in self.stages.bucket(recv_ubid).forward_input_meta
        ]
        with torch.cuda.stream(stream):
            use_hi_lo = global_src_rank > self.runtime.global_rank
            pp_group = self.runtime.pp_hi_lo if use_hi_lo else self.runtime.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

    def recv_meta(self, meta: list, peer_pp_rank: int, stream: torch.cuda.Stream) -> list:
        """Receive tensors described by (shape, dtype, requires_grad), in order (consumer routing)."""
        global_src_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)
        buf = [torch.empty(shape, dtype=dtype, requires_grad=rg, device=self.runtime.device)
               for shape, dtype, rg in meta]
        with torch.cuda.stream(stream):
            use_hi_lo = global_src_rank > self.runtime.global_rank
            pp_group = self.runtime.pp_hi_lo if use_hi_lo else self.runtime.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

    def recv_bwd(self, shape_meta: list, peer_pp_rank: int, stream: torch.cuda.Stream) -> list:
        global_src_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        buf = [
            torch.empty(shape, dtype=dtype, device=self.runtime.device)
            for shape, dtype in shape_meta
        ]
        with torch.cuda.stream(stream):
            use_hi_lo = global_src_rank > self.runtime.global_rank
            pp_group = self.runtime.pp_hi_lo if use_hi_lo else self.runtime.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

    def all_to_all(self, input_tensor: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        output_buf = torch.empty_like(input_tensor, device=self.runtime.device)
        with torch.cuda.stream(stream):
            dist.all_to_all_single(output_buf, input_tensor, group=self.runtime.ep_group)
        return output_buf

    def all_reduce_activation(
        self,
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        """Sum a boundary activation (or its gradient) across the TP group.

        Returns a fresh contiguous leaf rather than reducing in place: the input is
        an entry of the producer's ``detached_outs``/``inp_grads``, which the
        producer's own backward still reads, and an in-place op on a leaf that
        requires grad is an autograd error.

        # ponytail: rides on ep_group, which is idle whenever a region is TP-sharded
        # rather than EP-sharded, and is already a separate communicator from
        # dp_group so the collective gets its own NCCL proxy stream. Valid only while
        # tp_degree equals the place-group size; a real tp_group is needed to compose
        # TP with DP.
        """
        with torch.cuda.stream(stream):
            out = input_tensor.detach().clone(memory_format=torch.contiguous_format)
            dist.all_reduce(out, group=self.runtime.ep_group)
        return out

    def ring_exchange(
        self,
        tensors: list[torch.Tensor],
        shift: int,
        stream: torch.cuda.Stream,
    ) -> list[torch.Tensor]:
        """Rotate tensors one hop around the group: send to rank+shift, receive from rank-shift.

        Returns fresh contiguous leaves for the same reason as all_reduce_activation.
        Sends and receives are issued as one batch so a two-rank ring (where the
        peer is the same rank both ways) cannot deadlock. Rides on ep_group; see
        the ponytail note on all_reduce_activation for the limit that implies.
        """
        group = self.runtime.ep_group
        n = dist.get_world_size(group=group)
        me = dist.get_rank(group=group)
        dst = dist.get_global_rank(group, (me + shift) % n)
        src = dist.get_global_rank(group, (me - shift) % n)
        with torch.cuda.stream(stream):
            # clone, not contiguous(): a contiguous() that aliases the producer's
            # buffer would let the default stream free that block under an
            # in-flight isend on this stream. The clone lives in this stream's pool.
            sends = [t.detach().clone(memory_format=torch.contiguous_format) for t in tensors]
            recvs = [torch.empty_like(s) for s in sends]
            ops = [dist.P2POp(dist.isend, s, dst, group=group) for s in sends]
            ops += [dist.P2POp(dist.irecv, r, src, group=group) for r in recvs]
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        return recvs

    def all_reduce_fused(
        self,
        tensors: list[torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> list[torch.Tensor]:
        """One all-reduce covering several boundary activations.

        Each collective in the DAG re-pays the two ranks' arrival difference,
        because compute sits between them (notes/log.md F33); issuing several as
        one call pays it once. Returns views into the reduced buffer, in input
        order, so callers can hand each producer back its own slice.
        """
        shapes = [t.shape for t in tensors]
        with torch.cuda.stream(stream):
            flat = torch.cat([t.detach().reshape(-1) for t in tensors])
            dist.all_reduce(flat, group=self.runtime.ep_group)
            out, offset = [], 0
            for t, shape in zip(tensors, shapes):
                n = t.numel()
                out.append(flat[offset:offset + n].view(shape))
                offset += n
        return out

    def all_reduce_grads(self, ubid: Any, stream: torch.cuda.Stream) -> int:
        assert ubid is not None, "all_reduce_grads requires a non-None ubid"
        if not self.has_trainable_params_for_collective(ubid, "all_reduce_grads"):
            return 0
        bucket = self.stages.bucket(ubid)
        grad_tensors = []
        for idx in bucket.trainable_param_idxs:
            param = bucket.forward_args[idx]
            assert param is not None, (
                f"all_reduce_grads: ubid={ubid} idx={idx} param is None"
            )
            assert param.grad is not None, (
                f"all_reduce_grads: ubid={ubid} idx={idx} param.grad is None"
            )
            grad_tensors.append(param.grad)

        total_bytes = sum(grad.numel() * grad.element_size() for grad in grad_tensors)
        with torch.cuda.stream(stream):
            for grad in grad_tensors:
                dist.all_reduce(grad, group=self.runtime.dp_group)
        return total_bytes

    def reduce_scatter(self, ubid: Any, stream: torch.cuda.Stream) -> int:
        assert ubid is not None, "reduce_scatter requires a non-None ubid"
        if not self.has_trainable_params_for_collective(ubid, "reduce_scatter"):
            return 0
        assert ubid in self.stages.grad_sharded_ubids, (
            f"reduce_scatter: ubid={ubid} is not in grad_sharded_ubids="
            f"{self.stages.grad_sharded_ubids}"
        )
        bucket = self.stages.bucket(ubid)
        assert bucket.param_shard_info is not None, (
            f"reduce_scatter: missing param_shard_info for ubid={ubid}"
        )
        flat_grads = bucket.flat_grads
        rs_out = bucket.reduce_scatter_grads
        assert flat_grads is not None, (
            f"reduce_scatter: missing flat_grads buffer for ubid={ubid}"
        )
        assert rs_out is not None, (
            f"reduce_scatter: missing reduce_scatter_grads buffer for ubid={ubid}"
        )
        with torch.cuda.stream(stream):
            total_bytes = flat_grads.numel() * flat_grads.element_size()
            tmp = torch.empty_like(rs_out)
            dist.reduce_scatter_tensor(tmp, flat_grads, group=self.runtime.dp_group)
            rs_out.add_(tmp)
            return total_bytes

    def has_trainable_params_for_collective(self, ubid: Any, op_name: str) -> bool:
        bucket = self.stages.get_bucket(ubid)
        if bucket is not None and bucket.trainable_param_idxs:
            return True
        self.logger.debug(
            "%s: skipping collective for ubid=%s because it has no trainable param indices",
            op_name,
            ubid,
        )
        return False


def boundary_outputs(out_list: list, fwd_inputs: list) -> list:
    """Detach a segment's boundary outputs, except the ones it merely forwards.

    A segment output that *is* one of the segment's own inputs is forwarded, not
    produced -- the property notes/cp-design.md G1 relies on, decided here by
    object identity rather than by metadata so it cannot drift from what the
    GraphModule actually returned.

    Detaching a forwarded value builds a shadow leaf whose only job is to receive
    the downstream gradient and hand it to the real leaf one autograd call later.
    Emitting the input itself lets that gradient land directly on the leaf, and
    the produced outputs' backward accumulates the in-segment contribution on
    top: the same sum, one tensor and one graph entry fewer.

    The pairing rule this creates is load-bearing: ``pre_detach_outs[i] is
    detached_outs[i]`` marks a forwarded output, and ``ComputeExecutor.backward``
    must *not* drive such a pair or the downstream gradient is counted twice.
    See ``test/test_boundary_outputs_runtime.py``.
    """
    input_ids = {id(t) for t in fwd_inputs if isinstance(t, torch.Tensor)}
    out = []
    for t in out_list:
        if not isinstance(t, torch.Tensor) or not t.requires_grad:
            out.append(t)
        elif id(t) in input_ids:
            out.append(t)
        else:
            out.append(t.detach().requires_grad_(True))
    return out


@dataclass
class ComputeExecutor:
    """Actor-local bucket forward/backward execution."""

    runtime: RuntimeState
    stages: StageStore
    logger: Any

    def log_compute_loss_inputs(
        self,
        labels: Any,
        node: Any,
        fwd_key: Any,
        fwd_out: dict,
    ) -> None:
        def _summarize_value(value: Any) -> str:
            if isinstance(value, torch.Tensor):
                return (
                    f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                    f"requires_grad={value.requires_grad}, device={value.device})"
                )
            if isinstance(value, (list, tuple)):
                return "[" + ", ".join(_summarize_value(v) for v in value) + "]"
            if value is None:
                return "None"
            return type(value).__name__

        self.logger.debug(
            "compute_loss inputs: rank=%s node_uid=%s node_type=%s tag=%s "
            "fwd_key=%s labels=%s out_with_grad=%s pre_detach_outs=%s "
            "detached_outs=%s send_output=%s",
            self.runtime.global_rank,
            getattr(node, "uid", None),
            getattr(getattr(node, "task_type", None), "value", None),
            getattr(node, "tag", None),
            fwd_key,
            _summarize_value(labels),
            _summarize_value(fwd_out.get("out_with_grad")),
            _summarize_value(fwd_out.get("pre_detach_outs")),
            _summarize_value(fwd_out.get("detached_outs")),
            _summarize_value(fwd_out.get("send_output")),
        )

    def forward(self, ubid: Any, input_tensors: Any, compute_stream: torch.cuda.Stream) -> dict:
        bucket = self.stages.bucket(ubid)
        fwd_fn = bucket.forward_fn
        fwd_args = bucket.forward_args
        input_idxs = bucket.input_idxs

        if not isinstance(input_tensors, (list, tuple)):
            input_tensors = [input_tensors]

        for i, tensor in zip(input_idxs, input_tensors):
            if isinstance(tensor, (list, tuple)):
                tensor = tensor[0]
            if tensor.requires_grad:
                tensor = tensor.detach().requires_grad_(True)
            fwd_args[i] = tensor

        fwd_inputs = [fwd_args[i] for i in input_idxs]
        inp_with_grad = [t for t in fwd_inputs if t is not None and t.requires_grad]

        with torch.cuda.stream(compute_stream):
            output = fwd_fn(fwd_args)

        for i in input_idxs:
            fwd_args[i] = None

        out_list = list(output) if isinstance(output, tuple) else [output]
        possibly_detached = boundary_outputs(out_list, fwd_inputs)
        out_with_grad = [
            t for t in out_list if isinstance(t, torch.Tensor) and t.requires_grad
        ]

        return {
            "pre_detach_outs": out_list,
            "detached_outs": possibly_detached,
            "out_with_grad": out_with_grad,
            "send_output": output,
            "inp_with_grad": inp_with_grad,
            "fwd_inputs": fwd_inputs,
        }

    def backward(
        self,
        ubid: Any,
        mb_idx: int,
        outputs_or_loss: list,
        upstream_grads: Any,
        pre_detach_outs: Any,
        detached_outs: Any,
        inp_with_grad: Any,
        out_with_grad: Any,
        compute_stream: torch.cuda.Stream,
    ) -> dict | None:
        if pre_detach_outs is None:
            if upstream_grads is not None:
                with torch.cuda.stream(compute_stream):
                    torch.autograd.backward(outputs_or_loss, upstream_grads)
            else:
                with torch.cuda.stream(compute_stream):
                    outputs_or_loss[0].backward()
        else:
            # `p is d` marks a forwarded output (forward above emitted the input
            # itself). Its downstream gradient was assigned straight onto the
            # leaf, so driving it would accumulate the same gradient a second
            # time. Skip the pair; the value is already where inp_grads reads it.
            bwd_pairs = [
                (p, d.grad)
                for p, d in zip(pre_detach_outs, detached_outs)
                if (isinstance(d, torch.Tensor) and d.requires_grad
                    and d.grad is not None and p is not d)
            ]
            forwarded_with_grad = any(
                isinstance(d, torch.Tensor) and d.requires_grad
                and d.grad is not None and p is d
                for p, d in zip(pre_detach_outs, detached_outs)
            )
            assert bwd_pairs or forwarded_with_grad, (
                f"BWD ubid={ubid} mb={mb_idx}: detached boundary has no tensors "
                "with a materialized grad"
            )
            if bwd_pairs:
                outputs_bwd = [p for p, _g in bwd_pairs]
                grads_bwd = [g for _p, g in bwd_pairs]
                with torch.cuda.stream(compute_stream):
                    torch.autograd.backward(outputs_bwd, grads_bwd)

        if inp_with_grad:
            output_grads = [t.grad for t in inp_with_grad if t.grad is not None]
            if output_grads:
                return {"send_output": output_grads}
        return None

    @staticmethod
    def grad_with_param_layout(weight: Parameter, grad: torch.Tensor) -> torch.Tensor:
        if (
            grad.dtype == weight.dtype
            and grad.device == weight.device
            and grad.layout == weight.layout
            and tuple(grad.stride()) == tuple(weight.stride())
        ):
            return grad
        if weight.layout == torch.strided and grad.layout != torch.strided:
            grad = grad.to_dense()
        out = torch.empty_strided(
            tuple(weight.shape),
            tuple(weight.stride()),
            dtype=weight.dtype,
            device=weight.device,
        )
        out.copy_(grad)
        return out

    def fused_backward(self, stage_outputs_or_loss: list, output_grads: Any, weights: list) -> None:
        wts = [w for w in weights if w.requires_grad]
        if not wts:
            return
        dweights = torch.autograd.grad(
            stage_outputs_or_loss,
            inputs=wts,
            grad_outputs=output_grads,
            retain_graph=False,
            allow_unused=True,
        )
        for w, dw in zip(wts, dweights):
            if dw is None:
                continue
            dw = self.grad_with_param_layout(w, dw)
            if w.grad is None:
                w.grad = dw
            else:
                if (
                    w.grad.dtype != w.dtype
                    or w.grad.device != w.device
                    or w.grad.layout != w.layout
                    or tuple(w.grad.stride()) != tuple(w.stride())
                ):
                    w.grad = self.grad_with_param_layout(w, w.grad)
                w.grad += dw

    def backward_weight_from_outputs(
        self,
        stage_outputs_or_loss: list,
        output_grads: Any,
        weights: Any,
    ) -> None:
        self.fused_backward(stage_outputs_or_loss, output_grads, list(weights))

    def bucket_backward_input(
        self,
        stage_outputs_or_loss: list,
        output_grads: Any,
        input_values: list,
        weights: Any,
    ):
        weights = list(weights)

        if output_grads is None:
            output_grads = [torch.ones_like(o) for o in stage_outputs_or_loss]

        input_values = [inp for inp in input_values if inp.requires_grad]
        if not input_values:
            self.fused_backward(stage_outputs_or_loss, output_grads, weights)
            for i, t in enumerate(stage_outputs_or_loss):
                if isinstance(t, torch.Tensor):
                    stage_outputs_or_loss[i] = t.detach()
            return (), [], None

        stage_output_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, stage_outputs_or_loss)))
        stage_input_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, input_values)))
        weight_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, weights)))

        reverse_edges_dict = construct_reverse_graph(stage_output_grad_fns)
        param_groups = get_param_groups(stage_input_grad_fns, weight_grad_fns, reverse_edges_dict)
        handles = []
        for param_group in param_groups:
            for i, intermediate in enumerate(param_group["intermediates"]):
                def get_hook(pg, idx):
                    def hook(grad_inputs):
                        if pg.get("grads") is None:
                            pg["grads"] = [None] * len(pg["intermediates"])
                        pg["grads"][idx] = tuple(grad_inputs)
                    return hook
                handles.append(intermediate.register_prehook(get_hook(param_group, i)))

        dinputs = torch.autograd.grad(
            stage_outputs_or_loss,
            inputs=input_values,
            grad_outputs=output_grads,
            retain_graph=True,
            allow_unused=True,
        )
        for inp, dinput in zip(input_values, dinputs):
            if inp.grad is None:
                inp.grad = dinput
            else:
                inp.grad += dinput

        output_backward_ctx = {
            "stage_outputs_or_loss": list(stage_outputs_or_loss),
            "output_grads": output_grads,
        }

        for handle in handles:
            handle.remove()

        return dinputs, param_groups, output_backward_ctx

    def bucket_backward_weight(
        self,
        weights: Any,
        param_groups: list,
        ubid: Any | None = None,
        mb_idx: int | None = None,
    ) -> None:
        if not param_groups:
            return

        grad_acc_to_weight: dict[Node, Parameter] = {}

        for weight in weights:
            grad_acc = _get_grad_fn_or_grad_acc(weight)
            grad_acc_to_weight[grad_acc] = weight

        for param_group in param_groups:
            valid_edges: list[GradientEdge] = []
            valid_grad_outputs: list[torch.Tensor] = []

            for grads_tuple, intermediate in zip(
                param_group.get("grads", []), param_group["intermediates"]
            ):
                if grads_tuple is None:
                    continue
                for i, grad in enumerate(grads_tuple):
                    if grad is not None:
                        valid_edges.append(GradientEdge(intermediate, i))
                        valid_grad_outputs.append(grad)

            del param_group["intermediates"]

            if not valid_edges:
                continue

            weight_edges = tuple(GradientEdge(w, 0) for w in param_group["params"])
            if not weight_edges:
                continue

            dweights = torch.autograd.grad(
                valid_edges,
                weight_edges,
                grad_outputs=valid_grad_outputs,
                retain_graph=False,
            )

            del param_group["grads"]

            for grad_acc, dw in zip(param_group["params"], dweights):
                if dw is None or grad_acc not in grad_acc_to_weight:
                    continue
                weight = grad_acc_to_weight[grad_acc]
                dw = self.grad_with_param_layout(weight, dw)
                if weight.grad is None:
                    weight.grad = dw
                else:
                    if (
                        weight.grad.dtype != weight.dtype
                        or weight.grad.device != weight.device
                        or weight.grad.layout != weight.layout
                        or tuple(weight.grad.stride()) != tuple(weight.stride())
                    ):
                        weight.grad = self.grad_with_param_layout(weight, weight.grad)
                    weight.grad += dw


@dataclass
class DagExecutor:
    """Execute a loaded TrainingDAG in sorted order."""

    runtime: RuntimeState
    stages: StageStore
    buffers: BufferStore
    events: EventStore
    params: ParamStorage
    communication: CommunicationExecutor
    compute: ComputeExecutor
    logger: Any

    @staticmethod
    def _node_meta(node: Any) -> dict:
        return getattr(node, "node_meta", {}) or {}

    def _node_bucket_key(self, node: Any) -> Any | None:
        return self._node_meta(node).get("bucket_key")

    def _sync_payload_ubid(self, node: Any) -> Any | None:
        bk = self._node_bucket_key(node)
        if bk is not None:
            return bk
        if node.data_preds:
            pred_bk = self._node_bucket_key(node.data_preds[0])
            if pred_bk is not None:
                return pred_bk
        return None

    def _sync_payload_ubids(self, node: Any) -> list[Any]:
        sync_ubids = self._node_meta(node).get("sync_ubids")
        if sync_ubids:
            return list(sync_ubids)
        ubid = self._sync_payload_ubid(node)
        return [ubid] if ubid is not None else []

    def _rf_enter(self, label: str):
        if not self.runtime.pytorch_profiler_enabled:
            return None
        mt = torch.autograd.set_multithreading_enabled(False)
        mt.__enter__()
        rf = torch.profiler.record_function(label)
        rf.__enter__()
        return (rf, mt)

    @staticmethod
    def _rf_exit(rf) -> None:
        if rf is not None:
            rf_ctx, mt = rf
            rf_ctx.__exit__(None, None, None)
            mt.__exit__(None, None, None)

    @staticmethod
    def _node_tag_str(node: Any) -> str:
        tag = getattr(node, "tag", None)
        if not isinstance(tag, dict) or not tag:
            return "{}"
        items = sorted(tag.items(), key=lambda kv: kv[0])
        return "{" + ",".join(f"{k}={v}" for k, v in items) + "}"

    def _wait_for_all_gather(self, compute_node: Any) -> None:
        compute_stream = self.runtime.stream_for_task(compute_node)
        for pred in compute_node.data_preds:
            if pred.task_type == TaskType.ALL_GATHER:
                ag_evt = self.events.all_gather.get(pred.uid)
                if ag_evt is not None:
                    compute_stream.wait_event(ag_evt)

    def _all_to_all_ep_boundary(
        self,
        node: Any,
        tensor: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        direction = self._node_meta(node).get("direction")
        if direction == "outgoing":
            tensor = tensor.contiguous()
        elif direction != "incoming":
            raise ValueError(
                f"A2A node uid={getattr(node, 'uid', '<unknown>')} has invalid "
                f"direction={direction!r}; expected 'incoming' or 'outgoing'"
            )

        tensor = self.communication.all_to_all(tensor, stream=stream)
        if direction == "incoming":
            tensor = tensor.contiguous()
        return tensor

    def run(
        self,
        dag: Any,
        sorted_dag_nodes: list[Any],
        inputs: Any,
        labels: Any,
        loss_buffer: list,
        loss_fn=None,
    ) -> dict:
        """Run one iteration of the loaded TrainingDAG.

        Returns ``{"losses": [...]}`` for the microbatches whose backward computed
        the loss on this rank. Empty when the loss lives on another PP rank.
        """
        assert dag is not None, "load_training_dag() must be called before run_dag()"
        assert sorted_dag_nodes is not None, "load_training_dag() must initialize sorted node order"

        self._loss_events = []
        if self.SYNC_MODE != "device":
            prev = getattr(self, "_step_done", None)
            if prev is not None:
                # Stream-ordered, so the host does not block: every stream of
                # this step queues behind the previous step's optimizer.
                for st in self.runtime.streams.values():
                    st.wait_event(prev)
                # The host does have to block before reading last step's losses,
                # but by now its GPU work has had a full Ray round trip to finish.
                if self.SYNC_MODE == "defer":
                    # Only `defer` has host-side reading to do here; `narrow`
                    # has already converted its losses and needs the ordering
                    # alone, which the stream waits above provide.
                    prev.synchronize()
                self._step_done = None
            if self.SYNC_MODE == "defer":
                self._carried_losses = _drain_losses(getattr(self, "_pending_losses", []))
                self._pending_losses = []

        self.params.drain_pending_frees()
        debug_enabled = self.logger.isEnabledFor(logging.DEBUG)

        self.buffers.reset()
        self.events.reset()
        self.params.clear_param_grads(
            in_flight_streams=(
                self.runtime.streams.values() if self.SYNC_MODE != "device" else None
            )
        )
        default_stream = self.runtime.default_stream()
        self.params.zero_grad_buffers(default_stream)
        zero_evt = torch.cuda.Event()
        zero_evt.record(default_stream)
        for stream in self.runtime.streams.values():
            if stream is not default_stream:
                stream.wait_event(zero_evt)
        comp_events: dict[Any, torch.cuda.Event] = {}
        update_results: list[dict] = []

        self.buffers.init_refcounts(dag)
        fusion_results: dict[str, list] = {}
        last_comp_event_by_stream: dict[str, torch.cuda.Event] = {}

        # PIPER_MEM_TRACE=<dir>: allocated bytes before each dispatched node, one
        # TSV per rank per step. Experiment knob (log F42); off unless set.
        mem_trace_dir = os.environ.get("PIPER_MEM_TRACE")
        time_trace_dir = os.environ.get("PIPER_TIME_TRACE")
        trace_dir = mem_trace_dir or time_trace_dir
        # Memory sampling costs a device query per node; timing costs ~50 ns.
        want_mem = bool(mem_trace_dir)
        mem_trace: list[tuple] = []
        ag_host_sync = os.environ.get("PIPER_AG_HOST_SYNC") == "1"

        inner: dict[str, float] = {}

        def _inner(uid, t0):
            if trace_dir:
                inner[uid] = (time.perf_counter() - t0) * 1e6

        for node in sorted_dag_nodes:
            if trace_dir:
                mem_trace.append((node.uid, node.task_type.value,
                                  torch.cuda.memory_allocated() if want_mem else 0,
                                  torch.cuda.max_memory_allocated() if want_mem else 0,
                                  time.perf_counter()))
            task_type = node.task_type
            batch = node.batches[0]
            mb_idx = batch.mb_idx
            ubid = self._node_bucket_key(node)
            node_stream = self.runtime.stream_for_task(node)
            node_stream_id = self.runtime.stream_id(node)
            node_tag = self._node_tag_str(node)

            if debug_enabled:
                self.logger.debug(
                    f"run_dag dispatch: {task_type.value} "
                    f"tag={node_tag}"
                )

            task_label = f"{task_type.value}:{node_tag}:uid{node.uid}"
            self.runtime.nvtx_push(task_label)
            rf = self._rf_enter(task_label)

            match task_type:
                case TaskType.SEND if "route_out_idxs" in self._node_meta(node) or "route_grad_slots" in self._node_meta(node):
                    # Consumer routing (log F71): send only this edge's tensors.
                    compute_node = node.data_preds[0]
                    node_stream.wait_event(comp_events[compute_node.uid])
                    meta = self._node_meta(node)
                    cbuf = self.buffers.task[compute_node.uid]
                    if "route_out_idxs" in meta:
                        outs = cbuf["send_output"]
                        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
                        send_data = [outs[k] for k in meta["route_out_idxs"]]
                    else:
                        send_data = [cbuf["inp_grads"][j] for j in meta["route_grad_slots"]]
                        assert all(g is not None for g in send_data), (
                            f"{node.uid}: a routed gradient is missing for slots {meta['route_grad_slots']}")
                    self.communication.send(send_data, node.peer_pp_rank, stream=node_stream)
                    self.buffers.release(compute_node.uid)

                case TaskType.SEND:
                    compute_node = node.data_preds[0]
                    node_stream.wait_event(comp_events[compute_node.uid])
                    send_data = self.buffers.task[compute_node.uid]["send_output"]
                    self.communication.send(send_data, node.peer_pp_rank, stream=node_stream)
                    send_buf = self.buffers.task.get(compute_node.uid)
                    if isinstance(send_buf, dict):
                        send_buf["send_output"] = None
                    self.buffers.release(compute_node.uid)

                case TaskType.RECV:
                    compute_node = node.data_succs[0]
                    comp_evt = last_comp_event_by_stream.get(self.runtime.stream_id(compute_node))
                    if comp_evt is not None:
                        node_stream.wait_event(comp_evt)
                    rmeta = self._node_meta(node)
                    if compute_node.task_type == TaskType.FWD and "route_slots" in rmeta:
                        full = self.stages.bucket(self._node_bucket_key(compute_node)).forward_input_meta
                        recv_tensors = self.communication.recv_meta(
                            [full[j] for j in rmeta["route_slots"]], node.peer_pp_rank, stream=node_stream)
                    elif "route_grad_out_idxs" in rmeta:
                        fwd_uid = compute_node.node_meta.get("fwd_uid")
                        fwd_key = (compute_node.node_meta.get("bucket_key"), fwd_uid)
                        shape_all = self.buffers.task[("shape_all",) + fwd_key]
                        recv_tensors = self.communication.recv_meta(
                            [(*shape_all[k], False) for k in rmeta["route_grad_out_idxs"]],
                            node.peer_pp_rank, stream=node_stream)
                    elif compute_node.task_type == TaskType.FWD:
                        recv_ubid = self._node_bucket_key(compute_node)
                        recv_tensors = self.communication.recv_fwd(
                            recv_ubid, node.peer_pp_rank, stream=node_stream
                        )
                    else:
                        fwd_uid = compute_node.node_meta.get("fwd_uid")
                        fwd_key = (compute_node.node_meta.get("bucket_key"), fwd_uid)
                        shape_meta = self.buffers.task[("shape_ref",) + fwd_key]
                        recv_tensors = self.communication.recv_bwd(
                            shape_meta, node.peer_pp_rank, stream=node_stream
                        )
                    self.buffers.task[node.uid] = recv_tensors
                    recv_evt = torch.cuda.Event()
                    recv_evt.record(node_stream)
                    self.events.recv[node.uid] = recv_evt

                case TaskType.FWD_A2A:
                    fwd_pred = next(p for p in node.data_preds if p.task_type == TaskType.FWD)
                    node_stream.wait_event(comp_events[fwd_pred.uid])
                    tensor_idx = self._node_meta(node)["a2a_tensor_idx"]
                    fwd_buf = dict(self.buffers.task[fwd_pred.uid])
                    self.buffers.release(fwd_pred.uid)
                    detached_outs = list(fwd_buf["detached_outs"])
                    detached_outs[tensor_idx] = self._all_to_all_ep_boundary(
                        node, detached_outs[tensor_idx], node_stream
                    ).requires_grad_(True)
                    fwd_buf["detached_outs"] = detached_outs
                    self.buffers.task[node.uid] = fwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(node_stream)
                    self.events.a2a[node.uid] = a2a_evt

                case TaskType.BWD_A2A:
                    bwd_pred = next(
                        p for p in node.data_preds
                        if p.task_type in (TaskType.BWD, TaskType.BWD_I)
                    )
                    node_stream.wait_event(comp_events[bwd_pred.uid])
                    tensor_idx = self._node_meta(node)["a2a_tensor_idx"]
                    bwd_buf = dict(self.buffers.task[bwd_pred.uid])
                    self.buffers.release(bwd_pred.uid)
                    inp_grads = list(bwd_buf["inp_grads"])
                    grad_a2a_out = inp_grads[tensor_idx]
                    assert grad_a2a_out is not None, (
                        f"BWD_A2A tag={node_tag}: grad at a2a_tensor_idx={tensor_idx} is None"
                    )
                    inp_grads[tensor_idx] = self._all_to_all_ep_boundary(
                        node, grad_a2a_out, node_stream
                    )
                    bwd_buf["inp_grads"] = inp_grads
                    self.buffers.task[node.uid] = bwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(node_stream)
                    self.events.a2a[node.uid] = a2a_evt

                case TaskType.FWD_TP_ALL_REDUCE:
                    # g of Megatron's f/g pair: the region's output is a partial
                    # sum. Backward is identity, so no paired node is needed -- the
                    # consumer's BWD arm wires the gradient straight onto the
                    # pre-all-reduce output.
                    fwd_pred = next(p for p in node.data_preds if p.task_type == TaskType.FWD)
                    node_stream.wait_event(comp_events[fwd_pred.uid])
                    meta = self._node_meta(node)
                    tensor_idx = meta["tp_tensor_idx"]
                    gid = meta.get("fusion_group")
                    if gid is not None and meta["fusion_leader"] == node.uid:
                        # Collect every member's tensor before releasing anything:
                        # a member's buffer is released by that member's own node.
                        for src_uid, _idx in meta["fusion_sources"]:
                            evt = comp_events.get(src_uid)
                            if evt is not None:
                                node_stream.wait_event(evt)
                        parts = [
                            self.buffers.task[src_uid]["detached_outs"][idx]
                            for src_uid, idx in meta["fusion_sources"]
                        ]
                        fusion_results[gid] = self.communication.all_reduce_fused(
                            parts, node_stream
                        )
                    fwd_buf = dict(self.buffers.task[fwd_pred.uid])
                    self.buffers.release(fwd_pred.uid)
                    detached_outs = list(fwd_buf["detached_outs"])
                    reduced = (
                        fusion_results[gid][meta["fusion_index"]]
                        if gid is not None
                        else self.communication.all_reduce_activation(
                            detached_outs[tensor_idx], node_stream)
                    )
                    detached_outs[tensor_idx] = reduced.requires_grad_(True)
                    fwd_buf["detached_outs"] = detached_outs
                    self.buffers.task[node.uid] = fwd_buf
                    tp_evt = torch.cuda.Event()
                    tp_evt.record(node_stream)
                    self.events.a2a[node.uid] = tp_evt

                case TaskType.BWD_TP_ALL_REDUCE:
                    # f of Megatron's f/g pair: the gradient w.r.t. the region's
                    # input is a partial sum. Forward is identity.
                    bwd_pred = next(
                        p for p in node.data_preds
                        if p.task_type in (TaskType.BWD, TaskType.BWD_I)
                    )
                    node_stream.wait_event(comp_events[bwd_pred.uid])
                    meta = self._node_meta(node)
                    tensor_idx = meta["tp_tensor_idx"]
                    gid = meta.get("fusion_group")
                    if gid is not None and meta["fusion_leader"] == node.uid:
                        for src_uid, _idx in meta["fusion_sources"]:
                            evt = comp_events.get(src_uid)
                            if evt is not None:
                                node_stream.wait_event(evt)
                        parts = [
                            self.buffers.task[src_uid]["inp_grads"][idx]
                            for src_uid, idx in meta["fusion_sources"]
                        ]
                        assert all(p is not None for p in parts), (
                            f"BWD_TP_ALL_REDUCE fused group {gid}: a member's grad is None"
                        )
                        fusion_results[gid] = self.communication.all_reduce_fused(
                            parts, node_stream
                        )
                    bwd_buf = dict(self.buffers.task[bwd_pred.uid])
                    self.buffers.release(bwd_pred.uid)
                    inp_grads = list(bwd_buf["inp_grads"])
                    grad_tp = inp_grads[tensor_idx]
                    assert grad_tp is not None, (
                        f"BWD_TP_ALL_REDUCE tag={node_tag}: grad at "
                        f"tp_tensor_idx={tensor_idx} is None"
                    )
                    inp_grads[tensor_idx] = (
                        fusion_results[gid][meta["fusion_index"]]
                        if gid is not None
                        else self.communication.all_reduce_activation(grad_tp, node_stream)
                    )
                    bwd_buf["inp_grads"] = inp_grads
                    self.buffers.task[node.uid] = bwd_buf
                    tp_evt = torch.cuda.Event()
                    tp_evt.record(node_stream)
                    self.events.a2a[node.uid] = tp_evt

                case TaskType.FWD_RING_EXCHANGE:
                    # Ring step: hand our chunk to the next rank, take the previous
                    # rank's. The rotated tensors replace the forwarded slots so the
                    # consumer region sees the new chunk under the same input index.
                    meta = self._node_meta(node)
                    idxs = meta["ring_tensor_idxs"]
                    # Spliced: the source region is the predecessor. Hoisted: the
                    # supplier of the forwarded slots is -- the previous ring node
                    # or the defining region -- and the source region runs alongside.
                    fwd_pred = next(
                        p for p in node.data_preds
                        if p.task_type in (TaskType.FWD, TaskType.FWD_RING_EXCHANGE)
                    )
                    if fwd_pred.task_type == TaskType.FWD:
                        node_stream.wait_event(comp_events[fwd_pred.uid])
                    else:
                        ring_pred_evt = self.events.a2a.get(fwd_pred.uid)
                        if ring_pred_evt is not None:
                            node_stream.wait_event(ring_pred_evt)
                    fwd_buf = dict(self.buffers.task[fwd_pred.uid])
                    self.buffers.release(fwd_pred.uid)  # refcounted by data_succs
                    detached_outs = list(fwd_buf["detached_outs"])
                    rotated = self.communication.ring_exchange(
                        [detached_outs[i] for i in idxs], meta["ring_shift"], node_stream
                    )
                    for i, t in zip(idxs, rotated):
                        detached_outs[i] = t.requires_grad_(True)
                    if meta.get("hoisted"):
                        # Only the rotated slots are ours; the consumer takes the
                        # produced slots from the source region. Blank the rest so
                        # a wrong merge fails on None instead of using stale data.
                        detached_outs = [
                            t if i in idxs else None for i, t in enumerate(detached_outs)
                        ]
                    fwd_buf["detached_outs"] = detached_outs
                    self.buffers.task[node.uid] = fwd_buf
                    ring_evt = torch.cuda.Event()
                    ring_evt.record(node_stream)
                    self.events.a2a[node.uid] = ring_evt

                case TaskType.BWD_RING_EXCHANGE:
                    # Reverse ring: the gradient for the chunk we consumed goes back
                    # to the rank that held that chunk one step earlier, where its
                    # backward accumulates it with the gradient from its own use.
                    bwd_pred = next(
                        p for p in node.data_preds
                        if p.task_type in (TaskType.BWD, TaskType.BWD_I)
                    )
                    node_stream.wait_event(comp_events[bwd_pred.uid])
                    meta = self._node_meta(node)
                    idxs = meta["ring_tensor_idxs"]
                    bwd_buf = dict(self.buffers.task[bwd_pred.uid])
                    self.buffers.release(bwd_pred.uid)
                    inp_grads = list(bwd_buf["inp_grads"])
                    parts = [inp_grads[i] for i in idxs]
                    assert all(g is not None for g in parts), (
                        f"BWD_RING_EXCHANGE tag={node_tag}: grad at ring_tensor_idxs="
                        f"{idxs} is None; the forwarded slot did not receive a gradient"
                    )
                    rotated = self.communication.ring_exchange(parts, meta["ring_shift"], node_stream)
                    for i, g in zip(idxs, rotated):
                        inp_grads[i] = g
                    bwd_buf["inp_grads"] = inp_grads
                    self.buffers.task[node.uid] = bwd_buf
                    ring_evt = torch.cuda.Event()
                    ring_evt.record(node_stream)
                    self.events.a2a[node.uid] = ring_evt

                case TaskType.ALL_REDUCE:
                    bwd_node = node.data_preds[0]
                    ar_ubids = self._sync_payload_ubids(node)
                    assert ar_ubids, (
                        f"ALL_REDUCE node uid={node.uid} has no sync_payload_ubids"
                    )
                    node_stream.wait_event(comp_events[bwd_node.uid])
                    for ar_ubid in ar_ubids:
                        self.communication.all_reduce_grads(ar_ubid, stream=node_stream)
                    ar_evt = torch.cuda.Event()
                    ar_evt.record(node_stream)
                    self.events.all_reduce[node.uid] = ar_evt
                    self.buffers.release(bwd_node.uid)

                case TaskType.REDUCE_SCATTER:
                    bwd_node = node.data_preds[0]
                    rs_ubid = self._node_bucket_key(node)
                    assert rs_ubid is not None, (
                        f"REDUCE_SCATTER node uid={node.uid} has no bucket_key"
                    )
                    node_stream.wait_event(comp_events[bwd_node.uid])
                    rs_bytes = self.communication.reduce_scatter(rs_ubid, stream=node_stream)
                    rs_evt = torch.cuda.Event()
                    rs_evt.record(node_stream)
                    self.events.reduce_scatter[node.uid] = rs_evt
                    if rs_bytes:
                        self.params.defer_free_full_grads(rs_ubid, rs_evt)
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        assert False and "Param free should happen after a compute node"
                    self.buffers.release(bwd_node.uid)

                case TaskType.ALL_GATHER:
                    ag_ubid = self._node_bucket_key(node)
                    assert ag_ubid is not None, (
                        f"ALL_GATHER node uid={node.uid} has no bucket_key"
                    )
                    if ag_host_sync:
                        # F42 experiment: alloc_full_params runs on the host at
                        # dispatch, so a DAG budget edge cannot delay it. Block the
                        # host on the budget predecessor instead. Stalls dispatch;
                        # a knob, not a fix.
                        after = self._node_meta(node).get("prefetch_after_uid")
                        evt = comp_events.get(after) if after else None
                        if evt is not None:
                            evt.synchronize()
                    self.params.all_gather_full_params(ag_ubid, stream=node_stream)
                    ag_evt = torch.cuda.Event()
                    ag_evt.record(node_stream)
                    self.events.all_gather[node.uid] = ag_evt

                case TaskType.FWD if "input_sources" in self._node_meta(node):
                    # Consumer routing (log F71): each input slot names its supplier.
                    srcs = self._node_meta(node)["input_sources"]
                    vals: list = [None] * len(srcs)
                    for p in node.data_preds:
                        pm = self._node_meta(p)
                        if p.task_type == TaskType.RECV:
                            if p.uid in self.events.recv:
                                node_stream.wait_event(self.events.recv.pop(p.uid))
                            for slot, tensor in zip(pm["route_slots"], self.buffers.task[p.uid]):
                                vals[slot] = tensor
                            self.buffers.release(p.uid)
                        elif p.task_type in (TaskType.FWD, TaskType.FWD_TP_ALL_REDUCE):
                            # A TP all-reduce carries its region's outputs, one reduced (log F73).
                            producer = p
                            if p.task_type == TaskType.FWD_TP_ALL_REDUCE:
                                evt = self.events.a2a.pop(p.uid, None)
                                if evt is not None:
                                    node_stream.wait_event(evt)
                                producer = next(q for q in p.data_preds if q.task_type == TaskType.FWD)
                            seg = self._node_meta(producer).get("segment_id")
                            outs = self.buffers.task[p.uid]["detached_outs"]
                            for j, (kind, s_, k) in enumerate(srcs):
                                if kind == "seg" and s_ == seg:
                                    vals[j] = outs[k]
                            self.buffers.release(p.uid)
                        else:
                            raise NotImplementedError(
                                f"consumer routing does not yet combine with {p.task_type} before {node.uid}")
                    for j, (kind, i, _k) in enumerate(srcs):
                        if kind == "model":
                            vals[j] = inputs[i]
                    missing = [j for j, v in enumerate(vals) if v is None]
                    assert not missing, f"{node.uid}: no supplier delivered input slots {missing} ({srcs})"
                    self._wait_for_all_gather(node)
                    _t0 = time.perf_counter()
                    fwd_out = self.compute.forward(ubid, vals, node_stream)
                    _inner(node.uid, _t0)
                    fwd_out["routed"] = True
                    self.buffers.task[node.uid] = fwd_out
                    fwd_key = (node.node_meta.get("bucket_key"), node.uid)
                    self.buffers.task[("shape_ref",) + fwd_key] = [
                        (x.shape, x.dtype) for x in fwd_out["out_with_grad"]]
                    self.buffers.task[("shape_all",) + fwd_key] = [
                        (x.shape, x.dtype) for x in fwd_out["pre_detach_outs"]]
                    self.buffers.task[fwd_key] = fwd_out
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.FWD:
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.events.recv:
                        node_stream.wait_event(self.events.recv.pop(recv_pred.uid))

                    comm_preds = [
                        p for p in node.data_preds if p.task_type in _FWD_BOUNDARY_COMM_TASKS
                    ]
                    for p in comm_preds:
                        evt = self.events.a2a.get(p.uid)
                        if evt is not None:
                            node_stream.wait_event(evt)
                            # A hoisted ring's event is also awaited by the next
                            # ring node; only the last data successor may drop it.
                            if len(getattr(p, "data_succs", ())) <= 1:
                                self.events.a2a.pop(p.uid, None)

                    # Base input set. Only a *hoisted* ring changes the rule: then
                    # the compute predecessor carries the produced slots and the
                    # ring node the rotated ones. Otherwise this is upstream's
                    # first-match lookup, byte for byte.
                    hoisted_preds = [
                        p for p in comm_preds if self._node_meta(p).get("hoisted")
                    ]
                    if hoisted_preds:
                        fwd_data_pred = next(
                            (p for p in node.data_preds if p.task_type == TaskType.FWD), None
                        ) or hoisted_preds[0]
                    else:
                        fwd_data_pred = next(
                            (p for p in node.data_preds
                             if p.task_type == TaskType.FWD
                             or p.task_type in _FWD_BOUNDARY_COMM_TASKS), None
                        )
                    if fwd_data_pred is not None:
                        input_tensors = list(self.buffers.task[fwd_data_pred.uid]["detached_outs"])
                        self.buffers.release(fwd_data_pred.uid)
                        for p in hoisted_preds:
                            pm = self._node_meta(p)
                            if p.uid == fwd_data_pred.uid:
                                continue
                            ring_outs = self.buffers.task[p.uid]["detached_outs"]
                            for i in pm["ring_tensor_idxs"]:
                                t = ring_outs[i]
                                assert t is not None, (
                                    f"hoisted ring {p.uid} has no tensor at slot {i}"
                                )
                                # Allocated on the comm stream, read here: tell the
                                # allocator so a later comm-stream reuse waits for us.
                                t.record_stream(node_stream)
                                input_tensors[i] = t
                            self.buffers.release(p.uid)
                    elif recv_pred is not None:
                        input_tensors = self.buffers.task[recv_pred.uid]
                        self.buffers.release(recv_pred.uid)
                    else:
                        input_tensors = inputs

                    self._wait_for_all_gather(node)

                    _t0 = time.perf_counter()
                    fwd_out = self.compute.forward(ubid, input_tensors, node_stream)
                    _inner(node.uid, _t0)
                    self.buffers.task[node.uid] = fwd_out
                    fwd_key = (node.node_meta.get("bucket_key"), node.uid)
                    self.buffers.task[("shape_ref",) + fwd_key] = [
                        (t.shape, t.dtype) for t in fwd_out["out_with_grad"]
                    ]
                    self.buffers.task[fwd_key] = fwd_out
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.BWD:
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.events.recv:
                        node_stream.wait_event(self.events.recv.pop(recv_pred.uid))
                    self._wait_for_all_gather(node)

                    a2a_pred = next(
                        (p for p in node.data_preds
                         if p.task_type in _BWD_BOUNDARY_COMM_TASKS), None
                    )
                    if a2a_pred is not None and a2a_pred.uid in self.events.a2a:
                        node_stream.wait_event(self.events.a2a.pop(a2a_pred.uid))

                    fwd_uid = node.node_meta.get("fwd_uid")
                    fwd_key = (node.node_meta.get("bucket_key"), fwd_uid)
                    fwd_out = self.buffers.task[fwd_key]

                    routed = bool(fwd_out.get("routed")) and not self._node_meta(node).get("compute_loss", False)
                    if routed:
                        # Consumer routing (log F71): sum each output's gradient over every
                        # consumer that read it, then drive this segment's backward.
                        acc: dict = {}
                        my_seg = dag.nodes[fwd_uid].node_meta.get("segment_id")
                        for p in node.data_preds:
                            pm = self._node_meta(p)
                            if p.task_type == TaskType.RECV and "route_grad_out_idxs" in pm:
                                if p.uid in self.events.recv:
                                    node_stream.wait_event(self.events.recv.pop(p.uid))
                                pairs = zip(pm["route_grad_out_idxs"], self.buffers.task[p.uid])
                            elif p.task_type in (TaskType.BWD, TaskType.BWD_I, TaskType.BWD_TP_ALL_REDUCE):
                                # A TP all-reduce carries its consumer's input grads, one reduced (log F73).
                                consumer = p
                                if p.task_type == TaskType.BWD_TP_ALL_REDUCE:
                                    evt = self.events.a2a.pop(p.uid, None)
                                    if evt is not None:
                                        node_stream.wait_event(evt)
                                    consumer = next(q for q in p.data_preds
                                                    if q.task_type in (TaskType.BWD, TaskType.BWD_I))
                                c_sources = dag.nodes[self._node_meta(consumer)["fwd_uid"]].node_meta["input_sources"]
                                grads = self.buffers.task[p.uid]["inp_grads"]
                                pairs = [(k, grads[j]) for j, (kind, s_, k) in enumerate(c_sources)
                                         if kind == "seg" and s_ == my_seg and grads[j] is not None]
                            elif p.task_type in _BWD_BOUNDARY_COMM_TASKS:
                                raise NotImplementedError(
                                    f"consumer routing does not yet combine with {p.task_type} before {node.uid}")
                            else:
                                continue
                            # The sum must run on this node's stream, which is the one that
                            # waited for the recv/all-reduce events; a gradient made on another
                            # stream is also marked as read here before its buffer is released.
                            with torch.cuda.stream(node_stream):
                                for k, g in pairs:
                                    if p.task_type != TaskType.BWD:
                                        g.record_stream(node_stream)
                                    acc[k] = g if k not in acc else acc[k] + g
                            self.buffers.release(p.uid)
                        outs = fwd_out["pre_detach_outs"]
                        outputs_or_loss = [outs[k] for k in sorted(acc)]
                        upstream_grads = [acc[k] for k in sorted(acc)]
                        pre_detach_outs = None
                        detached_outs = None
                    elif self._node_meta(node).get("compute_loss", False):
                        assert loss_fn is not None
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.compute.log_compute_loss_inputs(labels, node, fwd_key, fwd_out)
                        with torch.cuda.stream(node_stream):
                            outputs_or_loss = [loss_fn(fwd_out["out_with_grad"][0], labels)]
                        # Keep the value, not just the graph that consumes it. Detached
                        # and unsynchronized on purpose: .item() here would block the
                        # dispatch loop mid-DAG and perturb the schedule being measured.
                        # _update converts after its existing synchronize().
                        loss_buffer.append(outputs_or_loss[0].detach())
                        # The loss is the only value _update needs on the host.
                        # Recording its readiness lets the narrow mode wait for
                        # it instead of for the whole device (log F56/F59/F60).
                        _levt = torch.cuda.Event()
                        _levt.record(node_stream)
                        self._loss_events.append(_levt)
                        upstream_grads = None
                    elif recv_pred is not None:
                        upstream_grads = self.buffers.task[recv_pred.uid]
                        self.buffers.release(recv_pred.uid)
                        if not isinstance(upstream_grads, (list, tuple)):
                            upstream_grads = [upstream_grads]
                        outputs_or_loss = fwd_out["out_with_grad"]
                        upstream_grads = list(upstream_grads)
                    else:
                        outputs_or_loss = fwd_out["out_with_grad"]
                        upstream_grads = None

                    if routed:
                        pass
                    elif a2a_pred is not None:
                        a2a_buf = self.buffers.task[a2a_pred.uid]
                        pre_detach_outs = fwd_out["pre_detach_outs"]
                        detached_outs = fwd_out["detached_outs"]
                        for d, g in zip(detached_outs, a2a_buf["inp_grads"]):
                            if (
                                d is not None
                                and isinstance(d, torch.Tensor)
                                and d.requires_grad
                                and g is not None
                            ):
                                d.grad = g
                        self.buffers.release(a2a_pred.uid)
                    else:
                        bwd_pred = next(
                            (p for p in node.data_preds
                             if p.task_type in (TaskType.BWD, TaskType.BWD_I)), None
                        )
                        if bwd_pred is not None and recv_pred is None:
                            prev_buf = self.buffers.task[bwd_pred.uid]
                            pre_detach_outs = fwd_out.get("pre_detach_outs")
                            detached_outs = fwd_out.get("detached_outs")
                            for d, g in zip(detached_outs, prev_buf["inp_grads"]):
                                if (
                                    d is not None
                                    and isinstance(d, torch.Tensor)
                                    and d.requires_grad
                                    and g is not None
                                ):
                                    d.grad = g
                            self.buffers.release(bwd_pred.uid)
                        else:
                            pre_detach_outs = None
                            detached_outs = None

                    inp_with_grad = fwd_out.get("inp_with_grad")
                    if self._node_meta(node).get("zero_alloc_full_grads_before"):
                        self.params.alloc_full_grads(ubid, node_stream)

                    _t0 = time.perf_counter()
                    bwd_out = self.compute.backward(
                        ubid, mb_idx, outputs_or_loss, upstream_grads,
                        pre_detach_outs, detached_outs, inp_with_grad,
                        fwd_out.get("out_with_grad"),
                        node_stream,
                    )
                    _inner(node.uid, _t0)
                    buf = bwd_out if bwd_out is not None else {}
                    fwd_inputs_full = fwd_out.get("fwd_inputs")
                    buf["inp_grads"] = (
                        [t.grad if (t is not None and t.requires_grad) else None
                         for t in fwd_inputs_full]
                        if fwd_inputs_full is not None
                        else [t.grad for t in (inp_with_grad or [])]
                    )
                    self.params.accumulate_zero_param_grads_to_flat(ubid, node_stream)
                    self.buffers.task[node.uid] = buf
                    fwd_out.clear()
                    del self.buffers.task[fwd_key]
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    self.events.backward[ubid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.BWD_I:
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.events.recv:
                        node_stream.wait_event(self.events.recv.pop(recv_pred.uid))
                    self._wait_for_all_gather(node)

                    fwd_uid = node.node_meta.get("fwd_uid")
                    fwd_key = (node.node_meta.get("bucket_key"), fwd_uid)
                    fwd_out = self.buffers.task[fwd_key]

                    if self._node_meta(node).get("compute_loss", False):
                        assert loss_fn is not None
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.compute.log_compute_loss_inputs(labels, node, fwd_key, fwd_out)
                        with torch.cuda.stream(node_stream):
                            stage_outputs_or_loss = [loss_fn(fwd_out["out_with_grad"][0], labels)]
                        loss_buffer.append(stage_outputs_or_loss[0].detach())
                        output_grads = None
                    elif recv_pred is not None:
                        upstream_raw = self.buffers.task[recv_pred.uid]
                        self.buffers.release(recv_pred.uid)
                        if not isinstance(upstream_raw, (list, tuple)):
                            upstream_raw = [upstream_raw]
                        stage_outputs_or_loss = fwd_out["out_with_grad"]
                        output_grads = list(upstream_raw)
                    else:
                        stage_outputs_or_loss = fwd_out["out_with_grad"]
                        output_grads = None

                    bwd_a2a_pred = next(
                        (p for p in node.data_preds
                         if p.task_type in _BWD_BOUNDARY_COMM_TASKS), None
                    )
                    if (
                        bwd_a2a_pred is not None
                        and not self._node_meta(node).get("compute_loss", False)
                        and recv_pred is None
                    ):
                        a2a_buf = self.buffers.task[bwd_a2a_pred.uid]
                        detached_outs = fwd_out["detached_outs"]
                        output_grads_full = [
                            a2a_buf["inp_grads"][i]
                            for i, t in enumerate(detached_outs)
                            if t is not None and getattr(t, "requires_grad", False)
                        ]
                        pairs = [
                            (o, g) for o, g in zip(stage_outputs_or_loss, output_grads_full)
                            if g is not None
                        ]
                        stage_outputs_or_loss = [p[0] for p in pairs]
                        output_grads = [p[1] for p in pairs]
                        self.buffers.release(bwd_a2a_pred.uid)

                    input_values = fwd_out.get("inp_with_grad") or []
                    weights = self.stages.bucket(ubid).weights()

                    with torch.cuda.stream(node_stream):
                        dinputs, param_groups, output_backward_ctx = self.compute.bucket_backward_input(
                            stage_outputs_or_loss, output_grads, input_values, iter(weights)
                        )
                    fwd_inputs_full = fwd_out.get("fwd_inputs")
                    if fwd_inputs_full is not None:
                        inp_grads_full = [
                            t.grad if (t is not None and t.requires_grad) else None
                            for t in fwd_inputs_full
                        ]
                    else:
                        inp_grads_full = list(dinputs)
                    self.buffers.task[node.uid] = {
                        "inp_grads": inp_grads_full,
                        "param_groups": param_groups,
                    }
                    if output_backward_ctx is not None:
                        self.buffers.task[node.uid]["output_backward_ctx"] = output_backward_ctx
                    if dinputs:
                        self.buffers.task[node.uid]["send_output"] = list(dinputs)
                    fwd_out.clear()
                    del stage_outputs_or_loss, fwd_out
                    del self.buffers.task[fwd_key]
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    self.events.backward[ubid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.BWD_W:
                    self._wait_for_all_gather(node)
                    if self._node_meta(node).get("zero_alloc_full_grads_before"):
                        self.params.alloc_full_grads(ubid, node_stream)
                    bwdi_node = next(p for p in node.data_preds if p.task_type == TaskType.BWD_I)
                    bwdi_buf = self.buffers.task[bwdi_node.uid]
                    param_groups = bwdi_buf["param_groups"]
                    weights = self.stages.bucket(ubid).weights()
                    with torch.cuda.stream(node_stream):
                        output_backward_ctx = bwdi_buf.get("output_backward_ctx")
                        if output_backward_ctx is not None:
                            self.compute.backward_weight_from_outputs(
                                output_backward_ctx["stage_outputs_or_loss"],
                                output_backward_ctx["output_grads"],
                                iter(weights),
                            )
                        else:
                            self.compute.bucket_backward_weight(
                                iter(weights), param_groups, ubid=ubid, mb_idx=mb_idx
                            )
                    self.params.accumulate_zero_param_grads_to_flat(ubid, node_stream)
                    self.buffers.task[node.uid] = {}
                    self.buffers.release(bwdi_node.uid)
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    self.events.backward[ubid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.UPD:
                    _t0 = time.perf_counter()
                    update_results.append(self._update(node_stream, loss_buffer))
                    _inner(node.uid, _t0)

                case TaskType.ORDER_DUMMY:
                    pass

            self._rf_exit(rf)
            self.runtime.nvtx_pop()

        if trace_dir:
            step = getattr(self, "_mem_trace_step", 0)
            self._mem_trace_step = step + 1
            os.makedirs(trace_dir, exist_ok=True)
            path = os.path.join(trace_dir, f"trace_rank{self.runtime.global_rank}_step{step}.tsv")
            # last_enqueue: after the final node was issued, before the drain
            # below; end: after it. Their difference is the host waiting for
            # the GPU rather than dispatching to it.
            last_enqueue = time.perf_counter()
            with open(path, "w") as f:
                f.write("uid\ttask\tallocated\tmax_allocated\thost_s\tinner_us\n")
                for uid, task, alloc, mx, ts in mem_trace:
                    f.write(f"{uid}\t{task}\t{alloc}\t{mx}\t{ts:.6f}\t{inner.get(uid, -1):.1f}\n")
                f.write(f"__last_enqueue__\tmarker\t0\t0\t{last_enqueue:.6f}\t-1\n")
                f.write(f"__end__\tmarker\t0\t0\t{time.perf_counter():.6f}\t-1\n")
                for k, v in (getattr(self, "_upd_parts", None) or {}).items():
                    f.write(f"__upd_{k}__\tmarker\t0\t0\t0\t{v:.1f}\n")
        return {
            "losses": [
                loss for result in update_results for loss in result["losses"]
            ],
        }

    # How the host waits at the end of a step. `narrow` is the default: it is
    # contract-identical to `device` and measurably faster, and was verified on
    # TP, CP=2, CP=4, TP x PP, EP and ZeRO-3, on two machines, with negative
    # controls and across optimizer steps (log F60).
    #
    # device : block the host until every stream is idle. Upstream behaviour,
    #          kept so the old timing can be reproduced exactly.
    # narrow : block only until the loss values exist. The optimizer and the
    #          trailing collectives stay in flight and the host returns early,
    #          so Ray's round trip and the next step's dispatch overlap this
    #          step's GPU tail. No change to what any caller observes.
    # defer  : do not block at all; hand the losses to the next step, which
    #          converts them after waiting on this step's completion event.
    #          Fastest and steadiest, but losses lag one step, so a caller must
    #          call piper_flush_losses() at the end of its loop.
    SYNC_MODE = os.environ.get("PIPER_SYNC_MODE", "narrow")

    def _update(self, stream: torch.cuda.Stream, loss_buffer: list):
        _parts = self._upd_parts = {} if os.environ.get("PIPER_TIME_TRACE") else None
        _t = time.perf_counter() if _parts is not None else 0.0
        self.params.drain_pending_frees()
        if _parts is not None:
            _parts["drain_frees"] = (time.perf_counter() - _t) * 1e6
            _parts["_t"] = time.perf_counter()
        if self.params.has_zero_shard_optimizers():
            self.params.step_zero_shard_optimizers(stream, self.events.reduce_scatter)
            # This path returned before the sync-mode branch below, so ZeRO-3
            # kept the device-wide block whatever PIPER_SYNC_MODE said. Give it
            # the same three modes (log F61).
            mode = self.SYNC_MODE
            if mode == "device":
                torch.cuda.synchronize()
                return {"losses": _drain_losses(loss_buffer)}
            evt = torch.cuda.Event()
            evt.record(stream)
            self._step_done = evt
            if mode == "narrow":
                for e in self._loss_events:
                    e.synchronize()
                self._loss_events.clear()
                return {"losses": _drain_losses(loss_buffer)}
            self._pending_losses = list(loss_buffer)
            loss_buffer.clear()
            self._loss_events.clear()
            out = {"losses": self._carried_losses}
            self._carried_losses = []
            return out

        for ar_evt in self.events.all_reduce.values():
            stream.wait_event(ar_evt)

        for ubid, bucket in self.stages.buckets.items():
            if bucket.optimizer is None:
                continue
            bwd_evt = self.events.backward.get(ubid)
            if bwd_evt is not None:
                stream.wait_event(bwd_evt)

            with torch.cuda.stream(stream):
                _ts = time.perf_counter() if getattr(self, "_upd_parts", None) is not None else 0.0
                bucket.optimizer.step()
                if getattr(self, "_upd_parts", None) is not None:
                    self._upd_parts["opt_step"] = self._upd_parts.get("opt_step", 0.0) + (
                        time.perf_counter() - _ts) * 1e6
                    self._upd_parts["n_buckets"] = self._upd_parts.get("n_buckets", 0) + 1

        _ts = time.perf_counter() if getattr(self, "_upd_parts", None) is not None else 0.0
        mode = self.SYNC_MODE
        if mode != "device":
            # The step's completion, for the next iteration's streams to queue
            # behind. Required in *both* non-device modes: `narrow` waits on the
            # loss events, and a rank that computes no loss -- every non-final PP
            # stage -- has none, so without this the host would enter the next
            # iteration and free grads the optimizer is still reading.
            evt = torch.cuda.Event()
            evt.record(stream)
            self._step_done = evt
        if mode == "defer":
            self._pending_losses = list(loss_buffer)
            loss_buffer.clear()
            self._loss_events.clear()
            out = {"losses": self._carried_losses}
            self._carried_losses = []
            if getattr(self, "_upd_parts", None) is not None:
                self._upd_parts["final_sync"] = (time.perf_counter() - _ts) * 1e6
                self._upd_parts["loop"] = (_ts - self._upd_parts.pop("_t", _ts)) * 1e6
            return out
        if mode == "narrow":
            # .item() copies on the current stream, so the loss's producing
            # stream has to be waited on first -- but only that one.
            for e in self._loss_events:
                e.synchronize()
            self._loss_events.clear()
        else:
            torch.cuda.synchronize()
        if getattr(self, "_upd_parts", None) is not None:
            # A full device sync sits at the end of the update node, so this
            # node absorbs whatever GPU work is still outstanding. Separating it
            # is the difference between "the optimizer is expensive" and "the
            # iteration's drain happens here" (log F56).
            self._upd_parts["final_sync"] = (time.perf_counter() - _ts) * 1e6
            self._upd_parts["loop"] = (_ts - self._upd_parts.pop("_t", _ts)) * 1e6

        return {
            "losses": _drain_losses(loss_buffer),
        }
