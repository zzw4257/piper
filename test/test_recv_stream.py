"""A receive runs on a CUDA stream of its own, so a blocking send cannot hold it up (log F103)."""
from types import SimpleNamespace

from src.runtime import RuntimeState


def test_a_receive_gets_its_own_stream_and_the_dag_is_unchanged() -> None:
    sid = lambda **kw: RuntimeState.stream_id(None, SimpleNamespace(**kw))  # noqa: E731
    assert sid(stream="pp_stream", node_kind="RECV_COMM") == "pp_stream_recv"
    assert sid(stream="pp_stream", node_kind="SEND_COMM") == "pp_stream"
    assert sid(stream="default_stream", node_kind="COMPUTE") == "default_stream"
    assert RuntimeState.stream_id(None, "pp_stream") == "pp_stream"
