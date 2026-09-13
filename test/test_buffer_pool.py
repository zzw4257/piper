"""_BufferPool: the next taker orders itself after the previous user's event on
its own stream; the host never waits; misses allocate fresh (log F43)."""
from src.runtime import _BufferPool


class _Stream:
    def __init__(self): self.waited = []
    def wait_event(self, evt): self.waited.append(evt)


def test_take_waits_on_the_releasers_event_and_reuses_fifo() -> None:
    pool = _BufferPool(); s = _Stream()
    assert pool.take("k", s) is None                # empty: caller allocates fresh
    pool.give("k", "bufA", "evtA"); pool.give("k", "bufB", "evtB")
    assert pool.take("k", s) == "bufA" and s.waited == ["evtA"]
    assert pool.take("k", s) == "bufB" and s.waited == ["evtA", "evtB"]
    assert pool.take("k", s) is None


def test_keys_do_not_mix_and_none_event_means_no_wait() -> None:
    pool = _BufferPool(); s = _Stream()
    pool.give(("param", 8, "f32"), "p", None); pool.give(("grad", 8, "f32"), "g", "e")
    assert pool.take(("grad", 8, "f32"), s) == "g" and s.waited == ["e"]
    assert pool.take(("param", 8, "f32"), s) == "p" and s.waited == ["e"]
    assert pool.take(("param", 16, "f32"), s) is None


def test_fresh_allocations_are_counted_per_key() -> None:
    pool = _BufferPool()
    assert pool.note_fresh("k") == 1 and pool.note_fresh("k") == 2 and pool.note_fresh("j") == 1


def test_release_detaches_by_replacing_the_tensor_and_reattach_restores_views() -> None:
    """set_ bounds-checks, so the released tensor cannot keep shape (8,) over an
    empty storage; the pool therefore hands the storage on and rebuilds the
    tensor object on the next alloc. This is the invariant the first GPU run of
    the pool violated."""
    import torch
    full = torch.arange(8, dtype=torch.float32)
    view = full[2:6].view(2, 2)
    pooled = full.untyped_storage()
    # release
    full = torch.empty(0, dtype=torch.float32)
    view = torch.empty(0, dtype=torch.float32)
    assert full.numel() == 0 and view.numel() == 0
    # re-attach
    full = torch.empty(0, dtype=torch.float32).set_(pooled, 0, (8,))
    assert full[2:6].view(2, 2).tolist() == [[2.0, 3.0], [4.0, 5.0]]
