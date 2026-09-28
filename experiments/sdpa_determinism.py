"""F93: which SDPA backward kernel runs, and is it reproducible, under each determinism setting (one GPU).

    python experiments/sdpa_determinism.py
"""
import sys, time, warnings, torch, torch.nn.functional as F
sys.argv = [a for a in sys.argv if not a.startswith("--temp-dir") and not a.startswith("/var/tmp/ziweizho-ray")]
torch.manual_seed(0)
q, k, v = (torch.randn(16, 12, 1024, 64, device="cuda") for _ in range(3))
def grads():
    qq, kk, vv = (t.clone().requires_grad_() for t in (q, k, v))
    F.scaled_dot_product_attention(qq, kk, vv).sum().backward()
    return qq.grad, kk.grad, vv.grad
def kernels():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        grads(); torch.cuda.synchronize()
    names = {e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA}
    return sorted({n[:60] for n in names if any(s in n.lower() for s in ("fmha", "flash", "attention", "cutlass", "softmax", "gemm", "sm90", "sm100"))})[:6]
for mode in ("default", "warn", "strict"):
    torch.use_deterministic_algorithms(mode != "default", warn_only=mode == "warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = grads(); b = grads()
    same = all(torch.equal(x, y) for x, y in zip(a, b))
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): grads()
    torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 100
    print(f"{mode:8s} grads bit-identical across two runs: {same}; {ms:6.1f} ms per fwd+bwd; warnings: {sorted({str(x.message)[:90] for x in w})}")
    print(f"         kernels: {kernels()}")
print("PASS")
