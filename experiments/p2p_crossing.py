"""Two ranks send to each other at once, then receive (log F103): when does that hang?

    CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc-per-node 2 experiments/p2p_crossing.py [--recv-stream]

Piper's 1F1B steady state does exactly this: an activation goes forward while a gradient comes
back. With the receive queued on the send's stream, dist.send makes that stream wait until the
peer has received, and the peer does the same. Each size runs in a fresh subprocess with a short
NCCL timeout, so a hang is reported, not waited on forever.
"""
import datetime, os, signal, subprocess, sys, time

import torch
import torch.distributed as dist

if os.environ.get("P2P_CHILD"):
    nbytes, own, k = int(os.environ["P2P_BYTES"]), os.environ.get("P2P_RECV_STREAM") == "1", int(os.environ["P2P_TENSORS"])
    rank = int(os.environ["RANK"]); dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=15))
    peer = 1 - rank
    # one communicator per direction, as Piper's pp_lo_hi / pp_hi_lo
    lo_hi, hi_lo = dist.new_group([0, 1]), dist.new_group([0, 1])
    out_g, in_g = (lo_hi, hi_lo) if rank == 0 else (hi_lo, lo_hi)
    # NCCL builds a pair's communicator on first use, and both sides must take part: warm each
    # direction up in an order that does not cross, as Piper's runtime has by the time it runs
    w = torch.zeros(1, device=dev)
    for g, src in ((lo_hi, 0), (hi_lo, 1)):
        (dist.send if rank == src else dist.recv)(w, peer, group=g)
    torch.cuda.synchronize()
    ts = [torch.ones(nbytes // 4, device=dev) for _ in range(k)]; bufs = [torch.empty_like(t) for t in ts]
    send_s, recv_s = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(send_s):
        for t in ts:                      # k tensors per transfer, sent one by one as Piper's send does
            dist.send(t, peer, group=out_g)
    with torch.cuda.stream(recv_s if own else send_s):
        for b in bufs:
            dist.recv(b, peer, group=in_g)
    torch.cuda.synchronize()
    assert all(b.sum().item() == b.numel() for b in bufs)
    dist.destroy_process_group()
    sys.exit(0)

own = "--recv-stream" in sys.argv
for k, kib in [(k, kib) for k in (1, 2) for kib in (64, 1024, 4096, 16384, 65536)]:
    env = dict(os.environ, P2P_CHILD="1", P2P_BYTES=str(kib * 1024), P2P_RECV_STREAM="1" if own else "0", P2P_TENSORS=str(k))
    t0 = time.time()
    pr = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-port", str(29600 + (kib + k) % 97), __file__],
                          env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        rc = pr.wait(timeout=60)
    except subprocess.TimeoutExpired:   # a hang the NCCL watchdog did not end: kill the whole group
        os.killpg(pr.pid, signal.SIGKILL); pr.wait(); rc = -1
    print(f"P2P recv on {'its own' if own else 'the send'} stream, {k} x {kib:6d} KiB: {'ok' if rc == 0 else 'HANG (NCCL timeout)'} in {time.time() - t0:.0f} s", flush=True)
print("PASS")
