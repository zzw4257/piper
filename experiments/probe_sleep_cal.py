"""Is torch.cuda._sleep's delivered duration proportional to the requested one?

The skew-transfer sweep found (waiting - base)/skew plateauing at 0.855
independent of payload. Either arrival skew is paid at 85.5%, or the injection
delivers 85.5% of what it is asked for. Calibrating at one long spin (10.8 ms)
and injecting short ones (0.025-3.2 ms) would do exactly that if the clock
boosts differently at the two durations.
"""
import os, sys, torch

def measure(cycles):
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); a.record(); torch.cuda._sleep(int(cycles)); b.record()
    torch.cuda.synchronize(); return a.elapsed_time(b) * 1000.0   # us

def main():
    torch.cuda.set_device(int(os.environ.get("DEV", "0")))
    print(torch.cuda.get_device_name(), flush=True)
    for _ in range(3): measure(2e7)
    long_us = min(measure(2e7) for _ in range(5))
    cpu = 2e7 / long_us
    print(f"calibration spin: 2e7 cycles -> {long_us:.0f} us  => {cpu:.1f} cycles/us")
    print(f"{'requested_us':>13}{'delivered_us':>14}{'ratio':>8}")
    for req in (25, 50, 100, 200, 400, 800, 1600, 3200, 10000):
        got = min(measure(req * cpu) for _ in range(5))
        print(f"{req:>13}{got:>14.1f}{got/req:>8.3f}")

if __name__ == "__main__":
    sys.exit(main())
