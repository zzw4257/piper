# results/

Raw measurements behind the findings in `notes/log.md`, one directory per
machine. Kept because they are small and because `logs/` is gitignored, so
without this the evidence would live only on two shared hosts.

`b200` is catalyst-fleet1 (NVLink 5, contended for most of the project);
`h200` is the ZJU host (NVLink 4, ~90 logged-in users). Every claim that is
supposed to hold on both machines has a file in each.

| file | finding | what it holds |
|---|---|---|
| `law_*.json` | F48, F51 | collective cost vs payload, three regimes, two collectives |
| `xfer_*.json` | F49 | cost vs injected arrival skew, three payloads |
| `knee_*.json` | F49, F51 | per-rank skew incidence, 16-96 MiB, locating the ring's localization knee |
| `hostbound_*.txt` | F50 | Piper's iteration against the same arithmetic run bare |
| `zero3_peak.txt` | F42 | peak memory vs depth, shipped / prefetch / plain DP |
| `zero3_mech.txt` | F43 | the host-sync knob and the per-node memory trace |
| `zero3_pool.txt` | F44 | bounded buffer pool slopes |
| `syncmodes_*.txt` | F60 | device / narrow / defer, correctness and timing |
| `ppsync_*.txt` | F60 | the same three modes under TP x PP |
| `epzero_*.txt` | F60 | the same three modes under EP and ZeRO-3 |
| `fwdskip_*.txt`, `ab_fwdskip_*.txt` | F58 | forwarded-output detach skip, verification and A/B |
| `h200*.txt` | F45, F46, F47 | the first H200 session: P3, CP=4, memory replication |

Timings in these files were taken on shared machines. The numbers quoted in the
log are minima over interleaved repetitions, and the log says so wherever it
matters; a single line here is not a result on its own.
