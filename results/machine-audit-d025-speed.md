# Machine audit

Other users' CPU is in cores (1.0 = one core busy). Samples are those the monitor took between the run's start and finish (server start-up included). A run with no coverage is not known to be clean.

| sweep | run | samples | other CPU mean | other CPU peak | other GPU procs | power-capped samples | slowdown samples | peak temp C | min clock ratio | flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d025-sat-gather-rtx5090-seed0 | b64-w0-r32 | 74 | 0.00 | 0.01 | 0 | 22 | 0 | 59 | 0.87 | clean |
| d025-sat-gather-rtx5090-seed1 | b64-w0-r32 | 71 | 0.00 | 0.01 | 0 | 21 | 0 | 64 | 0.92 | clean |
| d025-sat-gather-rtx5090-seed2 | b64-w0-r32 | 75 | 0.00 | 0.01 | 0 | 35 | 0 | 66 | 0.92 | clean |
| d025-sat-gather-rtx5090-seed3 | b64-w0-r32 | 72 | 0.00 | 0.01 | 0 | 28 | 0 | 62 | 0.92 | clean |
| d025-sat-kernel-rtx5090-seed0 | b64-w0-r32 | 53 | 0.00 | 0.01 | 0 | 2 | 0 | 54 | 0.93 | clean |
| d025-sat-kernel-rtx5090-seed1 | b64-w0-r32 | 54 | 0.00 | 0.01 | 0 | 0 | 0 | 51 | 0.93 | clean |
| d025-sat-kernel-rtx5090-seed2 | b64-w0-r32 | 52 | 0.00 | 0.01 | 0 | 2 | 0 | 55 | 0.93 | clean |
| d025-sat-kernel-rtx5090-seed3 | b64-w0-r32 | 53 | 0.00 | 0.01 | 0 | 0 | 0 | 52 | 0.93 | clean |

8 runs; 8 clean; 0 flagged.
