# Machine audit

Other users' CPU is in cores (1.0 = one core busy). Samples are those the monitor took between the run's start and finish (server start-up included). A run with no coverage is not known to be clean.

| sweep | run | samples | other CPU mean | other CPU peak | other GPU procs | power-capped samples | slowdown samples | peak temp C | min clock ratio | flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d020-sat-eager-rtx5090-seed0 | b64-w0-r32 | 56 | 0.00 | 0.03 | 0 | 0 | 0 | 47 | 0.93 | clean |
| d020-sat-eager-rtx5090-seed1 | b64-w0-r32 | 55 | 0.00 | 0.01 | 0 | 1 | 0 | 48 | 0.93 | clean |
| d020-sat-eager-rtx5090-seed2 | b64-w0-r32 | 55 | 0.00 | 0.01 | 0 | 0 | 0 | 47 | 0.94 | clean |
| d020-sat-eager-rtx5090-seed3 | b64-w0-r32 | 55 | 0.00 | 0.01 | 0 | 0 | 0 | 48 | 0.94 | clean |
| d020-sat-graphs-rtx5090-seed0 | b64-w0-r32 | 42 | 0.00 | 0.01 | 0 | 3 | 0 | 52 | 0.93 | clean |
| d020-sat-graphs-rtx5090-seed1 | b64-w0-r32 | 41 | 0.00 | 0.01 | 0 | 1 | 0 | 51 | 0.93 | clean |
| d020-sat-graphs-rtx5090-seed2 | b64-w0-r32 | 43 | 0.00 | 0.00 | 0 | 0 | 0 | 49 | 0.93 | clean |
| d020-sat-graphs-rtx5090-seed3 | b64-w0-r32 | 43 | 0.00 | 0.01 | 0 | 0 | 0 | 48 | 0.93 | clean |

8 runs; 8 clean; 0 flagged.
