# Machine audit

Other users' CPU is in cores (1.0 = one core busy). Samples are those the monitor took between the run's start and finish (server start-up included). A run with no coverage is not known to be clean.

| sweep | run | samples | other CPU mean | other CPU peak | other GPU procs | power-capped samples | slowdown samples | peak temp C | min clock ratio | flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sat-engine-rtx5090-seed0 | b32-w0-r32 | 79 | 0.00 | 0.01 | 0 | 3 | 0 | 58 | 0.92 | clean |
| sat-engine-rtx5090-seed0 | b64-w0-r32 | 73 | 0.00 | 0.01 | 0 | 25 | 0 | 64 | 0.92 | clean |
| sat-engine-rtx5090-seed1 | b32-w0-r32 | 79 | 0.00 | 0.01 | 0 | 4 | 0 | 56 | 0.92 | clean |
| sat-engine-rtx5090-seed1 | b64-w0-r32 | 67 | 0.00 | 0.01 | 0 | 22 | 0 | 62 | 0.92 | clean |
| sat-engine-rtx5090-seed2 | b32-w0-r32 | 82 | 0.00 | 0.01 | 0 | 2 | 0 | 61 | 0.92 | clean |
| sat-engine-rtx5090-seed2 | b64-w0-r32 | 75 | 0.00 | 0.01 | 0 | 35 | 0 | 66 | 0.92 | clean |
| sat-engine-rtx5090-seed3 | b32-w0-r32 | 82 | 0.00 | 0.02 | 0 | 1 | 0 | 61 | 0.91 | clean |
| sat-engine-rtx5090-seed3 | b64-w0-r32 | 72 | 0.00 | 0.01 | 0 | 18 | 0 | 61 | 0.92 | clean |
| sat-hf-static-rtx5090-seed0 | b64-w20-r32 | 95 | 0.00 | 0.01 | 0 | 13 | 0 | 63 | 0.83 | clean |
| sat-hf-static-rtx5090-seed1 | b64-w20-r32 | 92 | 0.00 | 0.01 | 0 | 8 | 0 | 62 | 0.84 | clean |
| sat-hf-static-rtx5090-seed2 | b64-w20-r32 | 93 | 0.00 | 0.01 | 0 | 12 | 0 | 65 | 0.83 | clean |
| sat-hf-static-rtx5090-seed3 | b64-w20-r32 | 91 | 0.00 | 0.01 | 0 | 14 | 0 | 59 | 0.83 | clean |
| sat-vllm-eager-rtx5090-seed0 | b32-w0-r16 | 49 | 0.00 | 0.01 | 0 | 1 | 0 | 49 | 0.94 | clean |
| sat-vllm-eager-rtx5090-seed0 | b32-w0-r64 | 65 | 0.00 | 0.01 | 0 | 3 | 0 | 55 | 0.93 | clean |
| sat-vllm-eager-rtx5090-seed1 | b32-w0-r16 | 50 | 0.00 | 0.01 | 0 | 2 | 0 | 48 | 0.94 | clean |
| sat-vllm-eager-rtx5090-seed1 | b32-w0-r64 | 66 | 0.00 | 0.01 | 0 | 8 | 0 | 54 | 0.93 | clean |
| sat-vllm-rtx5090-seed0 | b32-w0-r64 | 52 | 0.00 | 0.01 | 0 | 29 | 0 | 62 | 0.90 | clean |
| sat-vllm-rtx5090-seed1 | b32-w0-r64 | 52 | 0.00 | 0.01 | 0 | 29 | 0 | 64 | 0.89 | clean |
| sat-vllm-rtx5090-seed2 | b32-w0-r64 | 52 | 0.00 | 0.01 | 0 | 30 | 0 | 66 | 0.90 | clean |
| sat-vllm-rtx5090-seed3 | b32-w0-r64 | 51 | 0.00 | 0.01 | 0 | 30 | 0 | 61 | 0.90 | clean |

20 runs; 20 clean; 0 flagged.
