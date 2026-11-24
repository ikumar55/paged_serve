# Machine audit

Other users' CPU is in cores (1.0 = one core busy). Samples are those the monitor took between the run's start and finish (server start-up included). A run with no coverage is not known to be clean.

| sweep | run | samples | other CPU mean | other CPU peak | other GPU procs | power-capped samples | slowdown samples | peak temp C | min clock ratio | flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prefix-engine-off-rtx5090-seed0 | b32-w0-r4 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-off-rtx5090-seed0 | b32-w0-r8 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-off-rtx5090-seed0 | b32-w0-r16 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-off-rtx5090-seed1 | b32-w0-r4 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-off-rtx5090-seed1 | b32-w0-r8 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-off-rtx5090-seed1 | b32-w0-r16 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-off-rtx5090-seed2 | b32-w0-r4 | 24 | 0.00 | 0.01 | 0 | 2 | 0 | 49 | 0.94 | clean |
| prefix-engine-off-rtx5090-seed2 | b32-w0-r8 | 41 | 0.00 | 0.01 | 0 | 4 | 0 | 54 | 0.92 | clean |
| prefix-engine-off-rtx5090-seed2 | b32-w0-r16 | 52 | 0.00 | 0.01 | 0 | 10 | 0 | 59 | 0.92 | clean |
| prefix-engine-on-rtx5090-seed0 | b32-w0-r4 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-on-rtx5090-seed0 | b32-w0-r8 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-on-rtx5090-seed0 | b32-w0-r16 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-on-rtx5090-seed1 | b32-w0-r4 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-on-rtx5090-seed1 | b32-w0-r8 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-on-rtx5090-seed1 | b32-w0-r16 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-on-rtx5090-seed2 | b32-w0-r4 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-on-rtx5090-seed2 | b32-w0-r8 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-engine-on-rtx5090-seed2 | b32-w0-r16 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-off-rtx5090-seed0 | b32-w0-r4 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-off-rtx5090-seed0 | b32-w0-r8 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-off-rtx5090-seed0 | b32-w0-r16 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-off-rtx5090-seed1 | b32-w0-r4 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-off-rtx5090-seed1 | b32-w0-r8 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-off-rtx5090-seed1 | b32-w0-r16 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-off-rtx5090-seed2 | b32-w0-r4 | 48 | 0.00 | 0.01 | 0 | 8 | 0 | 54 | 0.94 | clean |
| prefix-vllm-off-rtx5090-seed2 | b32-w0-r8 | 52 | 0.00 | 0.01 | 0 | 7 | 0 | 55 | 0.93 | clean |
| prefix-vllm-off-rtx5090-seed2 | b32-w0-r16 | 51 | 0.00 | 0.00 | 0 | 18 | 0 | 62 | 0.91 | clean |
| prefix-vllm-on-rtx5090-seed0 | b32-w0-r4 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-on-rtx5090-seed0 | b32-w0-r8 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-on-rtx5090-seed0 | b32-w0-r16 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-on-rtx5090-seed1 | b32-w0-r4 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-on-rtx5090-seed1 | b32-w0-r8 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-on-rtx5090-seed1 | b32-w0-r16 | 0 | 0.00 | 0.00 | 0 | 0 | 0 | - | - | no-coverage |
| prefix-vllm-on-rtx5090-seed2 | b32-w0-r4 | 50 | 0.00 | 0.01 | 0 | 1 | 0 | 52 | 0.94 | clean |
| prefix-vllm-on-rtx5090-seed2 | b32-w0-r8 | 51 | 0.00 | 0.01 | 0 | 0 | 0 | 53 | 0.94 | clean |
| prefix-vllm-on-rtx5090-seed2 | b32-w0-r16 | 52 | 0.00 | 0.01 | 0 | 0 | 0 | 55 | 0.94 | clean |
| repro-engine-rtx5090-seed1 | b32-w0-r8 | 40 | 0.00 | 0.00 | 0 | 0 | 0 | 51 | 0.93 | clean |
| repro-engine-rtx5090-seed1 | b32-w0-r16 | 44 | 0.00 | 0.01 | 0 | 0 | 0 | 57 | 0.92 | clean |
| repro-engine-rtx5090-seed1 | b64-w0-r8 | 40 | 0.00 | 0.00 | 0 | 0 | 0 | 53 | 0.93 | clean |
| repro-engine-rtx5090-seed1 | b64-w0-r16 | 41 | 0.00 | 0.01 | 0 | 5 | 0 | 60 | 0.92 | clean |
| repro-engine-rtx5090-seed2 | b32-w0-r8 | 41 | 0.00 | 0.01 | 0 | 0 | 0 | 52 | 0.93 | clean |
| repro-engine-rtx5090-seed2 | b32-w0-r16 | 45 | 0.00 | 0.01 | 0 | 1 | 0 | 56 | 0.92 | clean |
| repro-engine-rtx5090-seed2 | b64-w0-r8 | 41 | 0.00 | 0.01 | 0 | 1 | 0 | 53 | 0.93 | clean |
| repro-engine-rtx5090-seed2 | b64-w0-r16 | 42 | 0.00 | 0.01 | 0 | 13 | 0 | 60 | 0.92 | clean |
| repro-hf-static-rtx5090-seed1 | b32-w20-r8 | 42 | 0.00 | 0.01 | 0 | 2 | 0 | 59 | 0.87 | clean |
| repro-hf-static-rtx5090-seed1 | b32-w20-r16 | 66 | 0.00 | 0.01 | 0 | 10 | 0 | 60 | 0.84 | clean |
| repro-hf-static-rtx5090-seed1 | b64-w20-r8 | 43 | 0.00 | 0.01 | 0 | 4 | 0 | 55 | 0.85 | clean |
| repro-hf-static-rtx5090-seed1 | b64-w20-r16 | 59 | 0.00 | 0.01 | 0 | 6 | 0 | 58 | 0.84 | clean |
| repro-hf-static-rtx5090-seed2 | b32-w20-r8 | 44 | 0.00 | 0.01 | 0 | 1 | 0 | 55 | 0.86 | clean |
| repro-hf-static-rtx5090-seed2 | b32-w20-r16 | 68 | 0.00 | 0.01 | 0 | 9 | 0 | 62 | 0.84 | clean |
| repro-hf-static-rtx5090-seed2 | b64-w20-r8 | 44 | 0.00 | 0.01 | 0 | 5 | 0 | 60 | 0.84 | clean |
| repro-hf-static-rtx5090-seed2 | b64-w20-r16 | 61 | 0.00 | 0.01 | 0 | 11 | 0 | 64 | 0.83 | clean |
| repro-vllm-rtx5090-seed1 | b32-w0-r16 | 50 | 0.00 | 0.01 | 0 | 0 | 0 | 53 | 0.94 | clean |
| repro-vllm-rtx5090-seed1 | b32-w0-r48 | 50 | 0.00 | 0.00 | 0 | 26 | 0 | 61 | 0.90 | clean |
| repro-vllm-rtx5090-seed2 | b32-w0-r16 | 51 | 0.00 | 0.01 | 0 | 6 | 0 | 58 | 0.92 | clean |
| repro-vllm-rtx5090-seed2 | b32-w0-r48 | 49 | 0.00 | 0.00 | 0 | 28 | 0 | 66 | 0.90 | clean |

56 runs; 29 clean; 27 flagged: prefix-engine-off-rtx5090-seed0/b32-w0-r4, prefix-engine-off-rtx5090-seed0/b32-w0-r8, prefix-engine-off-rtx5090-seed0/b32-w0-r16, prefix-engine-off-rtx5090-seed1/b32-w0-r4, prefix-engine-off-rtx5090-seed1/b32-w0-r8, prefix-engine-off-rtx5090-seed1/b32-w0-r16, prefix-engine-on-rtx5090-seed0/b32-w0-r4, prefix-engine-on-rtx5090-seed0/b32-w0-r8, prefix-engine-on-rtx5090-seed0/b32-w0-r16, prefix-engine-on-rtx5090-seed1/b32-w0-r4, prefix-engine-on-rtx5090-seed1/b32-w0-r8, prefix-engine-on-rtx5090-seed1/b32-w0-r16, prefix-engine-on-rtx5090-seed2/b32-w0-r4, prefix-engine-on-rtx5090-seed2/b32-w0-r8, prefix-engine-on-rtx5090-seed2/b32-w0-r16, prefix-vllm-off-rtx5090-seed0/b32-w0-r4, prefix-vllm-off-rtx5090-seed0/b32-w0-r8, prefix-vllm-off-rtx5090-seed0/b32-w0-r16, prefix-vllm-off-rtx5090-seed1/b32-w0-r4, prefix-vllm-off-rtx5090-seed1/b32-w0-r8, prefix-vllm-off-rtx5090-seed1/b32-w0-r16, prefix-vllm-on-rtx5090-seed0/b32-w0-r4, prefix-vllm-on-rtx5090-seed0/b32-w0-r8, prefix-vllm-on-rtx5090-seed0/b32-w0-r16, prefix-vllm-on-rtx5090-seed1/b32-w0-r4, prefix-vllm-on-rtx5090-seed1/b32-w0-r8, prefix-vllm-on-rtx5090-seed1/b32-w0-r16
