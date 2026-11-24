# Machine audit

Other users' CPU is in cores (1.0 = one core busy). Samples are those the monitor took between the run's start and finish (server start-up included). A run with no coverage is not known to be clean.

| sweep | run | samples | other CPU mean | other CPU peak | other GPU procs | power-capped samples | slowdown samples | peak temp C | min clock ratio | flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vllm-rtx5090-ceiling | b32-w0-r64 | 54 | 0.00 | 0.01 | 0 | 31 | 0 | 59 | 0.90 | clean |
| vllm-rtx5090-ceiling | b32-w0-r96 | 57 | 0.00 | 0.01 | 0 | 34 | 0 | 65 | 0.89 | clean |
| vllm-rtx5090-ceiling | b32-w0-r128 | 58 | 0.00 | 0.01 | 0 | 37 | 0 | 67 | 0.89 | clean |
| vllm-rtx5090-ceiling | b64-w0-r64 | 49 | 0.00 | 0.01 | 0 | 28 | 0 | 67 | 0.89 | clean |
| vllm-rtx5090-ceiling | b64-w0-r96 | 56 | 0.00 | 0.01 | 0 | 34 | 0 | 67 | 0.88 | clean |
| vllm-rtx5090-ceiling | b64-w0-r128 | 55 | 0.00 | 0.01 | 0 | 33 | 0 | 68 | 0.87 | clean |

6 runs; 6 clean; 0 flagged.
