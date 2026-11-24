# Repeat against original

| system | sweep | run | original req/s | repeat req/s | diff % | original TTFT p99 ms | repeat TTFT p99 ms | original contended | repeat contended |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| engine | repro-engine-rtx5090-seed1 | b32-w0-r8 | 7.90 | 7.90 | +0.1 | 79 | 91 | True | False |
| engine | repro-engine-rtx5090-seed1 | b64-w0-r8 | 7.88 | 7.88 | -0.1 | 70 | 88 | True | False |
| engine | repro-engine-rtx5090-seed1 | b32-w0-r16 | 13.62 | 13.71 | +0.7 | 7999 | 7657 | False | False |
| engine | repro-engine-rtx5090-seed1 | b64-w0-r16 | 15.13 | 15.06 | -0.5 | 1982 | 2338 | False | False |
| engine | repro-engine-rtx5090-seed2 | b32-w0-r8 | 7.61 | 7.66 | +0.7 | 79 | 63 | False | False |
| engine | repro-engine-rtx5090-seed2 | b64-w0-r8 | 7.50 | 7.67 | +2.2 | 86 | 71 | False | False |
| engine | repro-engine-rtx5090-seed2 | b32-w0-r16 | 12.62 | 13.02 | +3.2 | 10496 | 8916 | True | False |
| engine | repro-engine-rtx5090-seed2 | b64-w0-r16 | 14.23 | 14.47 | +1.7 | 1641 | 1265 | False | False |
| hf-static | repro-hf-static-rtx5090-seed1 | b32-w20-r8 | 7.30 | 7.45 | +2.0 | 6149 | 6612 | False | False |
| hf-static | repro-hf-static-rtx5090-seed1 | b64-w20-r8 | 7.45 | 7.45 | -0.1 | 5027 | 6415 | False | False |
| hf-static | repro-hf-static-rtx5090-seed1 | b32-w20-r16 | 8.03 | 8.39 | +4.6 | 57095 | 52015 | False | False |
| hf-static | repro-hf-static-rtx5090-seed1 | b64-w20-r16 | 9.44 | 9.62 | +1.9 | 39040 | 39013 | False | False |
| hf-static | repro-hf-static-rtx5090-seed2 | b32-w20-r8 | 7.06 | 7.07 | +0.1 | 6330 | 6278 | False | False |
| hf-static | repro-hf-static-rtx5090-seed2 | b64-w20-r8 | 6.99 | 7.10 | +1.4 | 5894 | 4759 | False | False |
| hf-static | repro-hf-static-rtx5090-seed2 | b32-w20-r16 | 7.70 | 8.01 | +4.1 | 60272 | 54540 | False | False |
| hf-static | repro-hf-static-rtx5090-seed2 | b64-w20-r16 | 8.98 | 9.11 | +1.5 | 42086 | 41617 | False | False |
| vllm | repro-vllm-rtx5090-seed1 | b32-w0-r16 | 15.60 | 15.60 | +0.0 | 40 | 42 | False | False |
| vllm | repro-vllm-rtx5090-seed1 | b32-w0-r48 | 47.54 | 47.54 | -0.0 | 98 | 102 | False | False |
| vllm | repro-vllm-rtx5090-seed2 | b32-w0-r16 | 15.50 | 15.50 | +0.0 | 44 | 42 | False | False |
| vllm | repro-vllm-rtx5090-seed2 | b32-w0-r48 | 48.30 | 48.31 | +0.0 | 420 | 113 | False | False |

Usable pairs (neither run contended):

- engine: 5 pairs, mean |diff| 1.2%, max |diff| 2.2%, mean signed +1.0%
- hf-static: 8 pairs, mean |diff| 1.9%, max |diff| 4.6%, mean signed +1.9%
- vllm: 4 pairs, mean |diff| 0.0%, max |diff| 0.0%, mean signed +0.0%
