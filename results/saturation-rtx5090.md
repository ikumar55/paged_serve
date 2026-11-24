# Saturation (capacity) report

Offered rates are above each system's capacity, so delivered throughput is the capacity (not capped by the arrivals). Seeds are the same workload draws for every system.

## Per run

| system | seed | req/s | tokens/s | TTFT p50 ms | ITL p50 ms | ok / sent | flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| engine b32 | 0 | 11.57 | 1683 | 56033 | 14 | 1646/1920 | - |
| engine b32 | 1 | 12.04 | 1713 | 53976 | 14 | 1728/1920 | - |
| engine b32 | 2 | 11.15 | 1606 | 61055 | 14 | 1692/1920 | - |
| engine b32 | 3 | 11.69 | 1683 | 59537 | 13 | 1741/1920 | - |
| engine b64 | 0 | 13.22 | 1921 | 46850 | 27 | 1737/1920 | - |
| engine b64 | 1 | 15.99 | 2283 | 26705 | 25 | 1918/1920 | - |
| engine b64 | 2 | 13.73 | 1976 | 35759 | 29 | 1838/1920 | - |
| engine b64 | 3 | 13.85 | 2003 | 40287 | 26 | 1806/1920 | - |
| hf-static b64 | 0 | 6.17 | 902 | 74249 | 15 | 1085/1920 | - |
| hf-static b64 | 1 | 7.07 | 993 | 82118 | 14 | 1167/1920 | - |
| hf-static b64 | 2 | 6.03 | 854 | 77104 | 14 | 1030/1920 | - |
| hf-static b64 | 3 | 8.48 | 1209 | 73953 | 14 | 1407/1920 | - |
| vllm 32 seqs | 0 | 59.63 | 8598 | 2197 | 3 | 3840/3840 | generator-saturated |
| vllm 32 seqs | 1 | 59.36 | 8602 | 2151 | 3 | 3840/3840 | - |
| vllm 32 seqs | 2 | 58.73 | 8495 | 3034 | 3 | 3840/3840 | - |
| vllm 32 seqs | 3 | 60.78 | 8824 | 1111 | 3 | 3840/3840 | - |

## Paired per-seed ratios

| ratio | metric | per seed | mean (min to max) |
| --- | --- | --- | --- |
| engine b64 / hf b64 | req/s | 2.14, 2.26, 2.28, 1.63 | 2.08 (1.63 to 2.28) |
| engine b64 / hf b64 | tokens/s | 2.13, 2.30, 2.31, 1.66 | 2.10 (1.66 to 2.31) |
| engine b32 / hf b64 | req/s | 1.87, 1.70, 1.85, 1.38 | 1.70 (1.38 to 1.87) |
| engine b32 / hf b64 | tokens/s | 1.87, 1.73, 1.88, 1.39 | 1.72 (1.39 to 1.88) |
| vllm / engine b64 | req/s | 4.51, 3.71, 4.28, 4.39 | 4.22 (3.71 to 4.51) |
| vllm / engine b64 | tokens/s | 4.48, 3.77, 4.30, 4.40 | 4.24 (3.77 to 4.48) |
| vllm / engine b32 | req/s | 5.15, 4.93, 5.27, 5.20 | 5.14 (4.93 to 5.27) |
| vllm / engine b32 | tokens/s | 5.11, 5.02, 5.29, 5.24 | 5.17 (5.02 to 5.29) |

## Engine thread time (D-021)

| run | seed | step share | route share | idle share | ms per step | ms per token | mean decode batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| engine b32 | 0 | 98% | 1.9% | 0.2% | 18.1 | 0.587 | 30.6 |
| engine b32 | 1 | 98% | 2.0% | 0.2% | 18.2 | 0.575 | 31.4 |
| engine b32 | 2 | 98% | 1.9% | 0.2% | 19.3 | 0.613 | 31.2 |
| engine b32 | 3 | 98% | 2.2% | 0.2% | 18.1 | 0.583 | 30.7 |
| engine b64 | 0 | 97% | 2.4% | 0.3% | 30.1 | 0.510 | 58.5 |
| engine b64 | 1 | 97% | 2.8% | 0.3% | 25.0 | 0.428 | 58.0 |
| engine b64 | 2 | 97% | 2.4% | 0.2% | 28.8 | 0.496 | 57.7 |
| engine b64 | 3 | 97% | 2.6% | 0.2% | 28.3 | 0.488 | 57.6 |
