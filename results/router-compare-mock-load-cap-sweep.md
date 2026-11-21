# Router policy comparison, mock fleet (load cap sweep: 64 prefixes, 0.25 GB pool per replica, 30 req/s, 1200 requests)

**MOCK BACKEND, UNCALIBRATED TIMING. These numbers describe the routing policies and the plumbing on a CPU, with placeholder step times (D-008). They are not GPU performance and must not be quoted as such.**

4 mock replicas, workload seeds [0, 1, 2], 1200 requests per run at 30 req/s. Each cell is the mean over seeds with the minimum to maximum in brackets; with this few seeds the brackets are a spread, not a confidence interval.

| variant | runs | prefix hit rate % | TTFT p50 ms | TTFT p99 ms | req/s | served imbalance | computed-token imbalance | failed requests | load fallbacks |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |
| round-robin | 3 | 43.3 (42.3 to 44.0) | 78 (76 to 79) | 133 (129 to 136) | 29.0 (28.8 to 29.2) | 1.00 (1.00 to 1.00) | 1.03 (1.02 to 1.03) | 0 | n/a |
| least-outstanding | 3 | 44.1 (43.1 to 44.9) | 83 (82 to 83) | 160 (153 to 170) | 29.0 (28.9 to 29.2) | 1.01 (1.00 to 1.01) | 1.07 (1.04 to 1.12) | 0 | n/a |
| approx-prefix[cap=4] | 3 | 44.4 (43.7 to 45.7) | 82 (82 to 83) | 157 (148 to 173) | 29.0 (28.9 to 29.2) | 1.02 (1.01 to 1.02) | 1.07 (1.04 to 1.10) | 0 | 3376 |
| approx-prefix[cap=8] | 3 | 52.3 (52.1 to 52.6) | 76 (74 to 79) | 157 (148 to 165) | 29.0 (28.8 to 29.2) | 1.02 (1.01 to 1.04) | 1.07 (1.05 to 1.11) | 0 | 1851 |
| approx-prefix[cap=16] | 3 | 71.5 (71.0 to 72.1) | 64 (63 to 65) | 151 (146 to 156) | 29.0 (28.9 to 29.2) | 1.09 (1.03 to 1.13) | 1.44 (1.32 to 1.51) | 0 | 27 |
| approx-prefix[cap=32] | 3 | 71.8 (71.7 to 72.0) | 64 (63 to 65) | 144 (138 to 155) | 29.0 (28.8 to 29.1) | 1.13 (1.09 to 1.19) | 1.34 (1.24 to 1.42) | 0 | 0 |
| approx-prefix[cap=1000] | 3 | 72.4 (71.1 to 73.9) | 64 (64 to 64) | 144 (139 to 153) | 29.0 (28.8 to 29.2) | 1.14 (1.10 to 1.20) | 1.32 (1.24 to 1.41) | 0 | 0 |
| precise-prefix[cap=4] | 3 | 44.5 (44.1 to 45.3) | 83 (82 to 84) | 165 (157 to 172) | 29.0 (28.9 to 29.2) | 1.01 (1.00 to 1.01) | 1.04 (1.03 to 1.06) | 0 | 2652 |
| precise-prefix[cap=8] | 3 | 54.9 (53.3 to 57.0) | 74 (72 to 75) | 158 (146 to 167) | 29.0 (28.8 to 29.1) | 1.02 (1.01 to 1.02) | 1.11 (1.07 to 1.16) | 0 | 1035 |
| precise-prefix[cap=16] | 3 | 70.4 (69.3 to 71.2) | 65 (64 to 66) | 149 (145 to 152) | 29.0 (28.8 to 29.2) | 1.06 (1.02 to 1.09) | 1.16 (1.13 to 1.18) | 0 | 11 |
| precise-prefix[cap=32] | 3 | 70.6 (68.3 to 72.5) | 65 (64 to 67) | 147 (138 to 154) | 29.0 (28.8 to 29.2) | 1.08 (1.04 to 1.13) | 1.33 (1.23 to 1.46) | 0 | 0 |
| precise-prefix[cap=1000] | 3 | 70.4 (68.4 to 71.6) | 65 (64 to 66) | 148 (139 to 154) | 29.0 (28.9 to 29.2) | 1.09 (1.04 to 1.14) | 1.29 (1.14 to 1.48) | 0 | 0 |

Imbalance is the busiest replica's share relative to an even split (1.00 is perfectly even). Computed tokens are prompt tokens a replica really ran through the model, i.e. not served from its prefix cache.

- Placeholder mock timing (see mock_timing), not calibrated against any real run.
- No warm-up: every policy starts with cold caches and pays for its first sight of each prefix.
- Fresh replica and router processes for every run; variants of a seed receive the identical workload (see workload_digest).
- Run on Windows: sleeps and scheduling here are not those of a Linux box, and the replicas, router and load generator share one machine.
- The shared-prefix workload shape (groups, length, skew) is a modelling choice, not a measurement of real traffic.
