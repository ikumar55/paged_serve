# Router policy comparison, mock fleet (tight cache, load cap off, lighter load: 64 prefixes, 0.25 GB pool per replica)

**MOCK BACKEND, UNCALIBRATED TIMING. These numbers describe the routing policies and the plumbing on a CPU, with placeholder step times (D-008). They are not GPU performance and must not be quoted as such.**

4 mock replicas, workload seeds [0, 1, 2], 600 requests per run at 30 req/s. Each cell is the mean over seeds with the minimum to maximum in brackets; with this few seeds the brackets are a spread, not a confidence interval.

| variant | runs | prefix hit rate % | TTFT p50 ms | TTFT p99 ms | req/s | served imbalance | computed-token imbalance | failed requests | load fallbacks |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |
| round-robin | 3 | 42.2 (40.9 to 43.1) | 78 (77 to 79) | 129 (122 to 133) | 27.9 (27.4 to 28.1) | 1.00 (1.00 to 1.00) | 1.06 (1.02 to 1.09) | 0 | n/a |
| least-outstanding | 3 | 43.7 (41.2 to 45.3) | 82 (79 to 84) | 153 (139 to 177) | 27.9 (27.5 to 28.2) | 1.02 (1.01 to 1.02) | 1.11 (1.07 to 1.14) | 0 | n/a |
| approx-prefix | 3 | 70.9 (69.7 to 72.7) | 64 (64 to 64) | 143 (132 to 162) | 27.9 (27.4 to 28.4) | 1.13 (1.09 to 1.19) | 1.31 (1.18 to 1.39) | 0 | 0 |
| precise-prefix | 3 | 67.8 (66.8 to 68.9) | 66 (64 to 67) | 140 (131 to 153) | 28.0 (27.6 to 28.3) | 1.12 (1.08 to 1.19) | 1.28 (1.22 to 1.34) | 0 | 0 |
| approx-prefix[max-entries=256] | 3 | 46.2 (44.0 to 48.4) | 81 (79 to 82) | 168 (161 to 172) | 27.9 (27.4 to 28.2) | 1.05 (1.02 to 1.07) | 1.07 (1.04 to 1.11) | 0 | 0 |
| approx-prefix[max-entries=1024] | 3 | 64.2 (63.2 to 65.3) | 67 (65 to 68) | 144 (138 to 153) | 28.0 (27.5 to 28.3) | 1.16 (1.13 to 1.21) | 1.33 (1.26 to 1.42) | 0 | 0 |

Imbalance is the busiest replica's share relative to an even split (1.00 is perfectly even). Computed tokens are prompt tokens a replica really ran through the model, i.e. not served from its prefix cache.

- Placeholder mock timing (see mock_timing), not calibrated against any real run.
- No warm-up: every policy starts with cold caches and pays for its first sight of each prefix.
- Fresh replica and router processes for every run; variants of a seed receive the identical workload (see workload_digest).
- Run on Windows: sleeps and scheduling here are not those of a Linux box, and the replicas, router and load generator share one machine.
- The shared-prefix workload shape (groups, length, skew) is a modelling choice, not a measurement of real traffic.
