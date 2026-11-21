# Router policy comparison, mock fleet (precise policy polling every 0.05 s instead of 0.5 s: same workload as the load cap sweep, cap off)

**MOCK BACKEND, UNCALIBRATED TIMING. These numbers describe the routing policies and the plumbing on a CPU, with placeholder step times (D-008). They are not GPU performance and must not be quoted as such.**

4 mock replicas, workload seeds [0, 1, 2], 1200 requests per run at 30 req/s. Each cell is the mean over seeds with the minimum to maximum in brackets; with this few seeds the brackets are a spread, not a confidence interval.

| variant | runs | prefix hit rate % | TTFT p50 ms | TTFT p99 ms | req/s | served imbalance | computed-token imbalance | failed requests | load fallbacks |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |
| approx-prefix | 3 | 72.3 (71.4 to 72.8) | 65 (64 to 66) | 143 (139 to 150) | 29.0 (28.9 to 29.1) | 1.15 (1.14 to 1.16) | 1.42 (1.23 to 1.55) | 0 | 0 |
| precise-prefix | 3 | 71.9 (70.2 to 73.0) | 73 (71 to 76) | 157 (151 to 160) | 29.0 (28.9 to 29.2) | 1.22 (1.18 to 1.24) | 1.22 (1.13 to 1.32) | 0 | 0 |

Imbalance is the busiest replica's share relative to an even split (1.00 is perfectly even). Computed tokens are prompt tokens a replica really ran through the model, i.e. not served from its prefix cache.

1 preemptions happened across the runs; prompt tokens re-admitted after a preemption are counted again in the hit rate.

- Placeholder mock timing (see mock_timing), not calibrated against any real run.
- No warm-up: every policy starts with cold caches and pays for its first sight of each prefix.
- Fresh replica and router processes for every run; variants of a seed receive the identical workload (see workload_digest).
- Run on Windows: sleeps and scheduling here are not those of a Linux box, and the replicas, router and load generator share one machine.
- The shared-prefix workload shape (groups, length, skew) is a modelling choice, not a measurement of real traffic.
