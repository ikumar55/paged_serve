# Router policy comparison, mock fleet (ample cache: 8 prefixes, 4 GB pool per replica)

**MOCK BACKEND, UNCALIBRATED TIMING. These numbers describe the routing policies and the plumbing on a CPU, with placeholder step times (D-008). They are not GPU performance and must not be quoted as such.**

4 mock replicas, workload seeds [0, 1, 2], 600 requests per run at 45 req/s. Each cell is the mean over seeds with the minimum to maximum in brackets; with this few seeds the brackets are a spread, not a confidence interval.

| variant | runs | prefix hit rate % | TTFT p50 ms | TTFT p99 ms | req/s | served imbalance | computed-token imbalance | failed requests | load fallbacks |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |
| round-robin | 3 | 82.9 (82.6 to 83.0) | 160 (144 to 181) | 646 (463 to 832) | 34.1 (32.8 to 36.2) | 1.00 (1.00 to 1.00) | 1.04 (1.01 to 1.07) | 0 | n/a |
| least-outstanding | 3 | 82.9 (82.6 to 83.0) | 161 (150 to 181) | 397 (300 to 542) | 34.3 (33.0 to 36.5) | 1.03 (1.01 to 1.05) | 1.04 (1.02 to 1.06) | 0 | n/a |
| approx-prefix | 3 | 82.9 (82.6 to 83.0) | 177 (157 to 216) | 461 (331 to 675) | 34.3 (32.8 to 36.6) | 1.02 (1.01 to 1.04) | 1.05 (1.03 to 1.09) | 0 | 1598 |
| precise-prefix | 3 | 82.9 (82.6 to 83.0) | 177 (156 to 210) | 488 (353 to 730) | 34.3 (32.9 to 36.5) | 1.03 (1.02 to 1.05) | 1.06 (1.03 to 1.10) | 0 | 1579 |

Imbalance is the busiest replica's share relative to an even split (1.00 is perfectly even). Computed tokens are prompt tokens a replica really ran through the model, i.e. not served from its prefix cache.

WARNING: in some runs the load generator sent late; latencies are understated.

- Placeholder mock timing (see mock_timing), not calibrated against any real run.
- No warm-up: every policy starts with cold caches and pays for its first sight of each prefix.
- Fresh replica and router processes for every run; variants of a seed receive the identical workload (see workload_digest).
- Run on Windows: sleeps and scheduling here are not those of a Linux box, and the replicas, router and load generator share one machine.
- The shared-prefix workload shape (groups, length, skew) is a modelling choice, not a measurement of real traffic.
