# Router policy comparison, mock fleet (tight cache: 64 prefixes, 0.25 GB pool per replica)

**MOCK BACKEND, UNCALIBRATED TIMING. These numbers describe the routing policies and the plumbing on a CPU, with placeholder step times (D-008). They are not GPU performance and must not be quoted as such.**

4 mock replicas, workload seeds [0, 1, 2], 600 requests per run at 45 req/s. Each cell is the mean over seeds with the minimum to maximum in brackets; with this few seeds the brackets are a spread, not a confidence interval.

| variant | runs | prefix hit rate % | TTFT p50 ms | TTFT p99 ms | req/s | served imbalance | computed-token imbalance | failed requests | load fallbacks |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |
| round-robin | 3 | 43.9 (43.1 to 44.5) | 3914 (3754 to 4056) | 9555 (8637 to 10033) | 23.6 (22.8 to 24.6) | 1.00 (1.00 to 1.00) | 1.09 (1.05 to 1.12) | 0 | n/a |
| least-outstanding | 3 | 43.9 (43.2 to 45.1) | 4159 (4011 to 4418) | 9182 (8585 to 9580) | 23.3 (22.0 to 24.7) | 1.03 (1.02 to 1.04) | 1.05 (1.02 to 1.08) | 0 | n/a |
| approx-prefix | 3 | 46.3 (44.6 to 48.4) | 3894 (3694 to 4195) | 8823 (8523 to 9335) | 23.4 (23.0 to 24.2) | 1.03 (1.03 to 1.05) | 1.05 (1.02 to 1.10) | 0 | 1510 |
| precise-prefix | 3 | 43.6 (42.2 to 46.1) | 3986 (3631 to 4417) | 8712 (7450 to 10197) | 23.5 (22.4 to 25.4) | 1.04 (1.02 to 1.08) | 1.04 (1.02 to 1.06) | 0 | 1253 |
| approx-prefix[max-entries=256] | 3 | 44.9 (44.0 to 45.8) | 4053 (3948 to 4230) | 9240 (9029 to 9413) | 22.9 (22.3 to 23.3) | 1.06 (1.03 to 1.08) | 1.05 (1.02 to 1.11) | 0 | 544 |
| approx-prefix[max-entries=1024] | 3 | 45.3 (43.4 to 46.6) | 4066 (3967 to 4202) | 8994 (8240 to 9926) | 23.7 (22.7 to 24.8) | 1.05 (1.03 to 1.07) | 1.07 (1.02 to 1.12) | 0 | 1171 |

Imbalance is the busiest replica's share relative to an even split (1.00 is perfectly even). Computed tokens are prompt tokens a replica really ran through the model, i.e. not served from its prefix cache.

952 preemptions happened across the runs; prompt tokens re-admitted after a preemption are counted again in the hit rate.

- Placeholder mock timing (see mock_timing), not calibrated against any real run.
- No warm-up: every policy starts with cold caches and pays for its first sight of each prefix.
- Fresh replica and router processes for every run; variants of a seed receive the identical workload (see workload_digest).
- Run on Windows: sleeps and scheduling here are not those of a Linux box, and the replicas, router and load generator share one machine.
- The shared-prefix workload shape (groups, length, skew) is a modelling choice, not a measurement of real traffic.
