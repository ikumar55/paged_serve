# Run-to-run variation

Configurations with at least 2 valid runs. Each cell is the mean with the minimum to maximum in brackets. Runs differ in workload seed (and in time), so this includes both system noise and which requests were drawn.

| batch | window ms | rate req/s | runs | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | e2e p99 s | req/s range / mean |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 32 | 0 | 4 | 3 | 4.05 (3.94 to 4.20) | 572 (536 to 593) | 25 (24 to 26) | 41 (39 to 46) | 1.0 (0.9 to 1.0) | 6% |
| 32 | 0 | 8 | 3 | 8.08 (7.90 to 8.25) | 1168 (1126 to 1243) | 24 (23 to 24) | 43 (43 to 43) | 1.0 (0.9 to 1.1) | 4% |
| 32 | 0 | 16 | 3 | 15.78 (15.50 to 16.23) | 2274 (2196 to 2408) | 22 (22 to 23) | 42 (39 to 44) | 1.1 (1.1 to 1.3) | 5% |
