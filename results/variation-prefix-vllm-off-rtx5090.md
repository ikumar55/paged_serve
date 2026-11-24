# Run-to-run variation

Configurations with at least 2 valid runs. Each cell is the mean with the minimum to maximum in brackets. Runs differ in workload seed (and in time), so this includes both system noise and which requests were drawn.

| batch | window ms | rate req/s | runs | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | e2e p99 s | req/s range / mean |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 32 | 0 | 4 | 3 | 4.04 (3.94 to 4.20) | 572 (536 to 593) | 33 (32 to 33) | 51 (46 to 58) | 1.0 (0.9 to 1.0) | 6% |
| 32 | 0 | 8 | 3 | 8.07 (7.90 to 8.25) | 1167 (1125 to 1242) | 34 (33 to 34) | 60 (54 to 68) | 1.1 (1.0 to 1.2) | 4% |
| 32 | 0 | 16 | 3 | 15.76 (15.48 to 16.21) | 2272 (2193 to 2405) | 36 (36 to 36) | 71 (67 to 73) | 1.4 (1.3 to 1.5) | 5% |
