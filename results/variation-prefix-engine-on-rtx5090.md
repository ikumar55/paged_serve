# Run-to-run variation

Configurations with at least 2 valid runs. Each cell is the mean with the minimum to maximum in brackets. Runs differ in workload seed (and in time), so this includes both system noise and which requests were drawn.

| batch | window ms | rate req/s | runs | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | e2e p99 s | req/s range / mean |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 32 | 0 | 4 | 3 | 3.94 (3.78 to 4.10) | 557 (515 to 579) | 32 (32 to 33) | 67 (55 to 80) | 3.6 (3.4 to 4.1) | 8% |
| 32 | 0 | 8 | 3 | 7.81 (7.63 to 7.93) | 1129 (1087 to 1185) | 38 (37 to 39) | 89 (80 to 99) | 4.8 (4.7 to 5.0) | 4% |
| 32 | 0 | 16 | 2 | 12.17 (11.83 to 12.51) | 1768 (1755 to 1780) | 8524 (6193 to 10856) | 16128 (13159 to 19096) | 20.0 (17.4 to 22.6) | 6% |
