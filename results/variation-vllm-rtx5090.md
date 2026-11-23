# Run-to-run variation

Configurations with at least 2 valid runs. Each cell is the mean with the minimum to maximum in brackets. Runs differ in workload seed (and in time), so this includes both system noise and which requests were drawn.

| batch | window ms | rate req/s | runs | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | e2e p99 s | req/s range / mean |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 32 | 0 | 4 | 4 | 4.06 (3.94 to 4.20) | 576 (536 to 593) | 24 (22 to 25) | 38 (36 to 43) | 1.0 (0.9 to 1.0) | 6% |
| 32 | 0 | 8 | 4 | 8.01 (7.83 to 8.25) | 1155 (1116 to 1243) | 22 (22 to 23) | 44 (41 to 46) | 1.0 (0.9 to 1.1) | 5% |
| 32 | 0 | 16 | 4 | 15.80 (15.50 to 16.24) | 2272 (2196 to 2410) | 23 (22 to 23) | 44 (40 to 46) | 1.1 (1.1 to 1.3) | 5% |
| 32 | 0 | 24 | 4 | 23.66 (23.16 to 23.97) | 3395 (3309 to 3500) | 23 (23 to 24) | 46 (42 to 51) | 1.2 (1.1 to 1.3) | 3% |
| 32 | 0 | 32 | 4 | 31.73 (31.23 to 31.94) | 4566 (4504 to 4618) | 23 (23 to 24) | 54 (52 to 56) | 1.2 (1.2 to 1.2) | 2% |
| 32 | 0 | 48 | 4 | 47.67 (47.30 to 48.30) | 6880 (6832 to 6943) | 25 (24 to 27) | 214 (98 to 420) | 1.4 (1.2 to 1.5) | 2% |
