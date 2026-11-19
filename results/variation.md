# Run-to-run variation

Configurations with at least 2 valid runs. Each cell is the mean with the minimum to maximum in brackets. Runs differ in workload seed (and in time), so this includes both system noise and which requests were drawn.

| batch | window ms | rate req/s | runs | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | e2e p99 s | req/s range / mean |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 16 | 20 | 4 | 4 | 3.65 (3.53 to 3.88) | 517 (480 to 548) | 3387 (2366 to 4529) | 7249 (5104 to 8965) | 9.2 (7.3 to 10.6) | 10% |
| 32 | 20 | 4 | 4 | 3.78 (3.56 to 4.06) | 536 (484 to 573) | 2089 (1855 to 2304) | 6099 (4992 to 7230) | 8.2 (7.1 to 9.0) | 13% |
| 64 | 20 | 4 | 4 | 3.73 (3.51 to 3.93) | 529 (478 to 556) | 2123 (1797 to 2887) | 5942 (4344 to 6707) | 8.2 (6.6 to 9.0) | 11% |
| 16 | 20 | 8 | 4 | 4.02 (3.76 to 4.16) | 579 (566 to 592) | 29048 (26698 to 31659) | 56855 (52261 to 64896) | 58.3 (54.0 to 66.1) | 10% |
| 32 | 20 | 8 | 4 | 5.26 (4.95 to 5.67) | 757 (730 to 797) | 14470 (12352 to 16550) | 29832 (25051 to 37401) | 32.1 (27.1 to 39.3) | 14% |
| 64 | 20 | 8 | 4 | 5.31 (4.66 to 5.65) | 764 (702 to 806) | 10281 (9227 to 12892) | 29548 (22865 to 43268) | 32.2 (27.1 to 45.0) | 19% |
