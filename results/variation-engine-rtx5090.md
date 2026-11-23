# Run-to-run variation

Configurations with at least 2 valid runs. Each cell is the mean with the minimum to maximum in brackets. Runs differ in workload seed (and in time), so this includes both system noise and which requests were drawn.

| batch | window ms | rate req/s | runs | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | e2e p99 s | req/s range / mean |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 32 | 0 | 4 | 4 | 3.94 (3.77 to 4.08) | 559 (513 to 576) | 33 (32 to 34) | 69 (60 to 84) | 4.0 (3.5 to 4.7) | 8% |
| 64 | 0 | 4 | 4 | 3.95 (3.78 to 4.11) | 560 (514 to 581) | 32 (30 to 33) | 62 (59 to 66) | 3.7 (3.4 to 4.5) | 8% |
| 32 | 0 | 8 | 3 | 7.69 (7.59 to 7.86) | 1116 (1082 to 1183) | 37 (35 to 40) | 77 (67 to 85) | 4.5 (3.9 to 5.0) | 4% |
| 64 | 0 | 8 | 3 | 7.64 (7.50 to 7.85) | 1110 (1069 to 1182) | 37 (35 to 40) | 81 (68 to 89) | 4.6 (3.9 to 5.3) | 4% |
| 32 | 0 | 12 | 4 | 11.62 (11.36 to 11.87) | 1660 (1595 to 1750) | 237 (42 to 813) | 1281 (509 to 3322) | 6.2 (4.8 to 8.7) | 4% |
| 64 | 0 | 12 | 4 | 11.56 (11.33 to 11.81) | 1651 (1573 to 1740) | 45 (41 to 55) | 321 (82 to 996) | 7.0 (5.3 to 11.1) | 4% |
| 32 | 0 | 16 | 3 | 12.84 (11.81 to 13.62) | 1852 (1753 to 1937) | 6656 (3635 to 12338) | 12646 (7999 to 18955) | 16.6 (12.2 to 23.1) | 14% |
| 64 | 0 | 16 | 4 | 14.35 (13.52 to 15.13) | 2062 (2006 to 2153) | 1563 (71 to 5314) | 3529 (1641 to 8061) | 12.4 (10.4 to 17.4) | 11% |
