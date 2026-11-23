# Run-to-run variation

Configurations with at least 2 valid runs. Each cell is the mean with the minimum to maximum in brackets. Runs differ in workload seed (and in time), so this includes both system noise and which requests were drawn.

| batch | window ms | rate req/s | runs | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | e2e p99 s | req/s range / mean |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 32 | 20 | 4 | 4 | 3.86 (3.72 to 4.07) | 548 (506 to 576) | 1446 (1301 to 1595) | 4362 (3246 to 5952) | 6.1 (4.9 to 7.6) | 9% |
| 64 | 20 | 4 | 4 | 3.84 (3.73 to 4.05) | 545 (508 to 573) | 1409 (1324 to 1503) | 3787 (3340 to 4168) | 5.9 (5.3 to 6.7) | 8% |
| 32 | 20 | 8 | 4 | 7.07 (6.50 to 7.43) | 1018 (979 to 1059) | 3549 (2915 to 4656) | 8226 (6106 to 14318) | 10.3 (8.3 to 16.0) | 13% |
| 64 | 20 | 8 | 4 | 7.20 (6.91 to 7.46) | 1037 (996 to 1064) | 2567 (2218 to 3335) | 6566 (5027 to 8932) | 8.9 (7.4 to 12.0) | 8% |
| 32 | 20 | 16 | 4 | 7.57 (6.80 to 8.03) | 1087 (1008 to 1142) | 32942 (28961 to 42043) | 64882 (57095 to 80136) | 66.5 (58.8 to 82.0) | 16% |
| 64 | 20 | 16 | 4 | 9.03 (8.44 to 9.44) | 1298 (1253 to 1342) | 21389 (17981 to 29295) | 44099 (39040 to 53011) | 46.0 (41.1 to 54.7) | 11% |
