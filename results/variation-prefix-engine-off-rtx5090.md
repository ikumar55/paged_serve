# Run-to-run variation

Configurations with at least 2 valid runs. Each cell is the mean with the minimum to maximum in brackets. Runs differ in workload seed (and in time), so this includes both system noise and which requests were drawn.

| batch | window ms | rate req/s | runs | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | e2e p99 s | req/s range / mean |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 32 | 0 | 4 | 3 | 3.93 (3.79 to 4.09) | 557 (515 to 579) | 41 (41 to 41) | 78 (71 to 83) | 3.8 (3.6 to 4.0) | 8% |
| 32 | 0 | 8 | 2 | 7.74 (7.60 to 7.89) | 1135 (1082 to 1188) | 46 (46 to 47) | 109 (90 to 129) | 5.2 (5.1 to 5.4) | 4% |
| 32 | 0 | 16 | 3 | 10.93 (10.74 to 11.16) | 1575 (1545 to 1593) | 13143 (11955 to 15368) | 23940 (22221 to 27265) | 27.8 (26.2 to 30.8) | 4% |
