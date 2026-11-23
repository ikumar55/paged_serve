# Static batching sweep

6 runs, 6 valid, 4 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 32, 20 | 3246 | 64, 20 | 3692 |
| 8 | sustained runs | 64, 20 | 5027 | 32, 20 | 6149 |
| 16 | none sustained (least bad) | 64, 20 | 39040 | 32, 20 | 57095 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 20 | 4 | 240/240 | 3.72 | 506 | 1301 | 3246 | 9 | 2.5 | 4.9 | 8.1 | 51% | sustained |
| 64 | 20 | 4 | 240/240 | 3.73 | 508 | 1503 | 3692 | 9 | 2.7 | 5.4 | 8.7 | 52% | sustained |
| 32 | 20 | 8 | 480/480 | 7.30 | 1026 | 3053 | 6149 | 10 | 4.4 | 8.3 | 23.0 | 60% | sustained |
| 64 | 20 | 8 | 480/480 | 7.45 | 1047 | 2369 | 5027 | 10 | 3.9 | 7.4 | 23.0 | 59% | sustained |
| 32 | 20 | 16 | 960/960 | 8.03 | 1142 | 29073 | 57095 | 10 | 30.5 | 58.8 | 28.3 | 60% | overloaded |
| 64 | 20 | 16 | 960/960 | 9.44 | 1342 | 17981 | 39040 | 14 | 20.3 | 41.1 | 48.1 | 64% | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
