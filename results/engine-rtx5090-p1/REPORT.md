# Engine sweep (continuous batching, paged KV cache)

8 runs, 8 valid, 6 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 2 | sustained runs | 64, 0 | 50 | 32, 0 | 66 |
| 4 | sustained runs | 64, 0 | 59 | 32, 0 | 70 |
| 8 | sustained runs | 32, 0 | 85 | 64, 0 | 89 |
| 16 | none sustained (least bad) | 64, 0 | 8061 | 32, 0 | 18955 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 2 | 120/120 | 1.89 | 284 | 31 | 66 | 9 | 1.3 | 3.5 | 2.8 | - | sustained |
| 64 | 0 | 2 | 120/120 | 1.88 | 282 | 32 | 50 | 9 | 1.4 | 3.4 | 2.8 | - | sustained |
| 32 | 0 | 4 | 240/240 | 3.92 | 576 | 33 | 70 | 10 | 1.4 | 3.7 | 5.9 | - | sustained |
| 64 | 0 | 4 | 240/240 | 3.91 | 576 | 33 | 59 | 10 | 1.4 | 3.4 | 5.8 | - | sustained |
| 32 | 0 | 8 | 480/480 | 7.86 | 1183 | 40 | 85 | 11 | 1.8 | 5.0 | 14.6 | - | sustained |
| 64 | 0 | 8 | 480/480 | 7.85 | 1182 | 40 | 89 | 11 | 1.8 | 5.3 | 15.3 | - | sustained |
| 32 | 0 | 16 | 960/960 | 11.81 | 1753 | 12338 | 18955 | 16 | 14.6 | 23.1 | 30.2 | - | overloaded |
| 64 | 0 | 16 | 960/960 | 13.52 | 2006 | 5314 | 8061 | 27 | 9.3 | 17.4 | 52.3 | - | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
