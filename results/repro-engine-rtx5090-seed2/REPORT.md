# Engine sweep (continuous batching, paged KV cache)

4 runs, 4 valid, 3 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 8 | sustained runs | 32, 0 | 63 | 64, 0 | 71 |
| 16 | sustained runs | 64, 0 | 1265 | 64, 0 | 1265 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 8 | 480/480 | 7.66 | 1091 | 32 | 63 | 9 | 1.3 | 3.4 | 10.4 | - | sustained |
| 64 | 0 | 8 | 480/480 | 7.67 | 1093 | 33 | 71 | 9 | 1.4 | 3.8 | 11.0 | - | sustained |
| 32 | 0 | 16 | 960/960 | 13.02 | 1845 | 4113 | 8916 | 14 | 6.2 | 12.9 | 28.0 | - | overloaded |
| 64 | 0 | 16 | 960/960 | 14.47 | 2051 | 110 | 1265 | 24 | 3.5 | 10.0 | 43.8 | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
