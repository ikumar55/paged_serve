# Engine sweep (continuous batching, paged KV cache)

4 runs, 4 valid, 3 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 8 | sustained runs | 64, 0 | 88 | 32, 0 | 91 |
| 16 | sustained runs | 64, 0 | 2338 | 64, 0 | 2338 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 8 | 480/480 | 7.90 | 1111 | 37 | 91 | 11 | 1.5 | 5.1 | 13.2 | - | sustained |
| 64 | 0 | 8 | 480/480 | 7.88 | 1107 | 36 | 88 | 10 | 1.4 | 5.1 | 12.4 | - | sustained |
| 32 | 0 | 16 | 960/960 | 13.71 | 1950 | 3766 | 7657 | 14 | 6.2 | 11.9 | 30.6 | - | overloaded |
| 64 | 0 | 16 | 960/960 | 15.06 | 2141 | 204 | 2338 | 23 | 3.8 | 11.0 | 49.2 | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
