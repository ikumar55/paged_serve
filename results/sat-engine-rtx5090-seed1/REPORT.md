# Engine sweep (continuous batching, paged KV cache)

2 runs, 1 valid, 0 sustained (valid and throughput >= 90% of the arrival rate).

Excluded from the best/worst comparison (still in the table): b32-w0-r32 (90% succeeded)

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 32 | none sustained (least bad) | 64, 0 | 55299 | 64, 0 | 55299 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 32 | 1728/1920 | 12.04 | 1713 | 53976 | 86609 | 14 | 56.2 | 88.8 | 31.4 | - | 90% succeeded |
| 64 | 0 | 32 | 1918/1920 | 15.99 | 2283 | 26705 | 55299 | 25 | 32.0 | 59.2 | 58.0 | - | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
