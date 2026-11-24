# Engine sweep (continuous batching, paged KV cache)

2 runs, 0 valid, 0 sustained (valid and throughput >= 90% of the arrival rate).

Excluded from the best/worst comparison (still in the table): b32-w0-r32 (91% succeeded), b64-w0-r32 (94% succeeded)

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 32 | 1741/1920 | 11.69 | 1683 | 59537 | 89974 | 13 | 61.7 | 92.2 | 30.7 | - | 91% succeeded |
| 64 | 0 | 32 | 1806/1920 | 13.85 | 2003 | 40287 | 67253 | 26 | 44.7 | 72.0 | 57.6 | - | 94% succeeded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
