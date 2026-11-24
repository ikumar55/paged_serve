# vLLM sweep

2 runs, 1 valid, 1 sustained (valid and throughput >= 90% of the arrival rate).

Excluded from the best/worst comparison (still in the table): b32-w0-r64 (load generator saturated)

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 16 | sustained runs | 32, 0 | 57 | 32, 0 | 57 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 16 | 960/960 | 15.41 | 2192 | 29 | 57 | 7 | 0.9 | 2.7 | - | - | sustained |
| 32 | 0 | 64 | 2847/3840 | 29.56 | 4298 | 23203 | 34016 | 7 | 24.1 | 35.9 | - | - | load generator saturated |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
