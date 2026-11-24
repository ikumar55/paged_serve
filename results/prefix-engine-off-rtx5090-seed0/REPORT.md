# Engine sweep (continuous batching, paged KV cache)

3 runs, 3 valid, 2 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 32, 0 | 71 | 32, 0 | 71 |
| 8 | sustained runs | 32, 0 | 129 | 32, 0 | 129 |
| 16 | none sustained (least bad) | 32, 0 | 27265 | 32, 0 | 27265 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 3.92 | 577 | 41 | 71 | 9 | 1.4 | 3.6 | 5.7 | - | sustained |
| 32 | 0 | 8 | 480/480 | 7.89 | 1188 | 47 | 129 | 11 | 1.9 | 5.4 | 15.2 | - | sustained |
| 32 | 0 | 16 | 960/960 | 10.74 | 1593 | 15368 | 27265 | 15 | 18.2 | 30.8 | 29.9 | - | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
