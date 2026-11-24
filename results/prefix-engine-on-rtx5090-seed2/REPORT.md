# Engine sweep (continuous batching, paged KV cache)

3 runs, 2 valid, 2 sustained (valid and throughput >= 90% of the arrival rate).

Excluded from the best/worst comparison (still in the table): b32-w0-r16 (contended)

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 32, 0 | 65 | 32, 0 | 65 |
| 8 | sustained runs | 32, 0 | 88 | 32, 0 | 88 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 4.10 | 579 | 33 | 65 | 9 | 1.2 | 3.4 | 5.8 | - | sustained |
| 32 | 0 | 8 | 480/480 | 7.63 | 1087 | 38 | 88 | 11 | 1.6 | 4.7 | 12.9 | - | sustained |
| 32 | 0 | 16 | 960/960 | 12.16 | 1723 | 7078 | 13010 | 15 | 9.6 | 17.1 | 28.8 | - | contended |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
