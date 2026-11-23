# Engine sweep (continuous batching, paged KV cache)

8 runs, 7 valid, 6 sustained (valid and throughput >= 90% of the arrival rate).

Excluded from the best/worst comparison (still in the table): b32-w0-r16 (contended)

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 64, 0 | 61 | 32, 0 | 62 |
| 8 | sustained runs | 32, 0 | 79 | 64, 0 | 86 |
| 12 | sustained runs | 64, 0 | 99 | 32, 0 | 678 |
| 16 | none sustained (least bad) | 64, 0 | 1641 | 64, 0 | 1641 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 4.08 | 576 | 32 | 62 | 9 | 1.2 | 3.5 | 5.8 | - | sustained |
| 64 | 0 | 4 | 240/240 | 4.11 | 581 | 30 | 61 | 9 | 1.2 | 3.4 | 5.6 | - | sustained |
| 32 | 0 | 8 | 480/480 | 7.61 | 1084 | 37 | 79 | 10 | 1.6 | 4.6 | 12.6 | - | sustained |
| 64 | 0 | 8 | 480/480 | 7.50 | 1069 | 37 | 86 | 10 | 1.6 | 4.6 | 12.3 | - | sustained |
| 32 | 0 | 12 | 720/720 | 11.49 | 1595 | 42 | 678 | 12 | 1.8 | 4.8 | 21.4 | - | sustained |
| 64 | 0 | 12 | 720/720 | 11.33 | 1573 | 42 | 99 | 12 | 1.8 | 5.3 | 22.1 | - | sustained |
| 32 | 0 | 16 | 960/960 | 12.62 | 1788 | 6096 | 10496 | 15 | 8.3 | 14.6 | 28.9 | - | contended |
| 64 | 0 | 16 | 960/960 | 14.23 | 2016 | 498 | 1641 | 25 | 4.0 | 10.8 | 46.6 | - | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
