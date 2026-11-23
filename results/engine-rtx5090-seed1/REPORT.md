# Engine sweep (continuous batching, paged KV cache)

8 runs, 6 valid, 5 sustained (valid and throughput >= 90% of the arrival rate).

Excluded from the best/worst comparison (still in the table): b32-w0-r8 (contended), b64-w0-r8 (contended)

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 64, 0 | 66 | 32, 0 | 84 |
| 12 | sustained runs | 64, 0 | 106 | 32, 0 | 616 |
| 16 | sustained runs | 64, 0 | 1982 | 64, 0 | 1982 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 3.77 | 513 | 34 | 84 | 10 | 1.3 | 3.9 | 5.3 | - | sustained |
| 64 | 0 | 4 | 240/240 | 3.78 | 514 | 33 | 66 | 9 | 1.2 | 4.5 | 5.1 | - | sustained |
| 32 | 0 | 8 | 480/480 | 7.90 | 1110 | 38 | 79 | 11 | 1.5 | 4.6 | 13.0 | - | contended |
| 64 | 0 | 8 | 480/480 | 7.88 | 1108 | 38 | 70 | 10 | 1.5 | 5.2 | 13.2 | - | contended |
| 32 | 0 | 12 | 720/720 | 11.77 | 1673 | 49 | 616 | 13 | 1.9 | 5.8 | 23.9 | - | sustained |
| 64 | 0 | 12 | 720/720 | 11.75 | 1671 | 43 | 106 | 12 | 1.8 | 5.9 | 23.0 | - | sustained |
| 32 | 0 | 16 | 960/960 | 13.62 | 1937 | 3996 | 7999 | 14 | 6.6 | 12.2 | 30.3 | - | overloaded |
| 64 | 0 | 16 | 960/960 | 15.13 | 2153 | 71 | 1982 | 22 | 3.3 | 10.4 | 46.9 | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
