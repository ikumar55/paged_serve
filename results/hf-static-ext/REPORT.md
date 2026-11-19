# Static batching sweep

12 runs, 12 valid, 6 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 2 | sustained runs | 64, 100 | 3949 | 64, 20 | 4193 |
| 4 | sustained runs | 64, 20 | 6096 | 32, 20 | 7230 |
| 8 | none sustained (least bad) | 32, 20 | 37401 | 64, 20 | 43268 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 20 | 2 | 120/120 | 1.82 | 274 | 1286 | 3971 | 10 | 3.0 | 6.4 | 4.4 | 41% | sustained |
| 32 | 100 | 2 | 120/120 | 1.82 | 273 | 1380 | 4125 | 11 | 3.0 | 6.6 | 4.6 | 42% | sustained |
| 64 | 20 | 2 | 120/120 | 1.85 | 278 | 1189 | 4193 | 10 | 2.8 | 6.1 | 4.4 | 43% | sustained |
| 64 | 100 | 2 | 120/120 | 1.84 | 277 | 1417 | 3949 | 11 | 3.0 | 6.3 | 4.7 | 43% | sustained |
| 32 | 20 | 4 | 240/240 | 3.62 | 533 | 2106 | 7230 | 11 | 3.6 | 9.0 | 11.0 | 50% | sustained |
| 32 | 100 | 4 | 240/240 | 3.58 | 527 | 2108 | 6677 | 11 | 3.6 | 8.8 | 11.0 | 52% | overloaded |
| 64 | 20 | 4 | 240/240 | 3.61 | 531 | 1797 | 6096 | 11 | 3.4 | 8.2 | 10.6 | 49% | sustained |
| 64 | 100 | 4 | 240/240 | 3.57 | 526 | 2243 | 6690 | 11 | 3.9 | 8.8 | 11.6 | 52% | overloaded |
| 32 | 20 | 8 | 480/480 | 4.95 | 745 | 16550 | 37401 | 14 | 18.2 | 39.3 | 24.1 | 57% | overloaded |
| 32 | 100 | 8 | 480/480 | 4.84 | 729 | 16615 | 39513 | 13 | 18.2 | 41.4 | 24.1 | 59% | overloaded |
| 64 | 20 | 8 | 480/480 | 4.66 | 702 | 9256 | 43268 | 24 | 13.6 | 45.0 | 34.5 | 62% | overloaded |
| 64 | 100 | 8 | 480/480 | 4.59 | 691 | 11238 | 38151 | 24 | 15.4 | 44.9 | 34.5 | 62% | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
