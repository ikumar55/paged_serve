# Static batching sweep

12 runs, 10 valid, 6 sustained (valid and throughput >= 90% of the arrival rate).

Excluded from the best/worst comparison (still in the table): b16-w20-r8 (contended), b16-w20-r16 (82% succeeded)

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 2 | sustained runs | 16, 20 | 3007 | 32, 20 | 3359 |
| 4 | sustained runs | 64, 20 | 4168 | 32, 20 | 4388 |
| 8 | none sustained (least bad) | 64, 20 | 8932 | 32, 20 | 14318 |
| 16 | none sustained (least bad) | 64, 20 | 53011 | 32, 20 | 80136 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 20 | 2 | 120/120 | 1.86 | 279 | 926 | 3007 | 9 | 2.4 | 5.3 | 3.6 | 38% | sustained |
| 32 | 20 | 2 | 120/120 | 1.85 | 278 | 986 | 3359 | 9 | 2.4 | 5.0 | 3.4 | 38% | sustained |
| 64 | 20 | 2 | 120/120 | 1.85 | 277 | 1123 | 3178 | 9 | 2.4 | 6.0 | 3.5 | 38% | sustained |
| 16 | 20 | 4 | 240/240 | 3.84 | 565 | 1576 | 4169 | 9 | 3.0 | 6.1 | 9.7 | 50% | sustained |
| 32 | 20 | 4 | 240/240 | 3.83 | 563 | 1595 | 4388 | 9 | 2.8 | 6.2 | 9.3 | 48% | sustained |
| 64 | 20 | 4 | 240/240 | 3.76 | 552 | 1465 | 4168 | 9 | 2.8 | 6.2 | 9.0 | 48% | sustained |
| 16 | 20 | 8 | 480/480 | 4.60 | 693 | 20190 | 43656 | 10 | 21.5 | 44.6 | 13.8 | 53% | contended |
| 32 | 20 | 8 | 480/480 | 6.50 | 979 | 4656 | 14318 | 11 | 6.8 | 16.0 | 23.0 | 58% | overloaded |
| 64 | 20 | 8 | 480/480 | 6.91 | 1041 | 3335 | 8932 | 14 | 5.4 | 12.0 | 28.4 | 61% | overloaded |
| 16 | 20 | 16 | 784/960 | 4.39 | 644 | 62214 | 117763 | 10 | 63.4 | 119.1 | 14.9 | 54% | 82% succeeded |
| 32 | 20 | 16 | 960/960 | 6.80 | 1008 | 42043 | 80136 | 11 | 44.7 | 82.0 | 28.3 | 59% | overloaded |
| 64 | 20 | 16 | 960/960 | 8.44 | 1253 | 29295 | 53011 | 15 | 31.7 | 54.7 | 48.1 | 63% | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
