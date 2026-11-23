# Static batching sweep

6 runs, 6 valid, 2 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 64, 20 | 3340 | 32, 20 | 3862 |
| 8 | none sustained (least bad) | 64, 20 | 5894 | 32, 20 | 6330 |
| 16 | none sustained (least bad) | 64, 20 | 42086 | 32, 20 | 60272 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 20 | 4 | 240/240 | 4.07 | 576 | 1322 | 3862 | 9 | 2.5 | 5.8 | 8.7 | 47% | sustained |
| 64 | 20 | 4 | 240/240 | 4.05 | 573 | 1324 | 3340 | 9 | 2.6 | 5.3 | 9.0 | 48% | sustained |
| 32 | 20 | 8 | 480/480 | 7.06 | 1006 | 3573 | 6330 | 10 | 5.1 | 8.6 | 23.0 | 58% | overloaded |
| 64 | 20 | 8 | 480/480 | 6.99 | 996 | 2345 | 5894 | 10 | 3.6 | 7.8 | 22.0 | 57% | overloaded |
| 32 | 20 | 16 | 960/960 | 7.70 | 1090 | 31689 | 60272 | 10 | 34.4 | 62.0 | 28.3 | 60% | overloaded |
| 64 | 20 | 16 | 960/960 | 8.98 | 1272 | 19552 | 42086 | 15 | 22.3 | 44.3 | 48.1 | 64% | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
