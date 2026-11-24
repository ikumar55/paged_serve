# Static batching sweep

4 runs, 4 valid, 2 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 8 | sustained runs | 64, 20 | 6415 | 32, 20 | 6612 |
| 16 | none sustained (least bad) | 64, 20 | 39013 | 32, 20 | 52015 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 20 | 8 | 480/480 | 7.45 | 1047 | 3400 | 6612 | 10 | 4.9 | 8.6 | 24.1 | 59% | sustained |
| 64 | 20 | 8 | 480/480 | 7.45 | 1047 | 2214 | 6415 | 9 | 3.6 | 8.5 | 22.0 | 59% | sustained |
| 32 | 20 | 16 | 960/960 | 8.39 | 1194 | 26460 | 52015 | 9 | 27.9 | 53.7 | 28.3 | 60% | overloaded |
| 64 | 20 | 16 | 960/960 | 9.62 | 1368 | 19414 | 39013 | 14 | 21.3 | 41.0 | 50.7 | 64% | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
