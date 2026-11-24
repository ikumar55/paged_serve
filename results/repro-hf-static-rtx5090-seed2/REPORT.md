# Static batching sweep

4 runs, 4 valid, 0 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 8 | none sustained (least bad) | 64, 20 | 4759 | 32, 20 | 6278 |
| 16 | none sustained (least bad) | 64, 20 | 41617 | 32, 20 | 54540 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 20 | 8 | 480/480 | 7.07 | 1006 | 2128 | 6278 | 9 | 3.5 | 7.5 | 20.1 | 57% | overloaded |
| 64 | 20 | 8 | 480/480 | 7.10 | 1010 | 2014 | 4759 | 9 | 3.3 | 6.7 | 20.1 | 55% | overloaded |
| 32 | 20 | 16 | 960/960 | 8.01 | 1135 | 30245 | 54540 | 10 | 31.5 | 56.0 | 27.5 | 60% | overloaded |
| 64 | 20 | 16 | 960/960 | 9.11 | 1291 | 20482 | 41617 | 15 | 22.7 | 43.4 | 48.1 | 64% | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
