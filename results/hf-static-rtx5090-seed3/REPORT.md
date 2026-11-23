# Static batching sweep

6 runs, 6 valid, 4 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 64, 20 | 3950 | 32, 20 | 5952 |
| 8 | sustained runs | 32, 20 | 6106 | 64, 20 | 6410 |
| 16 | none sustained (least bad) | 64, 20 | 42259 | 32, 20 | 62027 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 20 | 4 | 240/240 | 3.83 | 547 | 1565 | 5952 | 9 | 3.0 | 7.6 | 9.0 | 52% | sustained |
| 64 | 20 | 4 | 240/240 | 3.83 | 548 | 1342 | 3950 | 9 | 2.6 | 6.7 | 8.7 | 53% | sustained |
| 32 | 20 | 8 | 480/480 | 7.43 | 1059 | 2915 | 6106 | 10 | 4.4 | 8.3 | 23.0 | 57% | sustained |
| 64 | 20 | 8 | 480/480 | 7.46 | 1064 | 2218 | 6410 | 10 | 3.6 | 8.5 | 22.0 | 57% | sustained |
| 32 | 20 | 16 | 960/960 | 7.75 | 1107 | 28961 | 62027 | 11 | 30.8 | 63.0 | 27.5 | 59% | overloaded |
| 64 | 20 | 16 | 960/960 | 9.28 | 1324 | 18728 | 42259 | 14 | 21.0 | 43.8 | 48.1 | 64% | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
