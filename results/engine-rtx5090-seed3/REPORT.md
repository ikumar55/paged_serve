# Engine sweep (continuous batching, paged KV cache)

8 runs, 8 valid, 7 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 64, 0 | 60 | 32, 0 | 60 |
| 8 | sustained runs | 32, 0 | 67 | 64, 0 | 68 |
| 12 | sustained runs | 64, 0 | 82 | 32, 0 | 509 |
| 16 | sustained runs | 64, 0 | 2433 | 64, 0 | 2433 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 3.98 | 570 | 32 | 60 | 9 | 1.3 | 4.7 | 5.6 | - | sustained |
| 64 | 0 | 4 | 240/240 | 3.98 | 570 | 31 | 60 | 9 | 1.3 | 3.6 | 5.3 | - | sustained |
| 32 | 0 | 8 | 480/480 | 7.59 | 1082 | 35 | 67 | 10 | 1.5 | 3.9 | 11.8 | - | sustained |
| 64 | 0 | 8 | 480/480 | 7.57 | 1080 | 35 | 68 | 10 | 1.5 | 3.9 | 11.7 | - | sustained |
| 32 | 0 | 12 | 720/720 | 11.36 | 1621 | 43 | 509 | 12 | 1.9 | 5.7 | 21.7 | - | sustained |
| 64 | 0 | 12 | 720/720 | 11.35 | 1621 | 41 | 82 | 12 | 1.8 | 5.7 | 21.2 | - | sustained |
| 32 | 0 | 16 | 960/960 | 13.08 | 1867 | 3635 | 10983 | 14 | 6.3 | 14.3 | 29.8 | - | overloaded |
| 64 | 0 | 16 | 960/960 | 14.51 | 2072 | 367 | 2433 | 25 | 4.2 | 11.0 | 51.0 | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
