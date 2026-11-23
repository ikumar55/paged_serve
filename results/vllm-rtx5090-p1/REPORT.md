# vLLM sweep

8 runs, 8 valid, 8 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 2 | sustained runs | 32, 0 | 36 | 64, 0 | 36 |
| 4 | sustained runs | 64, 0 | 38 | 32, 0 | 43 |
| 8 | sustained runs | 64, 0 | 44 | 32, 0 | 46 |
| 16 | sustained runs | 32, 0 | 46 | 64, 0 | 48 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 2 | 120/120 | 1.91 | 287 | 25 | 36 | 2 | 0.4 | 1.0 | - | - | sustained |
| 64 | 0 | 2 | 120/120 | 1.91 | 287 | 24 | 36 | 2 | 0.4 | 1.0 | - | - | sustained |
| 32 | 0 | 4 | 240/240 | 4.00 | 588 | 25 | 43 | 3 | 0.4 | 1.0 | - | - | sustained |
| 64 | 0 | 4 | 240/240 | 4.00 | 588 | 23 | 38 | 3 | 0.4 | 1.0 | - | - | sustained |
| 32 | 0 | 8 | 480/480 | 8.25 | 1243 | 23 | 46 | 3 | 0.4 | 1.1 | - | - | sustained |
| 64 | 0 | 8 | 480/480 | 8.25 | 1243 | 22 | 44 | 3 | 0.4 | 1.1 | - | - | sustained |
| 32 | 0 | 16 | 960/960 | 16.24 | 2410 | 22 | 46 | 3 | 0.4 | 1.3 | - | - | sustained |
| 64 | 0 | 16 | 960/960 | 16.24 | 2408 | 23 | 48 | 3 | 0.4 | 1.3 | - | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
