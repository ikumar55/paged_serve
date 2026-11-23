# vLLM sweep

6 runs, 6 valid, 6 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 32, 0 | 37 | 32, 0 | 37 |
| 8 | sustained runs | 32, 0 | 41 | 32, 0 | 41 |
| 16 | sustained runs | 32, 0 | 40 | 32, 0 | 40 |
| 24 | sustained runs | 32, 0 | 42 | 32, 0 | 42 |
| 32 | sustained runs | 32, 0 | 56 | 32, 0 | 56 |
| 48 | sustained runs | 32, 0 | 98 | 32, 0 | 98 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 3.94 | 536 | 25 | 37 | 2 | 0.3 | 1.0 | - | - | sustained |
| 32 | 0 | 8 | 480/480 | 8.07 | 1135 | 22 | 41 | 3 | 0.4 | 1.0 | - | - | sustained |
| 32 | 0 | 16 | 960/960 | 15.60 | 2219 | 22 | 40 | 3 | 0.4 | 1.1 | - | - | sustained |
| 32 | 0 | 24 | 1440/1440 | 23.87 | 3379 | 24 | 42 | 3 | 0.4 | 1.1 | - | - | sustained |
| 32 | 0 | 32 | 1920/1920 | 31.94 | 4559 | 23 | 56 | 3 | 0.4 | 1.2 | - | - | sustained |
| 32 | 0 | 48 | 2880/2880 | 47.54 | 6832 | 25 | 98 | 3 | 0.4 | 1.2 | - | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
