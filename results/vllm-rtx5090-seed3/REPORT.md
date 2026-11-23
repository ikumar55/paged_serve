# vLLM sweep

6 runs, 6 valid, 6 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 32, 0 | 36 | 32, 0 | 36 |
| 8 | sustained runs | 32, 0 | 42 | 32, 0 | 42 |
| 16 | sustained runs | 32, 0 | 44 | 32, 0 | 44 |
| 24 | sustained runs | 32, 0 | 45 | 32, 0 | 45 |
| 32 | sustained runs | 32, 0 | 54 | 32, 0 | 54 |
| 48 | sustained runs | 32, 0 | 218 | 32, 0 | 218 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 4.10 | 586 | 22 | 36 | 3 | 0.4 | 1.0 | - | - | sustained |
| 32 | 0 | 8 | 480/480 | 7.83 | 1116 | 23 | 42 | 3 | 0.4 | 1.0 | - | - | sustained |
| 32 | 0 | 16 | 960/960 | 15.86 | 2265 | 23 | 44 | 3 | 0.4 | 1.2 | - | - | sustained |
| 32 | 0 | 24 | 1440/1440 | 23.16 | 3309 | 23 | 45 | 3 | 0.4 | 1.2 | - | - | sustained |
| 32 | 0 | 32 | 1920/1920 | 31.23 | 4504 | 24 | 54 | 3 | 0.4 | 1.2 | - | - | sustained |
| 32 | 0 | 48 | 2880/2880 | 47.30 | 6843 | 25 | 218 | 3 | 0.4 | 1.4 | - | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
