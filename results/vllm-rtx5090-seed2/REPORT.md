# vLLM sweep

6 runs, 6 valid, 6 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 32, 0 | 38 | 32, 0 | 38 |
| 8 | sustained runs | 32, 0 | 45 | 32, 0 | 45 |
| 16 | sustained runs | 32, 0 | 44 | 32, 0 | 44 |
| 24 | sustained runs | 32, 0 | 51 | 32, 0 | 51 |
| 32 | sustained runs | 32, 0 | 52 | 32, 0 | 52 |
| 48 | sustained runs | 32, 0 | 420 | 32, 0 | 420 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 4.20 | 593 | 23 | 38 | 3 | 0.3 | 0.9 | - | - | sustained |
| 32 | 0 | 8 | 480/480 | 7.90 | 1126 | 22 | 45 | 3 | 0.4 | 0.9 | - | - | sustained |
| 32 | 0 | 16 | 960/960 | 15.50 | 2196 | 23 | 44 | 3 | 0.4 | 1.1 | - | - | sustained |
| 32 | 0 | 24 | 1440/1440 | 23.62 | 3391 | 24 | 51 | 3 | 0.4 | 1.2 | - | - | sustained |
| 32 | 0 | 32 | 1920/1920 | 31.92 | 4581 | 23 | 52 | 3 | 0.4 | 1.2 | - | - | sustained |
| 32 | 0 | 48 | 2880/2880 | 48.30 | 6943 | 27 | 420 | 3 | 0.5 | 1.5 | - | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
