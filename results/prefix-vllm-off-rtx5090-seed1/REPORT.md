# vLLM sweep

3 runs, 3 valid, 3 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 32, 0 | 58 | 32, 0 | 58 |
| 8 | sustained runs | 32, 0 | 68 | 32, 0 | 68 |
| 16 | sustained runs | 32, 0 | 67 | 32, 0 | 67 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 3.94 | 536 | 33 | 58 | 3 | 0.3 | 1.0 | - | - | sustained |
| 32 | 0 | 8 | 480/480 | 8.07 | 1134 | 34 | 68 | 3 | 0.4 | 1.1 | - | - | sustained |
| 32 | 0 | 16 | 960/960 | 15.59 | 2217 | 36 | 67 | 3 | 0.5 | 1.3 | - | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
