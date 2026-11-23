# vLLM sweep

3 runs, 3 valid, 3 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 24 | sustained runs | 32, 0 | 47 | 32, 0 | 47 |
| 32 | sustained runs | 32, 0 | 53 | 32, 0 | 53 |
| 48 | sustained runs | 32, 0 | 119 | 32, 0 | 119 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 24 | 1440/1440 | 23.97 | 3500 | 23 | 47 | 3 | 0.4 | 1.3 | - | - | sustained |
| 32 | 0 | 32 | 1920/1920 | 31.80 | 4618 | 24 | 53 | 3 | 0.4 | 1.2 | - | - | sustained |
| 32 | 0 | 48 | 2880/2880 | 47.55 | 6901 | 24 | 119 | 3 | 0.4 | 1.3 | - | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
