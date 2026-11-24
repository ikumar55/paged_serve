# vLLM sweep

3 runs, 3 valid, 3 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 32, 0 | 39 | 32, 0 | 39 |
| 8 | sustained runs | 32, 0 | 43 | 32, 0 | 43 |
| 16 | sustained runs | 32, 0 | 39 | 32, 0 | 39 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 4.00 | 588 | 25 | 39 | 3 | 0.4 | 1.0 | - | - | sustained |
| 32 | 0 | 8 | 480/480 | 8.25 | 1243 | 24 | 43 | 3 | 0.4 | 1.1 | - | - | sustained |
| 32 | 0 | 16 | 960/960 | 16.23 | 2408 | 23 | 39 | 3 | 0.4 | 1.3 | - | - | sustained |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
