# Engine sweep (continuous batching, paged KV cache)

3 runs, 3 valid, 2 sustained (valid and throughput >= 90% of the arrival rate).

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 4 | sustained runs | 32, 0 | 83 | 32, 0 | 83 |
| 8 | sustained runs | 32, 0 | 90 | 32, 0 | 90 |
| 16 | none sustained (least bad) | 32, 0 | 22221 | 32, 0 | 22221 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 4 | 240/240 | 4.09 | 579 | 41 | 83 | 10 | 1.4 | 3.8 | 6.3 | - | sustained |
| 32 | 0 | 8 | 480/480 | 7.60 | 1082 | 46 | 90 | 11 | 1.7 | 5.1 | 13.4 | - | sustained |
| 32 | 0 | 16 | 960/960 | 10.90 | 1545 | 11955 | 22221 | 15 | 14.5 | 26.2 | 28.9 | - | overloaded |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
