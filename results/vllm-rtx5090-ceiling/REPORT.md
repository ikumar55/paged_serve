# vLLM sweep

6 runs, 2 valid, 2 sustained (valid and throughput >= 90% of the arrival rate).

Excluded from the best/worst comparison (still in the table): b32-w0-r96 (load generator saturated), b64-w0-r96 (load generator saturated), b32-w0-r128 (load generator saturated), b64-w0-r128 (load generator saturated)

## Best and worst configuration at each arrival rate

Ranked by p99 time to first token among the runs that kept up with the offered load. Where none did, every configuration is overloaded (its queue grows for the whole run), so the ranking is among valid runs and means only 'least bad'. The 90% cutoff for keeping up is a convention, so runs just either side of it are not meaningfully different.

| rate req/s | ranked among | best (batch, window ms) | TTFT p99 ms | worst (batch, window ms) | TTFT p99 ms |
| ---: | --- | --- | ---: | --- | ---: |
| 64 | sustained runs | 64, 0 | 74 | 32, 0 | 2989 |

## All runs

| batch | window ms | rate req/s | ok | req/s | tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | e2e p50 s | e2e p99 s | mean batch | wasted slots | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 64 | 3840/3840 | 60.55 | 8730 | 2157 | 2989 | 3 | 2.7 | 3.9 | - | - | sustained |
| 64 | 0 | 64 | 3840/3840 | 63.21 | 9115 | 28 | 74 | 3 | 0.5 | 1.5 | - | - | sustained |
| 32 | 0 | 96 | 4487/5760 | 58.82 | 8530 | 14282 | 17011 | 3 | 14.8 | 17.9 | - | - | load generator saturated |
| 64 | 0 | 96 | 5760/5760 | 82.57 | 11988 | 4808 | 9213 | 5 | 5.6 | 10.2 | - | - | load generator saturated |
| 32 | 0 | 128 | 4563/7680 | 59.91 | 8630 | 15614 | 16158 | 3 | 16.0 | 17.2 | - | - | load generator saturated |
| 64 | 0 | 128 | 5905/7680 | 82.63 | 11885 | 10684 | 11456 | 5 | 11.3 | 13.0 | - | - | load generator saturated |

Mean batch size and wasted slots come from the server's own counters and include the three unmeasured warm-up requests sent before each run (a fresh server per run).
