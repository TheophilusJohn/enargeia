# Benchmarks

Device: Apple M2 · shader-f16 yes · timestamp-query yes · maxStorageBufferBindingSize 2048 MiB (adapter max; default is 128 MiB)

## M1 — matmul, 1024³ fp32

| variant | GFLOP/s | ms | delta | notes |
|---|---|---|---|---|
| naive, no shared memory | 186.7 | 11.50 | — | baseline |
