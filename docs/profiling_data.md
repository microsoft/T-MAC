# Profiling Results


| Model              | Device         | NUM_THREADS | llama.cpp (CPU) (tokens/sec) | T-MAC (CPU) |
|--------------------|----------------|-------------|------------------------------|-------------|
| BitNet-3B          | M2-Ultra       | 1           | 6.49                         | 22.08       |
| BitNet-3B          | M2-Ultra       | 4           | 22.09                        | 54.46       |
| Llama-2-7B (W2)    | M2-Ultra       | 1           | 3.82                         | 16.68       |
| Llama-2-7B (W2)    | M2-Ultra       | 8           | 22.06                        | 51.01       |
| Llama-2-7B (W4)    | M2-Ultra       | 1           | 5.65                         | 8.97        |
| Llama-2-7B (W4)    | M2-Ultra       | 8           | 31.57                        | 35.65       |
|                    |                |             |                              |             |
| BitNet-3B          | AGX Orin       | 1           | 1.62                         | 8.18        |
| BitNet-3B          | AGX Orin       | 12          | 12.34                        | 26.02       |
| Llama-2-7B (W2)    | AGX Orin       | 1           | 0.79                         | 4.36        |
| Llama-2-7B (W2)    | AGX Orin       | 12          | 7.08                         | 15.62       |
| Llama-2-7B (W4)    | AGX Orin       | 1           | 1.04                         | 2.46        |
| Llama-2-7B (W4)    | AGX Orin       | 12          | 7.42                         | 8.09        |
|                    |                |             |                              |             |
| BitNet-3B          | Raspberry Pi 5 | 1           | 1.37                         | 8.03        |
| BitNet-3B          | Raspberry Pi 5 | 2           | 2.71                         | 11.09       |
| Llama-2-7B (W2)    | Raspberry Pi 5 | 1           | 0.66                         | 4.40        |
| Llama-2-7B (W2)    | Raspberry Pi 5 | 2           | 1.31                         | 5.92        |
| Llama-2-7B (W4)    | Raspberry Pi 5 | 1           | 0.85                         | 2.42        |
| Llama-2-7B (W4)    | Raspberry Pi 5 | 2           | 1.63                         | 3.35        |
|                    |                |             |                              |             |
| BitNet-3B          | Surface Book 3 | 1           | 5.65                         | 12.65       |
| BitNet-3B          | Surface Book 3 | 4           | 14.85                        | 28.60       |
| Llama-2-7B (W2)    | Surface Book 3 | 1           | 2.70                         | 6.77        |
| Llama-2-7B (W2)    | Surface Book 3 | 4           | 7.50                         | 16.82       |
| Llama-2-7B (W4)    | Surface Book 3 | 1           | 2.50                         | 3.74        |
| Llama-2-7B (W4)    | Surface Book 3 | 4           | 6.52                         | 9.34        |
