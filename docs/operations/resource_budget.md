## Resource budget (CPU/RAM profiles)

Target machine: Apple Silicon (12 cores), 64 GB RAM

### Profiles

| Mode      | --n-jobs | BLAS threads | Niceness | Background policy | Expected CPU | Notes |
|-----------|----------|--------------|----------|-------------------|--------------|-------|
| Daytime   | 4–6      | 1            | 5–10     | on                | 40–60%       | Leaves room for client work |
| Evening   | 8–10     | 1            | 0–5      | off               | 70–90%       | Moderate interactivity |
| Night     | -1       | 1            | 0        | off               | ~100%        | Full throughput |

### Memory

- Imputation workers: ~180–260 MB per process; main process ~300–400 MB
- Training/optimization varies by model; aim to keep total < 48 GB to avoid swap

### Practical limits

- Avoid starting more workers than physical cores
- Cap BLAS threads to 1 to prevent oversubscription
- Monitor with Activity Monitor or `top`/`vm_stat`


