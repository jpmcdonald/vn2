## Long-running jobs: operations guide

This guide describes how to reliably run, monitor, and share machine resources for long-running VN2 batch jobs on macOS (Apple Silicon, 12 cores, 64 GB RAM), while keeping interactive work responsive.

### Start jobs in persistent sessions (recommended)

Use tmux so jobs survive terminal/IDE crashes and you can detach/reattach.

```bash
# install if needed (Homebrew)
brew install tmux

# create and name a session
tmux new -s vn2-batch

# detach (keep running)
# press: Ctrl-b then d

# reattach later
tmux attach -t vn2-batch
```

For multiple concurrent jobs, create one session per job (e.g., `vn2-impute`, `vn2-train`, `vn2-opt`), each writing logs to a file (see Logging below).

### Avoid IDE terminals for production runs

- Do not start long jobs inside IDE/cursor-integrated terminals. Closing the IDE or the tab will end the process group.
- If you must, at least background and disown, but tmux is far more reliable.

### CPU scheduling and background policy

When you need the machine for interactive work, demote heavy jobs to background and reduce priority:

```bash
# lower priority (positive niceness = lower priority)
renice 10 -p <PID>

# apply macOS background policy (throttles CPU while interactive apps are active)
sudo /usr/bin/taskpolicy -b -p <PID>
```

To cap internal threading in math libs (avoid oversubscription when also using joblib `--n-jobs`):

```bash
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
```

Set these in the tmux session before starting the job. Use `--n-jobs` on our CLIs to control process-level parallelism.

### Resource profiles

- Daytime (shared): `--n-jobs 4..6`, niceness 5–10, background policy enabled, BLAS threads = 1
- Nighttime (exclusive): `--n-jobs -1` (all cores), default niceness, BLAS threads = 1

### Logging and progress heartbeat

- Write human-readable logs and a machine-readable heartbeat (CSV or JSONL) at a fixed interval (e.g., 60s) with fields:
  - timestamp, job_name, phase, items_total, items_done, items_failed, eta_minutes (if known), rss_mb, cpu_pct
- Rotate logs daily or at a size threshold (e.g., 50 MB).
- One line per completed unit of work (e.g., per stockout or per SKU) is acceptable when files are rotated.

Minimal pattern (example):

```text
2025-10-11T07:12:00Z,impute,scan,10517,0,0,NA,2650,980
2025-10-11T07:13:00Z,impute,imputing,10517,342,5,58,3120,980
```

### Safe shutdown and resume

- Prefer graceful stop via a sentinel file: job checks `stop.flag` and exits between work items.
- For robust resume, jobs must be idempotent and write a manifest of completed keys (see runbooks for each pipeline).

### Quick checklist

- Run in tmux; set BLAS thread caps; choose `--n-jobs` for the context
- Enable background policy and niceness when working interactively
- Log to file plus heartbeat; rotate logs
- Use checkpoint/resume manifests; test resume on a small slice
