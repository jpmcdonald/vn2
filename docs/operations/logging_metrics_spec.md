## Logging and metrics specification

### Goals

- Human-readable logs for debugging
- Machine-readable progress and metrics for monitoring and resume

### File layout

- `logs/<job>/<job>-YYYYMMDD.log` (rotated daily or 50 MB)
- `logs/<job>/<job>_heartbeat.csv` (append; compact per-minute status)

### Heartbeat schema (CSV)

| column          | type    | description                                       |
|-----------------|---------|---------------------------------------------------|
| timestamp       | string  | ISO8601 UTC                                       |
| job             | string  | e.g., impute/train/opt                            |
| phase           | string  | setup/scan/running/saving/done                    |
| items_total     | int     | total units of work                               |
| items_done      | int     | completed units                                   |
| items_failed    | int     | failed units                                      |
| eta_minutes     | float   | optional; -1 if unknown                           |
| rss_mb          | int     | resident memory of main process                   |
| cpu_pct         | int     | recent CPU percent (smoothed)                     |

### Log levels

- INFO for progress milestones; DEBUG for per-item details (optional); WARNING for recoverable issues; ERROR for fatal

### Resume invariants

- Logs and heartbeat are append-only
- Manifests enable idempotent restart (skip completed)


