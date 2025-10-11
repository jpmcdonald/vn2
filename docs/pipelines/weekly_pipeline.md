## Weekly pipeline

End-to-end batch to update imputation, forecasts, optimization, and build submission.

### Steps

1. Impute stockout weeks → update `demand_imputed.parquet`
2. Train/retrain champion + challengers; export metrics and artifacts
3. Optimize orders per SKU; export decisions
4. Build submission CSV

### Orchestration

- Shell entrypoint `run_weekly.sh` (to be added) calling each step with logging
- Schedule with launchd (macOS). Example plist:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.vn2.weekly</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>1</integer>
    <key>Hour</key><integer>2</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>WorkingDirectory</key><string>/Users/REPLACE/Code/vn2</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>source V2env/bin/activate && ./run_weekly.sh &>> logs/weekly.log</string>
  </array>
  <key>StandardOutPath</key><string>logs/weekly.out</string>
  <key>StandardErrorPath</key><string>logs/weekly.err</string>
</dict>
</plist>
```

Load with:

```bash
launchctl load ~/Library/LaunchAgents/com.vn2.weekly.plist
launchctl start com.vn2.weekly
```

### Artifacts

- Logs in `logs/` (rotated)
- Submission in `data/submissions/`
- Metrics reports in `reports/`


