# Artifacts

Persisted experiment and analysis outputs should live here.

Intended contents over time:
- run configs
- checkpoints
- extracted features
- local checkpoint manifests copied into canonical run directories
- local feature exports materialized into canonical train/test `FeatureCollection`
  files
- planned-run executor inputs copied into canonical run directories
- analysis outputs
- compact summaries used for later audit
- human-readable run reports
- benchmark result records and probe metrics

For the first runnable V-JEPA reference path, expect artifacts such as:

- `benchmark_result.json`
- `artifacts/run_summary.json`
- `analysis/run_report.md`

Historical artifacts should be additive and auditable rather than silently overwritten in place.
