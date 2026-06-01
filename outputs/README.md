## outputs/

Local **per-request JSON outputs** for the two-stage pipeline (classifier → detection).

- Intended path shape: `outputs/<request_id>.json`
- **Not committed**: this directory is gitignored (except this README and `.gitkeep`).

To generate a contract-only JSON in a CI-safe way:

```bash
python scripts/pipeline_request.py --dry-run --image tests/assets/sunflower.ppm --out outputs/{request_id}.json
```

