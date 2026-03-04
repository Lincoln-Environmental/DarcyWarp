# Testing API Dashboard (MVP)

This is a local interactive runner for option 1:
- run tests from a button
- run quick benchmark variants from buttons
- view live command output
- cancel a running job

## Start

From repo root:

```bash
python testing_api_dashboard/server.py
```

Then open:

```text
http://127.0.0.1:8787
```

The local runner UI is now at:

```text
http://127.0.0.1:8787/dashboard
```

The root page shows the paper companion layout.

## Notes

- Commands are allowlisted in `testing_api_dashboard/server.py`.
- Everything runs from the repo root (`QUICK_FLOW`).
- You can add optional extra CLI args per command in the UI.
- This is intentionally local/dev-only and has no auth.
