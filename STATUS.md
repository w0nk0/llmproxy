# Status

_Last updated 2026-11-21_

- `llmproxy.py`: a single-file logging proxy for an OpenAI-compatible LLM server, with token-speed
  stats (`/stats`), a live dashboard (`/`) and chat commands (`/tps`, `%tps_all`, ...).
- Dashboard (page v5): one band per request on the decode graph, the decode curve on the left scale
  and prefill as a bar on the right scale. The left scale turns logarithmic above 1000 tok/s.
  There is also a separate prefill chart, a live status chip and a per-request table.
- Started with no arguments, it reuses the last start parameters (`~/.cache/llmproxy/last.json`).
- Tests: `python3 llmproxy.py --selftest` (it also runs the dashboard script under node when node is installed).

- Prefill: cache hits (< 128 uncached tokens) are left out of the prefill charts and labelled in the table; the effective rate over all prompt tokens is shown next to it (2026-11-24). The running proxy picks this up at its next restart.

## Open
- `pi-proxy.py`, `tsp.js` and `x.txt` are unrelated leftovers from the original import.
