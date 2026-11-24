# Decisions

- **No external libraries on the dashboard.** The charts are hand-drawn on a canvas, so the page works offline and the proxy stays a single file.
- **The decode scale follows the moving average, not the raw rate.** Bursty raw samples (speculative decoding) are clipped at the top instead of stretching the axis.
- **The left scale goes logarithmic above 1000 tok/s (2026-11-21).** Occasional upward spikes in the average still squashed normal speeds into a flat line. Below 1000 the axis stays linear, which is easier to read. The prefill scales stay linear.
- **Prefill goes on its own right-hand scale.** Prefill is typically 10-100x the decode speed, so the two can't share an axis.
- **Request logs are not committed** (`*.jsonl` in `.gitignore`), because they hold full prompts in clear text.
- **Prefill tok/s counts only uncached prompt tokens, and skips cache hits (2026-11-24).** Cached tokens are not recomputed, so counting them would make the rate follow the cache, not the reading speed (a 16,720-token prompt with 4,947 cached: 624 tok/s real, 886 if all counted). Requests with fewer than 128 uncached tokens are left out of the prefill charts and shown as "cache hit": their time to first token is fixed overhead (1 new token in 117 ms showed as 9 tok/s). The effective rate (all prompt tokens / time to first token, what a user feels) is shown in the hover and the table and written to the log as `effective_tok_s`.
