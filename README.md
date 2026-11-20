# LLMproxy — transparent OpenAI proxy + tok/sec dashboard

One file, Python stdlib only: `llmproxy.py`.

It sits in front of any OpenAI-compatible server (LM Studio, llama-server, ollama's
`/v1` endpoint, …) and does two things:

1. **Passes everything through byte-for-byte** — request bodies, headers, SSE streams,
   upstream 4xx/5xx. Clients see the same bytes they would see talking to the upstream
   directly, so any OpenAI-compatible client works unchanged.
2. **Measures generation speed** and shows it on a tiny web page.

## Run

```bash
python3 ~/projects/LLMproxy/llmproxy.py                                  # upstream :8080, proxy :8787, log .
python3 ~/projects/LLMproxy/llmproxy.py http://localhost:11434 8790      # ollama, other port
python3 ~/projects/LLMproxy/llmproxy.py http://127.0.0.1:8899 8787 ~/logs
```

| Arg | Default | Meaning |
|-----|---------|---------|
| 1 `upstream` | `http://localhost:8080` | base URL of the LLM server |
| 2 `port` | `8787` | port the proxy listens on |
| 3 `logdir` | `.` (current dir) | directory for the request/response log |

The proxy listens on all interfaces, IPv4 and IPv6.

Started with **no arguments**, it reuses the parameters of the most recent run (cached in
`~/.cache/llmproxy/last.json`, or under `$XDG_CACHE_HOME`) and says so on startup. Any run with
arguments, even just the first, replaces the cache; omitted trailing arguments fall back to the
defaults, not to the cached values. With no cache yet it uses the defaults. The log directory is
cached as an absolute path.

Then point the client at `http://localhost:<port>` (e.g. `base_url="http://localhost:8787/v1"`).

## Dashboard

- `http://localhost:8787/` — auto-refreshing page (1 s): big number is the live tok/s
  (or the session average when nothing is generating), then one row per request with
  path, model, tokens, tok/s and time-to-first-token.
- `http://localhost:8787/stats` — the same numbers as JSON, for scripts.

These two paths (and the `%tps` chat command) are the only things **not** proxied. Everything else, including `/v1/models`,
goes to the upstream.

## Speed command: `%tps`

For harnesses that show no speed numbers. Type `%tps` (or `;tps`, `::tps`, `/tps`, `/tokens`) as an ordinary chat
message. Harnesses usually eat a leading `/` or `!`, so the default is a prefix none of them
use. The proxy matches the trigger as a whole word anywhere in the final user message of a
`chat/completions` request (so wrapped or prefixed messages still work),
answers it itself (streaming or not, matching the request) and does **not** forward it to the
LLM or log it. The reply lists the live request, the last five (tok/s, tokens, ttft, model)
and the running average. No harness changes needed, it only has to let you send the text.

`%tps_all` (also `;tps_all`, `::tps_all`) reports totals since the proxy started instead:
average generation speed, peak request, request and token counts, generation time, and
wall-clock throughput over the whole uptime (idle time included). The `%tps` reply hints at it.
The totals are kept in memory and reset on restart; `%tps` itself averages only the last 50 requests.

Caveats: the harness keeps the `%tps` exchange in its conversation history, so later requests
carry it; and a harness that intercepts your trigger will never
send it, in which case try another prefix, or use the dashboard or `/stats`.

## Log

One JSON line per successful completion (`200` on a path containing `completion`), appended
to `<logdir>/llmproxy-<start time>-<upstream port>.jsonl`. The filename is fixed when the proxy
starts, so separate runs never share a file:

```bash
tail -f ~/logs/llmproxy-20261112-031135-8080.jsonl
```

```json
{"t": "2026-11-12 03:11:35",
 "prompt": {"model": "...", "stream": true, "messages": [{"role": "user", "content": "ping"}]},
 "out": "full response text, streaming deltas reassembled",
 "model": "qwen3.8-flash-next-coder-iq1_m", "tokens": 40, "tok_s": 28.5, "ttft_ms": 1314}
```

`prompt` is the parsed request body, so it contains the full incoming conversation.

## Notes / limits

- Token count is `usage.completion_tokens` when the server sends it; otherwise it falls back
  to counting SSE delta events (or chars/4, whichever is larger). Servers that batch deltas
  or use non-Latin scripts will be off.
- tok/s is measured **at the proxy**: first delta byte to last delta byte. Any buffering
  between the upstream and the proxy is folded into the number. Non-streaming tok/s includes
  prefill (the page says so).
- HTTP/1.1 only (no HTTP/2), no retry if the client hangs up mid-stream.
- Log writes are append-only, no rotation, and prompts/responses are stored in the clear.

## Selftest

```bash
python3 ~/projects/LLMproxy/llmproxy.py --selftest   # -> selftest ok
```

## Harness switch helpers

Point a harness at the proxy without changing which model it uses — same provider, same
model id, only the `baseUrl` moves.

| Script | What it does |
|--------|--------------|
| `pi-proxy.py [on\|off] [port]` | reads pi's `defaultProvider` from `~/.pi/agent/models.json`, starts the proxy in front of that upstream if nothing is listening, rewrites the provider's `baseUrl` to `http://localhost:<port>/v1`, and saves the original file as `models.json.orig`. `off` restores it. |

```bash
~/projects/LLMproxy/pi-proxy.py on      # pi now talks to the proxy; restart pi to pick it up
~/projects/LLMproxy/pi-proxy.py off     # pi back on its real upstream
```

Not covered: opencode (its `~/.config/opencode/opencode.jsonc` has no provider to rewrite —
adding one is a different edit), and claude / codex, whose clients speak the Anthropic or
Codex wire APIs that a byte-for-byte OpenAI proxy cannot serve. Ask if you want those wired
up; they need a translating proxy, not this one.
