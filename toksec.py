#!/usr/bin/env python3
"""toksec: transparent OpenAI-compatible LLM proxy + tok/sec dashboard.

    python3 toksec.py [upstream] [port] [logdir]  # default http://localhost:8080 8787 .

Point any OpenAI-compatible client at http://localhost:PORT. Every request is
forwarded byte-for-byte (streaming included); the only path not proxied is /
(the dashboard) and /stats (the same numbers as JSON).

Each completion is appended to logdir/toksec-<start time>-<llm port>.jsonl (default: current dir) as
one JSON line: prompt in, response out, token stats.
"""
import sys, json, time, socket, urllib.request, urllib.error, urllib.parse, itertools, collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8787
LOGDIR = sys.argv[3] if len(sys.argv) > 3 else "."
P = urllib.parse.urlparse(UPSTREAM)
LOGFILE = (LOGDIR.rstrip("/") + f"/toksec-{time.strftime('%Y%m%d-%H%M%S')}-"
           f"{P.port or (443 if P.scheme == 'https' else 80)}.jsonl")
DROP = {"content-length", "transfer-encoding", "connection", "keep-alive", "host"}

LOG = collections.deque(maxlen=50)  # finished requests
LIVE = {}                           # in-flight requests, keyed by counter
KEY = itertools.count()


def dump(prompt, entry, out):
    try:
        prompt = json.loads(prompt)
    except (ValueError, TypeError):
        prompt = str(prompt)
    try:
        with open(LOGFILE, "a") as f:
            f.write(json.dumps({"t": time.strftime("%F %T"), "prompt": prompt, "out": out,
                                "model": entry["model"], "tokens": entry["tokens"],
                                "tok_s": entry["tok_s"], "ttft_ms": entry["ttft_ms"]},
                               ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"proxy: cannot write {LOGFILE}: {e}")

# shortcut: 1 SSE delta event = 1 token, with max(events, chars/4) as fallback
# for servers that batch deltas. upgrade: trust usage.completion_tokens only.
def sse(line):
    """-> (delta_events, delta_text, usage_completion_tokens) for one SSE line."""
    if not line.startswith(b"data:"):
        return 0, "", None
    try:
        d = json.loads(line[5:].strip())
    except ValueError:
        return 0, "", None
    if d.get("usage"):
        return 0, "", d["usage"].get("completion_tokens")
    txt = ""
    for c in d.get("choices", []):
        delta = c.get("delta") or c
        txt += delta.get("content") or delta.get("reasoning_content") or delta.get("text") or ""
    return (1 if txt else 0), txt, None


def est(text):
    return max(1, len(text) // 4)


def rate(e):
    return e["tokens"] / (e["last"] - e["first"]) if e["first"] and e["last"] > e["first"] else 0.0


def snapshot():
    done = list(LOG)
    live = [e for e in LIVE.values() if e["first"]]
    tok = sum(r["tokens"] for r in done)
    sec = sum(r["secs"] for r in done)
    return {
        "requests": len(done),
        "tokens": tok,
        "avg_tok_s": round(tok / sec, 1) if sec else 0.0,
        "live": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in e.items()
                  if k != "t0"} | {"ttft_ms": round((e["first"] - e["t0"]) * 1000),
                                   "tok_s": round(rate(e), 1)}
                 for e in live],
        "last": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}
                 for r in reversed(done)],
    }


def page():
    s = snapshot()
    head = f"<meta http-equiv='refresh' content='1'><style>body{{background:#111;color:#0f0;font:14px/1.6 monospace}}.big{{font:42px monospace}}.l{{color:#ff0}}</style>"
    big = s["live"][0]["tok_s"] if s["live"] else s["avg_tok_s"]
    rows = lambda xs, cls: "".join(
        f"<tr class='{cls}'><td>{x['path']}</td><td>{x['model']}</td><td>{x['tokens']} tok</td>"
        f"<td>{x['tok_s']} tok/s</td><td>{x['ttft_ms']} ms ttft</td></tr>" for x in xs)
    return (f"<html>{head}<body><h1>tok/sec</h1>"
            f"<div class='big'>{big}</div>"
            f"<h2>live</h2>{rows(s['live'], 'l')}"
            f"<h2>last {len(s['last'])} of {s['requests']} requests, {s['tokens']} tok</h2>"
            f"{rows(s['last'], 'd')}"
            f"<p>avg {s['avg_tok_s']} tok/s (non-streaming tok/s includes prefill)</p></body></html>")


class DualStackServer(ThreadingHTTPServer):
    """Listen on every interface, IPv4 and IPv6 (one dual-stack socket on ::)."""
    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


class H(BaseHTTPRequestHandler):
    def chunk(self, data):
        self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def route(self):
        if self.path.split("?")[0] in ("/", "/stats"):
            body = (json.dumps(snapshot()) if self.path == "/stats" else page()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/" + ("json" if self.path == "/stats" else "html"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)

        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in DROP}
        req = urllib.request.Request(UPSTREAM + self.path, body or None, headers, method=self.command)
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            resp = e                      # pass upstream 4xx/5xx through untouched
        except Exception as e:
            msg = f"proxy 502: upstream {UPSTREAM} unreachable: {e}".encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            return self.wfile.write(msg)

        code = getattr(resp, "status", None) or resp.code
        ctype = resp.headers.get("content-type", "")
        self.send_response(code)
        for k, v in resp.headers.items():
            if k.lower() not in DROP:
                self.send_header(k, v)

        model = ""
        try:
            model = json.loads(body).get("model", "")
        except Exception:
            pass

        if "event-stream" in ctype:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            key = next(KEY)
            e = LIVE[key] = {"path": self.path, "model": model, "tokens": 0,
                             "t0": t0, "first": None, "last": None, "out": ""}
            events = chars = usage = 0
            while True:
                line = resp.readline()
                if not line:
                    break
                n, txt, u = sse(line)
                events, chars = events + n, chars + len(txt)
                usage = max(usage, u or 0)          # cumulative in llama.cpp
                e["out"] += txt
                if n:
                    now = time.time()
                    e["first"] = e["first"] or now
                    e["last"] = now
                e["tokens"] = usage or max(events, chars // 4)
                self.chunk(line)
            self.wfile.write(b"0\r\n\r\n")
            e["secs"] = e["last"] - e["first"] if e["first"] else time.time() - t0
            e["tok_s"] = round(rate(e), 1)
            e["ttft_ms"] = round((e["first"] - t0) * 1000) if e["first"] else 0
            LOG.append({k: v for k, v in e.items() if k not in ("t0", "out")})
            dump(body, e, e["out"])
            del LIVE[key]
        else:
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            try:
                j = json.loads(data)
                out = ""
                for c in j.get("choices", []):
                    m = c.get("message") or c
                    out += m.get("content") or m.get("reasoning_content") or m.get("text") or ""
                tok = (j.get("usage") or {}).get("completion_tokens") or est(out)
            except Exception:
                out, tok = "", 0
            secs = time.time() - t0
            if code == 200 and "/completion" in self.path:
                entry = {"path": self.path, "model": model, "tokens": tok, "secs": secs,
                         "tok_s": round(tok / secs, 1) if secs else 0.0, "ttft_ms": 0}
                LOG.append(entry)
                dump(body, entry, out)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = route


def selftest():
    ev = ch = us = 0
    for line in (b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
                 b'data: {"choices":[{"delta":{"reasoning_content":" think"}}]}\n',
                 b'data: [DONE]\n',
                 b'data: {"usage":{"completion_tokens":2}}\n',
                 b'data: {"usage":{"completion_tokens":1}}\n'):
        n, txt, u = sse(line)
        ev, ch, us = ev + n, ch + len(txt), max(us, u or 0)
    assert (ev, ch, us) == (2, 11, 2), (ev, ch, us)
    n, txt, u = sse(b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n')
    assert (n, txt, u) == (1, "Hello", None)
    print("selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        raise SystemExit(selftest())
    print(f"proxy on :{PORT} -> {UPSTREAM}, dashboard http://localhost:{PORT}/, "
          f"log {LOGFILE}")
    try:
        srv = DualStackServer(("::", PORT), H)
    except OSError:                       # no IPv6 on this host: plain IPv4 on all interfaces
        srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    srv.serve_forever()
