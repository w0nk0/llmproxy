#!/usr/bin/env python3
"""llmproxy: transparent OpenAI-compatible LLM proxy + tok/sec dashboard.

    python3 llmproxy.py [upstream] [port] [logdir]  # default http://localhost:8080 8787 .
    python3 llmproxy.py                             # no arguments: reuse the last run's parameters

Point any OpenAI-compatible client at http://localhost:PORT. Every request is
forwarded byte-for-byte (streaming included); the only path not proxied is /
(the dashboard) and /stats (the same numbers as JSON).

Speed command: send `%tps` (also `;tps`, `::tps`, `/tps`) as a chat message from any client. The proxy
answers it itself with the current tok/s stats as an assistant reply; nothing reaches the LLM.

Each completion is appended to logdir/llmproxy-<start time>-<llm port>.jsonl (default: current dir) as
one JSON line: prompt in, response out, token stats.
"""
import os, sys, json, time, socket, urllib.request, urllib.error, urllib.parse, itertools, collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULTS = ["http://localhost:8080", "8787", "."]
CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                     "llmproxy", "last.json")


def load_cached():
    """Start parameters of the most recent run, or None."""
    try:
        with open(CACHE) as f:
            c = json.load(f)
        return [str(c["upstream"]), str(c["port"]), str(c["logdir"])]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_cached(upstream, port, logdir):
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as f:
            json.dump({"upstream": upstream, "port": port, "logdir": os.path.abspath(logdir)}, f)
    except OSError as e:
        print(f"proxy: cannot cache start parameters in {CACHE}: {e}")


ARGS = sys.argv[1:]
SELFTEST = ARGS[:1] == ["--selftest"]
FROM_CACHE = not ARGS and load_cached()      # started without parameters: reuse the last ones
# missing trailing arguments fall back to the defaults
UPSTREAM, PORT, LOGDIR = (FROM_CACHE or ARGS + DEFAULTS[len(ARGS):])[:3]
PORT = int(PORT)
P = urllib.parse.urlparse(UPSTREAM)
LOGFILE = (LOGDIR.rstrip("/") + f"/llmproxy-{time.strftime('%Y%m%d-%H%M%S')}-"
           f"{P.port or (443 if P.scheme == 'https' else 80)}.jsonl")
DROP = {"content-length", "transfer-encoding", "connection", "keep-alive", "host"}

LOG = collections.deque(maxlen=50)  # finished requests
TOTAL = {"requests": 0, "tokens": 0, "secs": 0.0, "peak": 0.0, "since": time.time()}  # since start
LIVE = {}                           # in-flight requests that are streaming, keyed by counter
PENDING = {}                        # requests sent upstream, no response yet (prefill phase)
KEY = itertools.count()
KEEP_EV = 8                         # finished requests that keep their per-token timeline (for the decode graph)
MIN_WINDOW = 10                     # the decode graph averages over at least this many tokens
MA_PERIOD = 20                      # samples in the moving average drawn over the decode curve


def record(entry):
    """Add a finished request to the recent list and the since-start totals."""
    LOG.append(entry)
    TOTAL["requests"] += 1
    TOTAL["tokens"] += entry["tokens"]
    TOTAL["secs"] += entry["secs"]
    TOTAL["peak"] = max(TOTAL["peak"], entry["tok_s"])
    for old in list(LOG)[:-KEEP_EV]:    # only the newest few keep their timeline
        old.pop("ev", None)


def dump(prompt, entry, out):
    try:
        prompt = json.loads(prompt)
    except (ValueError, TypeError):
        prompt = str(prompt)
    try:
        with open(LOGFILE, "a") as f:
            f.write(json.dumps({"t": time.strftime("%F %T"), "prompt": prompt, "out": out,
                                "model": entry["model"], "tokens": entry["tokens"],
                                "tok_s": entry["tok_s"], "ttft_ms": entry["ttft_ms"],
                                "prompt_tokens": entry.get("prompt_tokens"),
                                "cached_tokens": entry.get("cached"),
                                "prefill_tok_s": entry.get("prefill_tok_s"),
                                "effective_tok_s": entry.get("effective_tok_s")},
                               ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"proxy: cannot write {LOGFILE}: {e}")

# shortcut: 1 SSE delta event = 1 token, with max(events, chars/4) as fallback
# for servers that batch deltas. upgrade: trust usage.completion_tokens only.
def sse(line):
    """-> (delta_events, delta_text, usage dict or None) for one SSE line."""
    if not line.startswith(b"data:"):
        return 0, "", None
    try:
        d = json.loads(line[5:].strip())
    except ValueError:
        return 0, "", None
    if isinstance(d, dict) and d.get("usage"):
        return 0, "", d["usage"]
    txt = ""
    for c in d.get("choices", []) if isinstance(d, dict) else []:
        delta = c.get("delta") or c
        txt += delta.get("content") or delta.get("reasoning_content") or delta.get("text") or ""
    return (1 if txt else 0), txt, None


def usage_numbers(u):
    """-> (completion_tokens, prompt_tokens, cached_prompt_tokens), each 0/None when the server did not say."""
    u = u or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return u.get("completion_tokens") or 0, u.get("prompt_tokens"), cached


def est(text):
    return max(1, len(text) // 4)


def prompt_estimate(body):
    """Rough prompt size (chars/4) from a request body, for servers that report no prompt_tokens."""
    try:
        j = json.loads(body)
        chars = len(j["prompt"]) if isinstance(j.get("prompt"), str) else 0
        for m in j.get("messages") or []:
            c = m.get("content")
            chars += (sum(len(x.get("text", "")) for x in c if isinstance(x, dict)) if isinstance(c, list)
                      else len(c) if isinstance(c, str) else 0)
        if j.get("tools"):
            chars += len(json.dumps(j["tools"]))
        return max(1, chars // 4) if chars else 0
    except (ValueError, TypeError, AttributeError):
        return 0


def event_tokens(txt):
    """Tokens in one SSE delta: one, unless the server batches (long text per event)."""
    return 1 if len(txt) <= 12 else max(1, len(txt) // 4)


def rate(e):
    return e["tokens"] / (e["last"] - e["first"]) if e["first"] and e["last"] > e["first"] else 0.0


MIN_PREFILL_NEW = 128   # below this many uncached tokens the time to first token is fixed overhead, not reading speed


def prefill_stats(prompt, cached, ttft_ms):
    """-> (new prompt tokens, prefill tok/s or None, effective tok/s or None).

    prefill tok/s = uncached prompt tokens / time to first token: how fast the server reads. None when fewer than
    MIN_PREFILL_NEW tokens were new (a cache hit: the rate would only measure overhead, e.g. 1 token / 117 ms = 9 tok/s).
    effective tok/s = all prompt tokens / time to first token: the speed a user feels, cache included."""
    new = max(0, (prompt or 0) - (cached or 0))
    sec = ttft_ms / 1000 if ttft_ms and ttft_ms > 0 else 0
    pre = round(new / sec) if sec and new >= MIN_PREFILL_NEW else None
    eff = round(prompt / sec) if sec and prompt else None
    return new, pre, eff


def curve(ev, scale=1.0, win=MIN_WINDOW, maxpts=200):
    """Decode speed along one request: [(output token index, tok/s over the last >= win tokens, moving average)].

    The moving average is the mean of the last MA_PERIOD of those samples (fewer at the start of a request).

    ev is [(seconds since the first token, tokens so far), ...]. A window always spans at least `win`
    tokens, so one slow or one bursty token (speculative drafts arrive together) does not spike the line.
    """
    win = max(MIN_WINDOW, int(win))
    pts, j, recent, total = [], 0, collections.deque(), 0.0
    for i in range(1, len(ev)):
        t, c = ev[i][0], ev[i][1] * scale
        while j + 1 < i and c - ev[j + 1][1] * scale >= win:
            j += 1
        tj, cj = ev[j][0], ev[j][1] * scale
        if c - cj >= win and t > tj:
            v = (c - cj) / (t - tj)
            recent.append(v); total += v
            if len(recent) > MA_PERIOD:
                total -= recent.popleft()
            pts.append((round(c), round(v, 1), round(total / len(recent), 1)))   # (token, rate, moving average)
    if len(pts) > maxpts:
        step = -(-len(pts) // maxpts)
        pts = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
    return pts


def clean(e):
    """A request entry for JSON: rounded, without the raw timeline and the response text."""
    return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in e.items()
            if k not in ("t0", "ev", "out", "scaled", "prompt_guess")}


def snapshot(window=MIN_WINDOW):
    window = max(MIN_WINDOW, int(window))
    done = list(LOG)
    live = [e for e in list(LIVE.values()) if e["first"]]
    now = time.time()
    pending = [{"id": e["id"], "path": e["path"], "model": e["model"], "prompt_guess": e["prompt_guess"],
                "elapsed_s": round(now - e["t0"], 1)}
               for e in list(PENDING.values()) + [e for e in list(LIVE.values()) if not e["first"]]]
    tok = sum(r["tokens"] for r in done)
    sec = sum(r["secs"] for r in done)
    decode = []
    for e, is_live in [(r, False) for r in done[-KEEP_EV:]] + [(e, True) for e in live]:
        if e.get("ev"):
            scale = (e["tokens"] / e["ev"][-1][1]) if (not is_live and e.get("scaled") and e["ev"][-1][1]) else 1.0
            decode.append({"id": e.get("id"), "live": is_live, "start": round(e.get("start", 0), 1),
                           "model": e["model"], "tokens": e["tokens"],
                           "points": curve(e["ev"], scale, window)})
    roll = [p[1] for d in decode for p in d["points"]]
    return {
        "requests": len(done),
        "tokens": tok,
        "avg_tok_s": round(tok / sec, 1) if sec else 0.0,
        "window": window,
        "uptime_s": round(now - TOTAL["since"]),
        "total_requests": TOTAL["requests"],
        "total_tokens": TOTAL["tokens"],
        "peak_request_tok_s": TOTAL["peak"],
        "peak_rolling_tok_s": max(roll) if roll else 0.0,
        "pending": pending,
        "live": [clean(e) | {"ttft_ms": round((e["first"] - e["t0"]) * 1000), "tok_s": round(rate(e), 1)}
                 for e in live],
        "last": [clean(r) for r in reversed(done)],
        "decode": decode,
        "prefill": [{k: r.get(k) for k in ("id", "start", "prompt_tokens", "cached", "new_tokens", "ttft_ms",
                                           "prefill_tok_s", "effective_tok_s", "prompt_est", "model")}
                    for r in done if r.get("prefill_tok_s") is not None],
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llmproxy</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--line:#262d38;--text:#e6edf3;--dim:#8b96a5;--accent:#3fb950;--live:#f2b84b;
--prefill:#58a6ff;--bad:#f85149;--grid:#222a35}
@media (prefers-color-scheme:light){:root{--bg:#f4f6f8;--panel:#fff;--line:#d8dee6;--text:#1b2430;--dim:#5d6a7a;
--accent:#1a7f37;--live:#b7791f;--prefill:#0969da;--bad:#cf222e;--grid:#e6ebf0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 40px}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:16px}
h1{font-size:18px;margin:0;font-weight:650;letter-spacing:.2px}
.sub{color:var(--dim);font-size:12px}
.chip{display:inline-flex;align-items:center;gap:7px;padding:3px 11px;border-radius:999px;border:1px solid var(--line);
background:var(--panel);font-size:12px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--dim)}
.chip.gen .dot{background:var(--live);animation:p 1s infinite}.chip.pre .dot{background:var(--prefill);animation:p .7s infinite}
.chip.err .dot{background:var(--bad)}
@keyframes p{50%{opacity:.25}}
.spacer{flex:1}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{background:var(--panel);color:var(--dim);border:0;padding:4px 11px;font:inherit;font-size:12px;cursor:pointer}
.seg button+button{border-left:1px solid var(--line)}
.seg button.on{background:var(--accent);color:#fff}
.hero{display:grid;grid-template-columns:minmax(220px,1.1fr) 3fr;gap:12px;margin-bottom:12px}
@media (max-width:760px){.hero{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.big{font:600 56px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.big.live{color:var(--live)}
.bigl{color:var(--dim);font-size:12px;margin-top:6px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.card{border-left:3px solid var(--line);padding:2px 0 2px 10px}
.card b{display:block;font:600 20px/1.2 ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.card span{color:var(--dim);font-size:11.5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
@media (max-width:900px){.grid2{grid-template-columns:1fr}}
.panel h2{margin:0 0 2px;font-size:13px;font-weight:650}
.panel .hint{color:var(--dim);font-size:11.5px;margin-bottom:8px}
.chartwrap{position:relative;height:230px}
canvas{width:100%;height:100%;display:block}
.tip{position:absolute;pointer-events:none;background:var(--bg);border:1px solid var(--line);border-radius:8px;
padding:5px 9px;font-size:12px;white-space:nowrap;display:none;z-index:2;box-shadow:0 4px 14px #0006}
.empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--dim);font-size:12.5px}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--dim);margin-top:6px}
.legend i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{color:var(--dim);font-weight:500;text-align:right;padding:5px 8px;border-bottom:1px solid var(--line)}
td{padding:5px 8px;text-align:right;font-variant-numeric:tabular-nums;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
td.m{max-width:230px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
tr.l td{color:var(--live)}
.tw{overflow-x:auto}
footer{color:var(--dim);font-size:11.5px;margin-top:10px}
</style></head><body><main>
<header>
 <h1>llmproxy</h1><span class="sub" id="target"></span>
 <span class="chip" id="status"><span class="dot"></span><span id="stext">connecting</span></span>
 <span class="spacer"></span>
 <span class="sub">decode window</span>
 <span class="seg" id="win"><button data-w="10">10 tok</button><button data-w="25">25 tok</button><button data-w="50">50 tok</button></span>
</header>

<section class="hero">
 <div class="panel"><div class="big" id="big">-</div><div class="bigl" id="bigl">tok/s</div></div>
 <div class="panel"><div class="cards" id="cards"></div></div>
</section>

<section class="grid2">
 <div class="panel">
  <h2>Inference (decode) speed</h2>
  <div class="hint" id="dhint">tok/s over a sliding window of at least 10 tokens, against the output token position. Newest request is brightest.</div>
  <div class="chartwrap"><canvas id="dc"></canvas><div class="empty" id="de">no request measured yet</div><div class="tip" id="dt"></div></div>
  <div class="legend" id="dl"></div>
 </div>
 <div class="panel">
  <h2>Prefill speed</h2>
  <div class="hint">uncached prompt tokens / time to first token, one bar per request. Requests with fewer than 128 uncached tokens (cache hits) are left out: their time to first token is overhead, not reading speed. Hover shows the effective rate over all prompt tokens.</div>
  <div class="chartwrap"><canvas id="pc"></canvas><div class="empty" id="pe">no prefill measured yet</div><div class="tip" id="pt"></div></div>
  <div class="legend"><span><i style="background:var(--prefill)"></i>measured prompt size</span>
   <span><i style="background:var(--prefill);opacity:.4"></i>prompt size estimated (server sent no usage)</span></div>
 </div>
</section>

<section class="panel">
 <h2>Recent requests</h2>
 <div class="tw"><table><thead><tr><th>time</th><th>model</th><th>prompt</th><th>cached</th><th>prefill tok/s</th>
 <th>ttft</th><th>out tok</th><th>decode tok/s</th></tr></thead><tbody id="rows"></tbody></table></div>
</section>
<footer><span id="ver"></span> Token counts come from the server's usage when it sends it. Speed is measured at the proxy.
Non-streaming requests have no prefill/decode split. <a style="color:var(--dim)" href="/stats">/stats</a> has the raw numbers.</footer>
</main>
<script>
"use strict";
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const MA=20;
let win=10;try{win=Math.max(10,+localStorage.getItem("llmproxy.win")||10)}catch(e){}
let S=null;
const fmt=(n,d=1)=>n==null?"-":(+n).toLocaleString(undefined,{maximumFractionDigits:d});
const hms=t=>new Date(t*1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
const dur=s=>s<90?s+" s":s<5400?Math.round(s/60)+" min":(s/3600).toFixed(1)+" h";
const median=a=>{if(!a.length)return null;a=[...a].sort((x,y)=>x-y);return a[a.length>>1]};
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

for(const b of document.querySelectorAll("#win button")){
  b.onclick=()=>{win=+b.dataset.w;try{localStorage.setItem("llmproxy.win",win)}catch(e){};mark();poll()};
}
function mark(){for(const b of document.querySelectorAll("#win button"))b.classList.toggle("on",+b.dataset.w===win);
  $("dhint").textContent="one band per request, oldest left, newest right. Decode: tok/s over a sliding window of at least "+win+" tokens against the output token position, with a "+MA+"-sample moving average in front (left scale, logarithmic above "+LOG_ABOVE+" tok/s). Blue bar: prefill tok/s (right scale)."}

function nice(max){if(max<=0)return 10;const p=10**Math.floor(Math.log10(max)),f=max/p;
  return (f<=1?1:f<=2?2:f<=2.5?2.5:f<=5?5:10)*p}

// left y scale: linear, but logarithmic (1 .. next power of 10) once the data exceeds LOG_ABOVE tok/s
const LOG_ABOVE=1000;
function yscale(max,T,ph,canLog){
  if(canLog&&max>LOG_ABOVE){const top=10**Math.ceil(Math.log10(max*1.08)),dec=Math.log10(top),ticks=[];
    for(let k=0;k<=dec;k++)ticks.push(10**k);
    return{log:true,max:top,ticks,Y:y=>T+ph-Math.log10(Math.max(y,1))/dec*ph}}
  const ymax=nice(max*1.08);
  return{log:false,max:ymax,ticks:[0,1,2,3,4].map(i=>ymax*i/4),Y:y=>T+ph-(y/ymax)*ph}}

// chart geometry: a second scale (m.ymax2) needs room for tick labels on the right too
function pad(m){return{L:46,R:m.ymax2?46:10,T:8,B:24}}

// ---- a small canvas chart: lines (decode) or bars (prefill) with hover read-out
function Chart(canvas,tip,empty){
  const c=canvas.getContext("2d");let model=null;
  function size(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;
    if(canvas.width!==Math.round(r.width*d)||canvas.height!==Math.round(r.height*d)){
      canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d)}
    c.setTransform(d,0,0,d,0,0);return[r.width,r.height]}
  function draw(m,hover){
    model=m;const[w,h]=size();c.clearRect(0,0,w,h);
    const has=m&&m.items.length;empty.style.display=has?"none":"flex";if(!has)return;
    const{L,R,T,B}=pad(m),pw=w-L-R,ph=h-T-B;
    const ys=yscale(m.ymax,T,ph,m.canLog),ymax=ys.max,ymax2=m.ymax2?nice(m.ymax2*1.08):null;
    const xmin=m.xmin,xmax=m.xmax>m.xmin?m.xmax:m.xmin+1;
    const X=x=>L+(x-xmin)/(xmax-xmin)*pw,Y=ys.Y;
    c.font="11px system-ui,sans-serif";c.textBaseline="middle";
    c.strokeStyle=css("--grid");c.lineWidth=1;
    ys.ticks.forEach(v=>{const y=Math.round(Y(v))+.5;
      c.beginPath();c.moveTo(L,y);c.lineTo(w-R,y);c.stroke();
      c.textAlign="right";c.fillStyle=css("--dim");c.fillText(fmt(v,ymax<10?1:0),L-6,y)});
    if(ymax2)for(let i=0;i<=4;i++){const y=Math.round(T+ph-ph*i/4)+.5;
      c.textAlign="left";c.fillStyle=css("--prefill");c.fillText(fmt(ymax2*i/4,ymax2<10?1:0),w-R+6,y)}
    c.textAlign="center";c.textBaseline="top";c.fillStyle=css("--dim");
    const nt=Math.max(2,Math.min(8,Math.floor(pw/90)));
    for(let i=0;i<=nt;i++){const x=xmin+(xmax-xmin)*i/nt;c.fillText(m.xfmt(x),X(x),T+ph+6)}
    m.draw(c,X,Y,{L,T,pw,ph,hover,Y2:y=>ymax2?T+ph-(y/ymax2)*ph:Y(y)});
  }
  canvas.onmousemove=ev=>{if(!model)return;const r=canvas.getBoundingClientRect(),mx=ev.clientX-r.left,my=ev.clientY-r.top;
    const hit=model.hit(mx,my,r.width,r.height);
    if(!hit){tip.style.display="none";draw(model,null);return}
    tip.innerHTML=hit.html;tip.style.display="block";
    tip.style.left=Math.min(r.width-tip.offsetWidth-4,Math.max(4,hit.x+12))+"px";
    tip.style.top=Math.max(2,hit.y-tip.offsetHeight-8)+"px";draw(model,hit)};
  canvas.onmouseleave=()=>{tip.style.display="none";if(model)draw(model,null)};
  return{draw,redraw:()=>model&&draw(model,null)};
}
const dchart=Chart($("dc"),$("dt"),$("de")),pchart=Chart($("pc"),$("pt"),$("pe"));

// one band per request, oldest on the left: decode curve on the left scale, prefill bar on the right scale
function decodeModel(s){
  const series=s.decode.filter(d=>d.points.length).map((d,i)=>({d,i}));
  if(!series.length)return null;
  const pf={};for(const p of s.prefill)pf[p.id]=p;
  const n=series.length;
  let ymax=0,ymax2=0;
  for(const{d}of series)for(const p of d.points)ymax=Math.max(ymax,p[2]);   // scale on the moving average: a bursty raw sample stays off it
  for(const{d}of series){const q=pf[d.id];if(q)ymax2=Math.max(ymax2,q.prefill_tok_s)}
  const col=d=>d.live?css("--live"):css("--accent");
  const alpha=i=>d=>d.live?1:0.28+0.72*((i+1)/n);
  const bx=(i,d,p)=>i+(d.tokens?Math.min(1,p[0]/d.tokens):0);   // output token position inside the request's band
  const barw=pw=>Math.max(3,pw/n*.16);
  return{items:series.map(x=>x.d),ymax,ymax2:ymax2||null,canLog:true,xmin:0,xmax:n,
    xfmt:x=>{const d=series[Math.min(n-1,Math.max(0,Math.floor(x)))];return d?hms(d.d.start):""},
    draw(c,X,Y,g){
      c.lineJoin="round";const bw=barw(g.pw);
      series.forEach(({d,i})=>{const q=pf[d.id];if(!q)return;                     // prefill, own scale, at the start of the band
        c.globalAlpha=.55;c.fillStyle=css("--prefill");const y=g.Y2(q.prefill_tok_s);c.fillRect(X(i+.05),y,bw,g.T+g.ph-y)});
      c.globalAlpha=1;
      series.forEach(({d,i})=>{c.globalAlpha=Math.max(0.2,alpha(i)(d)*0.5);c.strokeStyle=col(d);c.lineWidth=1.3;   // raw rate: dim, cut off at the top
        c.beginPath();d.points.forEach((p,k)=>k?c.lineTo(X(bx(i,d,p)),Y(p[1])):c.moveTo(X(bx(i,d,p)),Y(p[1])));c.stroke()});
      series.forEach(({d,i})=>{c.globalAlpha=alpha(i)(d);c.strokeStyle=col(d);c.lineWidth=d.live?2.6:2;  // moving average: bright, in front
        c.beginPath();d.points.forEach((p,k)=>k?c.lineTo(X(bx(i,d,p)),Y(p[2])):c.moveTo(X(bx(i,d,p)),Y(p[2])));c.stroke();
        if(d.live){const p=d.points[d.points.length-1];c.fillStyle=col(d);c.beginPath();c.arc(X(bx(i,d,p)),Y(p[2]),3.5,0,7);c.fill()}});
      c.globalAlpha=1;
      if(g.hover){c.strokeStyle=css("--dim");c.setLineDash([3,3]);c.beginPath();c.moveTo(g.hover.px,g.T);c.lineTo(g.hover.px,g.T+g.ph);c.stroke();c.setLineDash([])}},
    hit(mx,my,w,h){const{L,R,T,B}=pad({ymax2}),pw=w-L-R,ph=h-T-B;
      if(my<T||my>T+ph)return null;
      const X=x=>L+x/n*pw,Y=yscale(ymax,T,ph,true).Y,bw=barw(pw);
      const Y2=v=>ymax2?T+ph-v/nice(ymax2*1.08)*ph:null;
      for(const{i,d}of series){const q=pf[d.id],x0=X(i+.05);          // the bar is narrow: read it before the lines
        if(q&&mx>=x0-2&&mx<=x0+bw+2&&my>=Y2(q.prefill_tok_s)-2)
          return{x:x0+bw/2,y:Y2(q.prefill_tok_s),
            html:"<b>"+fmt(q.prefill_tok_s,0)+" tok/s</b> prefill (right scale)<br>"+fmt(q.new_tokens,0)+" new tokens"+(q.prompt_est?" (est.)":"")+
            (q.cached?", "+fmt(q.cached,0)+" cached":"")+(q.effective_tok_s?"<br>"+fmt(q.effective_tok_s,0)+" tok/s effective (all prompt tokens)":"")+"<br>ttft "+fmt(q.ttft_ms,0)+" ms<br><span style='color:var(--dim)'>"+hms(q.start)+"</span>"}}
      let best=null,bd=1e9;
      for(const{d,i}of series)for(const p of d.points){const px=X(bx(i,d,p)),dd=Math.abs(mx-px)+Math.abs(my-Y(p[2]))*.35;
        if(dd<bd){bd=dd;best={d,p,px}}}
      if(!best||bd>40)return null;const{d,p,px}=best;
      return{x:px,y:Y(p[2]),px,
        html:"<b>"+fmt(p[2])+" tok/s</b> decode (left scale)<br>"+fmt(p[1])+" tok/s raw<br>token "+fmt(p[0],0)+" of "+fmt(d.tokens,0)+
        "<br><span style='color:var(--dim)'>"+hms(d.start)+(d.live?" - live":"")+"</span>"}}};
}
function prefillModel(s){
  const items=s.prefill.slice(-30);if(!items.length)return null;
  const ymax=Math.max(...items.map(p=>p.prefill_tok_s)),n=items.length;
  return{items,ymax,xmin:0,xmax:n,xfmt:x=>x>=0&&x<n&&items[Math.min(n-1,Math.floor(x))]?hms(items[Math.min(n-1,Math.floor(x))].start):"",
    draw(c,X,Y,g){const bw=Math.max(2,g.pw/n*.7);
      items.forEach((p,i)=>{const x=X(i+.5)-bw/2,y=Y(p.prefill_tok_s);
        c.globalAlpha=p.prompt_est?.4:(g.hover&&g.hover.i===i?1:.85);c.fillStyle=css("--prefill");c.fillRect(x,y,bw,g.T+g.ph-y)});c.globalAlpha=1},
    hit(mx,my,w,h){const{L,R,T,B}=pad({}),pw=w-L-R,ph=h-T-B,i=Math.floor((mx-L)/pw*n);if(i<0||i>=n||my<T||my>T+ph)return null;
      const p=items[i],ymax2=nice(ymax*1.08);
      return{i,x:L+(i+.5)/n*pw,y:T+ph-p.prefill_tok_s/ymax2*ph,
        html:"<b>"+fmt(p.prefill_tok_s,0)+" tok/s</b> prefill<br>"+fmt(p.new_tokens,0)+" new tokens"+(p.prompt_est?" (est.)":"")+
        (p.cached?", "+fmt(p.cached,0)+" cached":"")+(p.effective_tok_s?"<br>"+fmt(p.effective_tok_s,0)+" tok/s effective (all prompt tokens)":"")+"<br>ttft "+fmt(p.ttft_ms,0)+" ms<br><span style='color:var(--dim)'>"+hms(p.start)+"</span>"}}};
}

function card(label,val,color){return"<div class='card'"+(color?" style='border-color:"+color+"'":"")+"><b>"+val+"</b><span>"+label+"</span></div>"}

function render(s){
  S=s;
  const live=s.live[0],pend=s.pending[0];
  const lv=s.decode.find(d=>d.live),lp=lv&&lv.points.length?lv.points[lv.points.length-1][1]:null;
  const chip=$("status");chip.className="chip";
  if(pend&&!live){chip.classList.add("pre");$("stext").textContent="prefill "+fmt(pend.elapsed_s,1)+" s"+(pend.prompt_guess?" (~"+fmt(pend.prompt_guess,0)+" tok)":"")}
  else if(live){chip.classList.add("gen");$("stext").textContent="generating, "+fmt(live.tokens,0)+" tok"}
  else $("stext").textContent="idle";
  const big=$("big");
  if(live){big.textContent=fmt(lp!=null?lp:live.tok_s);big.className="big live";$("bigl").textContent="tok/s now (last "+s.window+"+ tokens)"}
  else{big.textContent=fmt(s.avg_tok_s);big.className="big";$("bigl").textContent="tok/s, average of last "+s.requests+" requests"}
  const ttft=median(s.last.map(r=>r.ttft_ms).filter(x=>x>0)),pf=s.prefill.length?s.prefill[s.prefill.length-1].prefill_tok_s:null;
  $("cards").innerHTML=card("peak, rolling "+s.window+" tok",fmt(s.peak_rolling_tok_s),css("--accent"))+
    card("last prefill tok/s",fmt(pf,0),css("--prefill"))+card("median time to first token",ttft==null?"-":fmt(ttft,0)+" ms")+
    card("requests since start",fmt(s.total_requests,0))+card("tokens since start",fmt(s.total_tokens,0))+card("uptime",dur(s.uptime_s));
  dchart.draw(decodeModel(s),null);pchart.draw(prefillModel(s),null);
  $("dl").innerHTML="<span><i style='background:var(--accent)'></i>finished request</span><span><i style='background:var(--live)'></i>live</span><span><i style='height:2px;background:var(--dim)'></i>"+MA+"-sample moving average (bright)</span><span><i style='height:1px;background:var(--dim);opacity:.5'></i>raw rate (dim, cut off at the top)</span><span><i style='background:var(--prefill);height:8px;opacity:.55'></i>prefill tok/s (right scale)</span>";
  const rows=[];
  for(const e of s.live)rows.push("<tr class='l'><td>"+esc(hms(e.start))+"</td><td class='m'>"+esc(e.model)+"</td><td>"+fmt(e.prompt_tokens,0)+"</td><td>"+fmt(e.cached,0)+"</td><td>-</td><td>"+fmt(e.ttft_ms,0)+" ms</td><td>"+fmt(e.tokens,0)+"</td><td>"+(lp!=null?fmt(lp):fmt(e.tok_s))+" (live)</td></tr>");
  for(const e of s.last.slice(0,15))rows.push("<tr><td>"+esc(e.start?hms(e.start):"")+"</td><td class='m' title='"+esc(e.path)+"'>"+esc(e.model||e.path)+"</td><td>"+(e.prompt_tokens!=null?(e.prompt_est?"~":"")+fmt(e.prompt_tokens,0):"-")+"</td><td>"+(e.cached?fmt(e.cached,0):"-")+"</td><td>"+(e.prefill_tok_s!=null?fmt(e.prefill_tok_s,0):(e.prompt_tokens&&e.ttft_ms?"<span style='color:var(--dim)' title='fewer than 128 uncached tokens: the time to first token is overhead, not reading speed'>cache hit</span>":"-"))+(e.effective_tok_s?"<br><span style='color:var(--dim)' title='all prompt tokens / time to first token'>"+fmt(e.effective_tok_s,0)+" eff.</span>":"")+"</td><td>"+(e.ttft_ms?fmt(e.ttft_ms,0)+" ms":"-")+"</td><td>"+fmt(e.tokens,0)+"</td><td>"+fmt(e.tok_s)+"</td></tr>");
  $("rows").innerHTML=rows.join("")||"<tr><td colspan='8' style='text-align:center;color:var(--dim)'>no requests yet</td></tr>";
}
let busy=false;
function poll(){if(busy||document.hidden)return;busy=true;
  fetch("/stats?window="+win,{cache:"no-store"}).then(r=>r.json()).then(render).catch(()=>{
    const chip=$("status");chip.className="chip err";$("stext").textContent="proxy unreachable"}).finally(()=>busy=false)}
addEventListener("resize",()=>{dchart.redraw();pchart.redraw()});
document.addEventListener("visibilitychange",poll);
mark();$("target").textContent=location.host;$("ver").textContent="page v5 (decode left scale goes logarithmic above 1000 tok/s) -";poll();setInterval(poll,1000);
</script></body></html>
"""


def page():
    return PAGE


# Harnesses swallow most leading "/" or "!" input, so the trigger is a plain word with a
# prefix no harness uses, matched as a whole token anywhere in the last message.
# "/tps" stays for clients that pass it through.
TRIGGERS = {p + w for p in ("%", ";", "::") for w in ("tps", "tps_all")} | {"/tps", "/tokens"}


def command(body):
    """Return the trigger if the final message of a chat request contains one."""
    try:
        last = (json.loads(body).get("messages") or [])[-1]
        if last.get("role") != "user":    # e.g. tool result: an agent loop is mid-turn
            return None
    except (ValueError, IndexError, AttributeError):
        return None
    c = last.get("content")
    if isinstance(c, list):               # content-parts form
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return next((t for t in (c or "").lower().split() if t in TRIGGERS), None)


def report_all():
    t = TOTAL
    up = time.time() - t["since"]
    if not t["requests"]:
        return f"llmproxy since start ({up / 60:.0f} min): no completions measured yet."
    return "\n".join([
        f"llmproxy tok/s since start ({time.strftime('%F %T', time.localtime(t['since']))}, up {up / 60:.0f} min)",
        f"- generation speed: {t['tokens'] / t['secs']:.1f} tok/s average" if t["secs"] else
        "- generation speed: n/a",
        f"- peak request: {t['peak']} tok/s",
        f"- {t['requests']} requests, {t['tokens']} tok in {t['secs']:.0f} s of generation",
        f"- wall-clock throughput: {t['tokens'] / up:.2f} tok/s over the whole uptime (includes idle time)"])


def report():
    s = snapshot()
    if not (s["live"] or s["last"]):
        return "llmproxy: no completions measured yet.\n(send %tps_all for totals since start)"
    lines = ["llmproxy tok/s"]
    for e in s["live"]:
        lines.append(f"- live: {e['tok_s']} tok/s, {e['tokens']} tok so far, {e['ttft_ms']} ms ttft")
    for i, e in enumerate(s["last"][:5]):
        lines.append(f"- {'last' if i == 0 else f'-{i}  '}: {e['tok_s']} tok/s, {e['tokens']} tok, "
                     f"{e['ttft_ms']} ms ttft, {e['model']}")
    lines.append(f"- average over the last {s['requests']} requests: {s['avg_tok_s']} tok/s ({s['tokens']} tok)")
    lines.append("(send %tps_all for totals since the proxy started)")
    return "\n".join(lines)


def completion(text, model, stream):
    """Synthetic OpenAI chat completion (or SSE stream) carrying `text`."""
    base = {"id": "llmproxy-cmd", "created": int(time.time()), "model": model or "llmproxy"}
    if not stream:
        return json.dumps({**base, "object": "chat.completion", "choices": [{
            "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}).encode()
    ev = lambda delta, fin=None: b"data: " + json.dumps({**base, "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}).encode() + b"\n\n"
    return (ev({"role": "assistant", "content": ""}) + ev({"content": text}) + ev({}, "stop")
            + b"data: [DONE]\n\n")


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
        route, _, query = self.path.partition("?")
        if route in ("/", "/stats"):
            if route == "/stats":          # /stats?window=N: decode curve averaged over >= N tokens (min 10)
                try:
                    window = int(urllib.parse.parse_qs(query).get("window", [MIN_WINDOW])[0])
                except ValueError:
                    window = MIN_WINDOW
                body = json.dumps(snapshot(window)).encode()
            else:
                body = page().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/json" if route == "/stats" else "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)

        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        trigger = command(body) if self.command == "POST" and "chat/completions" in self.path else None
        if trigger:
            try:                           # answered here, never sent upstream, not logged
                req = json.loads(body)
            except ValueError:
                req = {}
            stream = bool(req.get("stream"))
            data = completion(report_all() if trigger.endswith("_all") else report(), req.get("model"), stream)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)

        headers = {k: v for k, v in self.headers.items() if k.lower() not in DROP}
        req = urllib.request.Request(UPSTREAM + self.path, body or None, headers, method=self.command)
        t0 = time.time()
        model = ""
        try:
            model = json.loads(body).get("model", "")
        except Exception:
            pass
        key = next(KEY)
        p_est = prompt_estimate(body) if self.command == "POST" else 0
        if self.command == "POST" and "completion" in self.path:
            PENDING[key] = {"id": key, "path": self.path, "model": model, "t0": t0, "prompt_guess": p_est,
                            "first": None}      # shown as "prefill" on the dashboard until the first token
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            resp = e                      # pass upstream 4xx/5xx through untouched
        except Exception as e:
            PENDING.pop(key, None)
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

        if "event-stream" in ctype:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            e = LIVE[key] = {"id": key, "path": self.path, "model": model, "tokens": 0, "start": t0,
                             "t0": t0, "first": None, "last": None, "out": "", "ev": [],
                             "prompt_guess": p_est}
            PENDING.pop(key, None)          # from here on LIVE carries it (a pending row until its first token)
            events = chars = usage = cum = 0
            try:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    n, txt, u = sse(line)
                    comp, prompt, cached = usage_numbers(u)
                    events, chars = events + n, chars + len(txt)
                    usage = max(usage, comp)                # cumulative in llama.cpp
                    if prompt is not None:
                        e["prompt_tokens"], e["cached"] = prompt, cached or 0
                    e["out"] += txt
                    if n:
                        now = time.time()
                        if e["first"] is None:
                            e["first"] = now
                            e["ev"].append((0.0, 0))        # the decode timeline starts at the first token
                        e["last"] = now
                        cum += event_tokens(txt)
                        e["ev"].append((now - e["first"], cum))
                    e["tokens"] = usage or max(events, chars // 4)
                    self.chunk(line)
                self.wfile.write(b"0\r\n\r\n")
                e["secs"] = e["last"] - e["first"] if e["first"] else time.time() - t0
                e["tok_s"] = round(rate(e), 1)
                e["ttft_ms"] = round((e["first"] - t0) * 1000) if e["first"] else 0
                e["scaled"] = bool(usage)             # the server's own count calibrates the per-event estimate
                if "prompt_tokens" not in e and p_est:
                    e["prompt_tokens"], e["cached"], e["prompt_est"] = p_est, 0, True
                else:
                    e["prompt_est"] = False
                e["new_tokens"], e["prefill_tok_s"], e["effective_tok_s"] = prefill_stats(e.get("prompt_tokens"), e.get("cached"), e["ttft_ms"])
                record({k: v for k, v in e.items() if k not in ("t0", "out", "prompt_guess")})
                dump(body, e, e["out"])
            finally:
                LIVE.pop(key, None)
        else:
            try:
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            finally:
                PENDING.pop(key, None)
            usage_j = None
            try:
                j = json.loads(data)
                out = ""
                for c in j.get("choices", []):
                    m = c.get("message") or c
                    out += m.get("content") or m.get("reasoning_content") or m.get("text") or ""
                usage_j = j.get("usage")
                tok = (usage_j or {}).get("completion_tokens") or est(out)
            except Exception:
                out, tok = "", 0
            secs = time.time() - t0
            if code == 200 and "/completion" in self.path:
                _, prompt, cached = usage_numbers(usage_j)
                entry = {"id": key, "path": self.path, "model": model, "tokens": tok, "secs": secs, "start": t0,
                         "tok_s": round(tok / secs, 1) if secs else 0.0, "ttft_ms": 0,
                         "prompt_tokens": prompt if prompt is not None else (p_est or None),
                         "cached": cached or 0, "prompt_est": prompt is None}
                record(entry)
                dump(body, entry, out)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = route


def selftest():
    assert prefill_stats(485, 484, 117) == (1, None, 4145), prefill_stats(485, 484, 117)      # cache hit: no prefill rate
    assert prefill_stats(16720, 4947, 18875) == (11773, 624, 886), prefill_stats(16720, 4947, 18875)
    assert prefill_stats(40, 0, 280) == (40, None, 143)                                         # tiny prompt: overhead only
    assert prefill_stats(None, None, 0) == (0, None, None)
    ev = ch = us = 0
    for line in (b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
                 b'data: {"choices":[{"delta":{"reasoning_content":" think"}}]}\n',
                 b'data: [DONE]\n',
                 b'data: {"usage":{"completion_tokens":2}}\n',
                 b'data: {"usage":{"completion_tokens":1}}\n'):
        n, txt, u = sse(line)
        ev, ch, us = ev + n, ch + len(txt), max(us, usage_numbers(u)[0])
    assert (ev, ch, us) == (2, 11, 2), (ev, ch, us)
    n, txt, u = sse(b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n')
    assert (n, txt, u) == (1, "Hello", None)
    # usage on the final chunk, next to an empty delta (Strata, llama.cpp): prompt, completion and cached tokens
    n, txt, u = sse(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":24,'
                    b'"completion_tokens":38,"prompt_tokens_details":{"cached_tokens":10}}}\n')
    assert (n, txt, usage_numbers(u)) == (0, "", (38, 24, 10)), (n, txt, u)
    assert usage_numbers(None) == (0, None, None)
    # prefill: only the tokens that were really read count, over the time to the first token
    assert prefill_stats(1000, 0, 500) == (1000, 2000, 2000)
    assert prefill_stats(1000, 900, 500) == (100, None, 2000)          # 100 new tokens: below MIN_PREFILL_NEW
    assert prefill_stats(50, 50, 500) == (0, None, 100) and prefill_stats(None, None, 500) == (0, None, None)
    assert prompt_estimate(json.dumps({"messages": [{"role": "user", "content": "x" * 400}]})) == 100
    assert prompt_estimate(b"not json") == 0
    # decode curve: 40 tokens at a steady 20 tok/s -> every point 20, windows of >= 10 tokens
    steady = [(i / 20, i) for i in range(41)]
    pts = curve(steady, 1.0, 10)
    assert pts and pts[0][0] == 10 and all(abs(v - 20) < 0.2 and abs(m - 20) < 0.2 for _, v, m in pts), pts[:3]
    assert curve(steady[:9], 1.0, 10) == []                       # under 10 tokens: no point yet
    assert curve(steady, 1.0, 1)[0][0] == 10                      # the window never drops below 10
    burst = [(0.0, 0)] + [(0.5, k) for k in range(1, 7)] + [(1.0, k) for k in range(7, 13)]
    assert all(v >= 10 for _, v, _m in curve(burst, 1.0, 10)), curve(burst, 1.0, 10)   # a burst spreads over its window
    assert len(curve([(i / 50, i) for i in range(2001)], 1.0, 10)) <= 201
    ramp = [(i * i / 200, i) for i in range(1, 80)]                 # a slowing request: the average lags above the raw rate
    rp = curve(ramp, 1.0, 10)
    assert len(rp) > MA_PERIOD and rp[-1][2] > rp[-1][1] and rp[0][2] == rp[0][1], rp[:2]
    one = [(i * 0.05, i) for i in range(30)] + [(1.5 + i * 0.5, 30 + i) for i in range(1, 30)]   # fast, then slow
    op = curve(one, 1.0, 10)
    assert abs(op[-1][2] - sum(p[1] for p in op[-MA_PERIOD:]) / MA_PERIOD) < 0.15, op[-1]   # mean of the last 20 samples
    assert curve(steady, 2.0, 10)[0][0] == 10 and abs(curve(steady, 2.0, 10)[0][1] - 40) < 0.5   # usage calibration
    LIVE[99] = {"id": 99, "path": "/v1/chat/completions", "model": "m", "tokens": 12, "t0": time.time() - 3,
                "start": time.time() - 3, "first": time.time() - 1, "last": time.time(), "out": "secret",
                "ev": [(i / 12, i) for i in range(13)], "prompt_guess": 5}
    PENDING[98] = {"id": 98, "path": "/v1/chat/completions", "model": "m", "t0": time.time() - 2,
                   "prompt_guess": 7, "first": None}
    snap = snapshot(25)
    assert snap["window"] == 25 and snapshot(3)["window"] == 10
    assert snap["pending"][0]["prompt_guess"] == 7 and snap["live"][0]["id"] == 99
    assert "out" not in snap["live"][0] and "ev" not in snap["live"][0], snap["live"][0].keys()
    assert snap["decode"][-1]["live"] and snap["decode"][-1]["points"] == []      # 12 tokens: under a 25-token window
    json.dumps(snap)
    LIVE.pop(99), PENDING.pop(98)
    assert "<canvas" in page() and "/stats?window=" in page()
    import re, shutil, subprocess, tempfile            # the dashboard script must parse and run (checked when node is installed)
    node = shutil.which("node")
    if node:
        harness = """\
globalThis.devicePixelRatio=1;globalThis.location={host:"x"};globalThis.setInterval=()=>{};
globalThis.localStorage={getItem:()=>null,setItem(){}};globalThis.addEventListener=()=>{};
globalThis.fetch=()=>{const o={then:()=>o,catch:()=>o,finally:()=>o};return o};
globalThis.getComputedStyle=()=>({getPropertyValue:()=>"#fff"});
globalThis.calls=[];
const stubctx=()=>new Proxy({},{get:(t,k)=>(...a)=>{calls.push(k);return 0},set:()=>true});
const stubel=()=>({style:{},dataset:{},classList:{toggle(){},add(){}},textContent:"",innerHTML:"",
                   getContext:stubctx,getBoundingClientRect:()=>({width:420,height:230}),offsetWidth:100,offsetHeight:20});
const stubels={};globalThis.document={documentElement:{},getElementById:id=>stubels[id]||(stubels[id]=stubel()),
                querySelectorAll:()=>[],addEventListener(){}};
"""
        demo = {"requests": 2, "tokens": 300, "avg_tok_s": 30.0, "window": 10, "uptime_s": 100,
                "total_requests": 2, "total_tokens": 300, "peak_request_tok_s": 40.0, "peak_rolling_tok_s": 600.0,
                "pending": [], "live": [], "last": [],
                "decode": [{"id": 1, "live": False, "start": 1700000000, "model": "m", "tokens": 20,
                            "points": [[5, 600.0, 30.0], [10, 500.0, 32.0], [20, 400.0, 34.0]]},
                           {"id": 2, "live": True, "start": 1700000001, "model": "m", "tokens": 12,
                            "points": [[10, 200.0, 25.0]]}],
                "prefill": [{"id": 1, "start": 1700000000, "prompt_tokens": 9000, "cached": 0, "new_tokens": 9000,
                             "ttft_ms": 900, "prefill_tok_s": 10000, "prompt_est": False, "model": "m"}]}
        check = """\
const m=decodeModel(DEMO);
if(m.ymax!==34)throw new Error("left scale must follow the moving average, not the raw spikes: "+m.ymax);
if(m.ymax2!==10000)throw new Error("prefill must keep its own right scale: "+m.ymax2);
dchart.draw(m,null);
if(!calls.includes("fillRect"))throw new Error("no prefill bar drawn");
const hm=decodeModel(DEMO),bar=hm.hit(60,107,420,230),line=hm.hit(128,79,420,230);
if(!bar||!bar.html.includes("right scale"))throw new Error("prefill bar hover: "+JSON.stringify(bar));
if(!line||!line.html.includes("left scale"))throw new Error("decode line hover: "+JSON.stringify(line));
const lin=yscale(900,0,100,true),lg=yscale(5000,0,100,true);
if(lin.log||lg.log!==true)throw new Error("log scale only above "+LOG_ABOVE+" tok/s");
if(lg.ticks.join()!=="1,10,100,1000,10000"||lg.Y(1)!==100||lg.Y(10000)!==0||lg.Y(0)!==100||Math.abs(lg.Y(100)-50)>1e-9)
  throw new Error("log scale geometry: "+lg.ticks+" "+lg.Y(100));
if(yscale(5000,0,100,false).log)throw new Error("prefill chart stays linear");
"""
        with tempfile.NamedTemporaryFile("w", suffix=".js") as f:
            f.write(harness + re.search(r"<script>(.*)</script>", page(), re.S).group(1)
                    + check.replace("DEMO", json.dumps(demo)))
            f.flush()
            r = subprocess.run([node, f.name], capture_output=True, text=True)
            assert r.returncode == 0, (r.stdout + r.stderr)[:600]
    for content in ("/tps", "  %TPS\n", ";tps", "please ::tps now", [{"type": "text", "text": "/tokens"}]):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"},
                                        {"role": "user", "content": content}]})
        assert command(body), content
    assert command(json.dumps({"messages": [{"role": "user", "content": "%tps_all"}]})) == "%tps_all"
    record({"path": "/v1/chat/completions", "model": "m", "tokens": 10, "secs": 2.0, "tok_s": 5.0, "ttft_ms": 1})
    assert "5.0 tok/s average" in report_all() and "%tps_all" in report()
    assert not command(json.dumps({"messages": [{"role": "user", "content": "what is tps?"}]}))
    assert not command(json.dumps({"messages": [{"role": "user", "content": "%tps"},
                                                {"role": "tool", "content": "x"}]}))
    assert not command(b"not json")
    assert json.loads(completion("x", "m", False))["choices"][0]["message"]["content"] == "x"
    assert completion("x", "m", True).endswith(b"data: [DONE]\n\n")
    assert load_cached is not None and DEFAULTS[1] == "8787"
    print("selftest ok")


if __name__ == "__main__":
    if SELFTEST:
        raise SystemExit(selftest())
    save_cached(UPSTREAM, PORT, LOGDIR)
    if FROM_CACHE:
        print(f"proxy: no parameters given, reusing the last ones from {CACHE}")
    print(f"proxy on :{PORT} -> {UPSTREAM}, dashboard http://localhost:{PORT}/, "
          f"log {LOGFILE}")
    try:
        srv = DualStackServer(("::", PORT), H)
    except OSError:                       # no IPv6 on this host: plain IPv4 on all interfaces
        srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    srv.serve_forever()
