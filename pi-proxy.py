#!/usr/bin/env python3
"""pi-proxy.py [on|off] [port] — point pi's configured LLM through the toksec proxy.

on  : reads pi's default provider from ~/.pi/agent/models.json, starts the proxy in
      front of whatever baseUrl pi is configured to use, and rewrites that baseUrl to
      the proxy. The original file is saved as models.json.orig.
off : restores models.json.orig (pi back on its real upstream).

Nothing else about pi's config is touched: same provider, same model, same key.
"""
import json, os, shutil, subprocess, sys, urllib.parse

CFG = os.path.expanduser("~/.pi/agent/models.json")
SETTINGS = os.path.expanduser("~/.pi/agent/settings.json")
BAK = CFG + ".orig"
PROXY = os.path.expanduser("~/bin/LLMproxy/toksec.py")
LOGDIR = os.path.expanduser("~/logs/llmproxy")

STATE = sys.argv[1] if len(sys.argv) > 1 else "on"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8787


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


if STATE == "off":
    if os.path.exists(BAK):
        os.replace(BAK, CFG)
        print(f"pi: restored {CFG} (proxy left running, kill it if you want)")
    else:
        print("pi: no backup found, config unchanged")
    raise SystemExit()

if STATE != "on":
    raise SystemExit(f"unknown state {STATE!r}: use on or off")

d = load(CFG)
prov = load(SETTINGS).get("defaultProvider")
if prov not in d["providers"]:
    raise SystemExit(f"pi: defaultProvider {prov!r} not in {CFG}")
p = d["providers"][prov]
orig = p["baseUrl"]

path = urllib.parse.urlparse(orig).path          # usually "/v1"
upstream = orig[: len(orig) - len(path)] if path else orig

if urllib.parse.urlparse(orig).port == PORT and "localhost" in orig:
    print(f"pi: already on the proxy ({orig})")
    raise SystemExit()

listening = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
if f":{PORT} " not in listening:
    os.makedirs(LOGDIR, exist_ok=True)
    log = open(f"{LOGDIR}/proxy-{PORT}.log", "ab")
    subprocess.Popen([sys.executable, PROXY, upstream, str(PORT), LOGDIR],
                     stdout=log, stderr=log, start_new_session=True)
    how = "started"
else:
    how = "reusing existing proxy"

p["baseUrl"] = f"http://localhost:{PORT}{path}"
shutil.copyfile(CFG, BAK)
save(CFG, d)
print(f"pi: provider {prov} baseUrl {orig} -> {p['baseUrl']} "
      f"(upstream {upstream}, proxy {how} on :{PORT}, log {LOGDIR})")
print(f"pi: restart pi to pick it up; dashboard http://localhost:{PORT}/ ; undo with pi-proxy.py off")
