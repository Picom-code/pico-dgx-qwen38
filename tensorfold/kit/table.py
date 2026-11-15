#!/usr/bin/env python3
# table.py NAME=suite.log ... : one bake-off row per suite log.
import re, sys
def g(pat, s, n=1, d="-"):
    m = re.search(pat, s, re.M); return m.group(n) if m else d
rows = []
for a in sys.argv[1:]:
    name, p = a.split("=", 1); s = open(p, errors="replace").read()
    r = {"engine": name}
    r["ready s / GPU GiB"] = "-"
    r["ttft ms"] = g(r"ttft short prompt: median (\d+)", s)
    r["chat c1"] = g(r"^chat 1 user: median ([\d.]+)", s); r["code c1"] = g(r"^code 1 user: median ([\d.]+)", s)
    r["sdm prose T0"] = g(r"sd-method prose T0 think-off: median ([\d.]+)", s); r["sdm prose T.7 think"] = g(r"sd-method prose T0.7 think-on: median ([\d.]+)", s)
    r["sdm code"] = g(r"sd-method code T0 think-off \(task 0\): median ([\d.]+)", s); r["sdm struct"] = g(r"sd-method structured T0 think-off: median ([\d.]+)", s)
    sp = re.findall(r"sdrun: c1 prose run \d: ([\d.]+)", s); sc = re.findall(r"sdrun: c1 code run \d: ([\d.]+)", s)
    med = lambda v: "-" if not v else f"{sorted(map(float, v))[len(v)//2]:.1f}"
    r["sparkDash prose/code"] = med(sp) + " / " + med(sc)
    r["sD catalogs p/s/c"] = " / ".join(g(rf"catalog {k}: \d+ prompts, mean ([\d.]+)", s) for k in ("prose", "structural", "code"))
    r["held chat/code/math"] = " / ".join(g(rf"^held {k}: ([\d.]+)", s) for k in ("chat", "code", "math")); r["held all"] = g(r"^held ALL: ([\d.]+)", s)
    for n in (4, 8, 16):
        r[f"distinct {n}"] = g(rf"^== distinct{n} [\d:]+\ndistinct {n} users: ([\d.]+)", s); r[f"copies {n}"] = g(rf"^== copies{n} [\d:]+\ncopies {n} users: ([\d.]+)", s)
    r["max slots distinct/copies"] = g(r"^== distinct_max(\d+)", s) + ": " + g(r"^== distinct_max\d+ [\d:]+\ndistinct \d+ users: ([\d.]+)", s) + " / " + g(r"^== copies_max\d+ [\d:]+\ncopies \d+ users: ([\d.]+)", s)
    r["16 queued"] = g(r"^== distinct16_queued [\d:]+\ndistinct 16 users: ([\d.]+)", s)
    r["sd wave5 prose/code"] = g(r"wave 5 prose T0.7 think-on: aggregate ([\d.]+)", s) + " / " + g(r"wave 5 code T0 think-off: aggregate ([\d.]+)", s)
    r["prefill 8K/32K/128K"] = " / ".join(re.findall(r"^prefill target \d+: .* (\d+) tok/s", s, re.M)) or "-"
    r["needle 200K"] = g(r"^needle 200000: prompt_tokens (\d+)  recall (\d/3)", s, 2); r["ctx"] = g(r"^ctx \d+: prompt_tokens (\d+)  recall (\d/3)", s, 1) + " " + g(r"^ctx \d+: prompt_tokens (\d+)  recall (\d/3)", s, 2)
    r["identity on/off"] = g(r"^identical (\d+/\d+)", s); r["img"] = g(r"^img: (\w+)", s)
    q = re.findall(r"^\s+(\w+)\s+(\d+)/(\d+)\s*$", s, re.M); r["qeval"] = f"{sum(int(a) for _, a, _ in q)}/{sum(int(b) for _, _, b in q)}" if q else "-"
    r["idle MemAvail GiB / GPU MiB"] = g(r"idle after load: MemAvailable (\d+) GiB, GPU MiB (\d*)", s) + " / " + g(r"idle after load: MemAvailable (\d+) GiB, GPU MiB (\d*)", s, 2)
    r["min MemAvail MiB"] = g(r"min MemAvailable during suite: (\d+)", s); r["GPU C max"] = g(r"gpu temp C min/max during suite: \d+/(\d+)", s)
    r["greedy"] = "REFUSED" if "GREEDY REFUSED" in s else "ok"
    rows.append(r)
keys = list(rows[0].keys()); w0 = max(len(k) for k in keys); ws = [max(len(r["engine"]), max(len(str(r[k])) for k in keys)) for r in rows]
line = "+" + "-" * (w0 + 2) + "+" + "+".join("-" * (w + 2) for w in ws) + "+"
print(line)
for k in keys:
    print("| " + k.ljust(w0) + " | " + " | ".join(str(r[k]).ljust(w) for r, w in zip(rows, ws)) + " |")
    if k == "engine": print(line)
print(line)
