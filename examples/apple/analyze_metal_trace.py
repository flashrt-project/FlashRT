"""Measure encoder-level gaps from xctrace XML exports, without double counting.

Export the metal-gpu-intervals and os-signpost (target-pid=SINGLE) tables to
gpu.xml and signposts.xml. Pass their directory and the recorded Python PID.
CPU signposts describe submission, not GPU stage execution. Active encoders
may themselves contain stalls; this analysis cannot measure shader occupancy.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def xml_rows(path):
    refs = {}
    columns = []
    for _, element in ET.iterparse(path, events=["end"]):
        if element.tag == "schema":
            columns = [col.findtext("mnemonic") for col in element.findall("col")]
        if element.tag != "row":
            continue

        def resolve(node):
            if "ref" in node.attrib:
                return refs[node.attrib["ref"]]
            value = (node.text or "", node.get("fmt", node.text or ""))
            for child in node:
                resolve(child)
            if "id" in node.attrib:
                refs[node.attrib["id"]] = value
            return value

        yield {key: resolve(node) for key, node in zip(columns, element)}
        element.clear()


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def measure(start, end, intervals):
    active = merge((max(start, a), min(end, b)) for a, b in intervals
                   if b > start and a < end)
    gaps = []
    cursor = start
    for a, b in active:
        if a > cursor:
            gaps.append([cursor, a])
        cursor = b
    if cursor < end:
        gaps.append([cursor, end])
    busy = sum(b - a for a, b in active)
    idle = end - start - busy
    leading = active[0][0] - start if active else end - start
    trailing = end - active[-1][1] if active else 0
    return dict(wall_ms=(end-start)/1e6, busy_ms=busy/1e6,
                idle_ms=idle/1e6, idle_pct=100*idle/(end-start),
                leading_ms=leading/1e6, trailing_ms=trailing/1e6,
                internal_idle_ms=(idle-leading-trailing)/1e6,
                max_gap_ms=max((b-a for a, b in gaps), default=0)/1e6,
                active=active, gaps=sorted(gaps, key=lambda x: x[1]-x[0], reverse=True))


def analyze(directory, pid):
    opened, spans = {}, []
    for row in xml_rows(directory / "signposts.xml"):
        if row["subsystem"][1] != "com.flashrt.apple":
            continue
        if not row["process"][1].endswith(f"({pid})"):
            continue
        key = row["identifier"][0]
        time = int(row["time"][0])
        if row["event-type"][0] == "Begin":
            opened[key] = (time, row["message"][1])
        elif row["event-type"][0] == "End":
            start, name = opened.pop(key)
            spans.append(dict(start=start, end=time, name=name))
    if opened:
        raise ValueError("Incomplete stage intervals: recording may have been truncated")
    inferences = sorted((s for s in spans if s["name"].startswith("inference.")),
                        key=lambda s: s["start"])
    if not inferences:
        raise ValueError("No complete inference signposts found")
    lo, hi = inferences[0]["start"], inferences[-1]["end"]
    gpu = []
    for row in xml_rows(directory / "gpu.xml"):
        start = int(row["start"][0])
        end = start + int(row["duration"][0])
        if end <= lo or start >= hi or row["state"][1] != "Active":
            continue
        gpu.append(dict(start=start, end=end, process=row["process"][1],
                        channel=row["channel-name"][1], label=row["event-label"][1],
                        encoder_id=row["encoder-id"][0]))
    own = [g for g in gpu if g["process"].endswith(f"({pid})")]
    if not own:
        raise ValueError("No target GPU intervals found")
    submissions = []
    if (directory / "submissions.xml").exists():
        for row in xml_rows(directory / "submissions.xml"):
            start = int(row["start"][0])
            if lo <= start < hi and row["process"][1].endswith(f"({pid})"):
                submissions.append(dict(start=start,
                    end=start+int(row["duration"][0]),
                    encoders=int(row["num-encoders"][0]),
                    label=row["event-label"][1]))
    driver = []
    if (directory / "driver.xml").exists():
        for row in xml_rows(directory / "driver.xml"):
            start = int(row["start"][0])
            end = start + int(row["duration"][0])
            if end > lo and start < hi and row["process"][1].endswith(f"({pid})"):
                driver.append(dict(start=start, end=end, kind=row["event-type"][1],
                                   label=row["event-label"][1], thread=row["thread"][1]))
    for inf in inferences:
        start, end = inf["start"], inf["end"]
        inf["target"] = measure(start, end, [(g["start"], g["end"]) for g in own])
        inf["device"] = measure(start, end, [(g["start"], g["end"]) for g in gpu])
        inf["command_buffers"] = [s for s in submissions if start <= s["start"] < end]
        inf["native"] = []
        inf["driver"] = [d for d in driver if d["end"] > start and d["start"] < end]
        wiring = [d for d in inf["driver"] if d["kind"] == "Wire Memory"]
        inf["wire_memory_count"] = len(wiring)
        inf["wire_memory_ms"] = measure(start, end,
            [(d["start"], d["end"]) for d in wiring])["busy_ms"]
        for span in spans:
            if span["name"] in ("native.execute", "native.full_pipeline") and start <= span["start"] < end:
                inf["native"].append(dict(**span, **measure(span["start"], span["end"],
                    [(g["start"], g["end"]) for g in own])))
    summary = dict(pid=pid, resolution="Active Metal encoder interval union; not shader occupancy",
                   inferences=inferences, stages=spans)
    (directory / "gaps.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Perfetto / chrome://tracing compatible extract. Preserve a common origin
    # across runs so inter-run gaps are visible but excluded from statistics.
    events = []
    for span in spans:
        events.append(dict(name=span["name"], cat="host stage", ph="X", pid=pid,
                           tid="CPU submission ranges", ts=(span["start"]-lo)/1000,
                           dur=(span["end"]-span["start"])/1000))
    for g in gpu:
        events.append(dict(name=g["label"], cat="GPU encoder", ph="X", pid=g["process"],
                           tid=g["channel"], ts=(g["start"]-lo)/1000,
                           dur=(g["end"]-g["start"])/1000,
                           args=dict(encoder_id=g["encoder_id"])))
    for d in driver:
        events.append(dict(name=d["label"], cat="Metal driver memory", ph="X", pid=pid,
                           tid=d["thread"], ts=(d["start"]-lo)/1000,
                           dur=(d["end"]-d["start"])/1000))
    (directory / "timeline.json").write_text(json.dumps(dict(traceEvents=events)))
    render(directory, inferences)
    for name in sorted({i["name"] for i in inferences}):
        group = [i for i in inferences if i["name"] == name]
        print(name, {k: round(statistics.median(i["target"][k] for i in group), 3)
                     for k in ("wall_ms", "busy_ms", "idle_ms", "idle_pct",
                               "leading_ms", "trailing_ms", "internal_idle_ms", "max_gap_ms")})
        print("device idle median ms", statistics.median(i["device"]["idle_ms"] for i in group))


def render(directory, inferences):
    width, left = 1100, 180
    maximum = max(i["target"]["wall_ms"] for i in inferences)
    scale = (width-left-30)/maximum
    height = 60 + len(inferences)*94
    body = [f'<svg viewBox="0 0 1100 {height}" role="img" aria-label="GPU encoder activity timeline">']
    for tick in range(0, int(maximum)+1, 50):
        x = left+tick*scale
        body.append(f'<line x1="{x}" y1="35" x2="{x}" y2="{height-25}" stroke="#dce2eb"/>'
                    f'<text x="{x}" y="22" font-size="12">{tick} ms</text>')
    counts = {}
    for index, inf in enumerate(inferences):
        y = 55+index*94
        start = inf["start"]
        m = inf["target"]
        name = inf["name"].removeprefix("inference.")
        counts[name] = counts.get(name, 0) + 1
        body.append(f'<text x="5" y="{y+17}" font-size="14">{html.escape(name)} #{counts[name]}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{m["wall_ms"]*scale}" height="25" fill="#fee2e2"/>')
        for a,b in m["active"]:
            body.append(f'<rect x="{left+(a-start)/1e6*scale}" y="{y}" width="{(b-a)/1e6*scale}" height="25" fill="#2563eb"/>')
        for n in inf["native"]:
            body.append(f'<rect x="{left+(n["start"]-start)/1e6*scale}" y="{y+28}" width="{(n["end"]-n["start"])/1e6*scale}" height="5" fill="#16a34a"/>')
        for d in inf.get("driver", []):
            if d["kind"] == "Wire Memory":
                a, b = max(start, d["start"]), min(inf["end"], d["end"])
                body.append(f'<rect x="{left+(a-start)/1e6*scale}" y="{y+35}" width="{(b-a)/1e6*scale}" height="5" fill="#f59e0b"><title>{html.escape(d["label"])}</title></rect>')
        a,b = m["gaps"][0]
        body.append(f'<text x="{left}" y="{y+52}" font-size="12">Wall {m["wall_ms"]:.2f} ms · active {m["busy_ms"]:.2f} ms · no target encoder {m["idle_pct"]:.1f}% · largest gap {(b-a)/1e6:.2f} ms</text>')
        if inf.get("wire_memory_count"):
            body.append(f'<text x="{left}" y="{y+69}" font-size="12" fill="#92400e">Wire Memory: {inf["wire_memory_count"]} events · {inf["wire_memory_ms"]:.2f} ms interval union</text>')
    body.append('</svg>')
    page = '''<!doctype html><meta charset="utf-8"><title>FlashRT GPU pipeline gaps</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#172033}svg{width:100%}p{line-height:1.6}</style>
<h1>FlashRT — Metal encoder timeline</h1>
<p>Blue: target GPU encoder active. Pink: no active target encoder. Green underline: native execution CPU scope. Orange: driver Wire Memory events.
Each row starts at its inference signpost. Warmup, model loading and inter-inference gaps are excluded.</p>
'''+''.join(body)+'''<p>These are instrumented runs. Encoder activity is not shader occupancy: stalls within an encoder are invisible here.
CPU stage markers are asynchronous submission ranges, not GPU stage durations. Other applications may use the GPU during pink regions.</p>
<p><a href="timeline.json">Perfetto / Chrome trace JSON</a> · <a href="gaps.json">Exact statistics</a></p>'''
    table = ['<h2>Command buffer encoding → submission</h2><p>Times relative to each inference start. GPU work starts after submission; host encoding may span multiple model stages.</p><table><tr><th>Run</th><th>Start ms</th><th>Submit ms</th><th>Encoders</th></tr>']
    for index, inf in enumerate(inferences):
        for cb in inf["command_buffers"]:
            table.append(f'<tr><td>{html.escape(inf["name"])} #{index+1}</td><td>{(cb["start"]-inf["start"])/1e6:.3f}</td><td>{(cb["end"]-inf["start"])/1e6:.3f}</td><td>{cb["encoders"]}</td></tr>')
    table.append('</table><style>td,th{padding:6px 16px;text-align:left;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>')
    (directory / "timeline.html").write_text(page+''.join(table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    analyze(args.directory, args.pid)
