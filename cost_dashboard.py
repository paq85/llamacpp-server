#!/usr/bin/env python3
"""Self-contained costs dashboard data collection and HTML rendering.

Reads from the same CSV files as CostLogger and produces a complete HTML page
with inline scripts (zero external CDN dependencies).
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_dashboard_data(
    requests_path: str,
    daily_path: str,
    cost_input_price: float,
    cost_cached_price: float,
    cost_output_price: float,
    model_name: str = "PAQ_LLAMACPP_SERVER",
) -> dict[str, Any] | None:
    """Read cost CSV logs and build a data dict for the dashboard page."""
    if not os.path.exists(requests_path):
        return None

    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())
    seven_days_ago_start = datetime.combine(today - timedelta(days=7), datetime.min.time())
    thirty_days_ago_start = datetime.combine(today - timedelta(days=30), datetime.min.time())

    # -- Daily summary series --
    daily_rows: list[dict[str, Any]] = []
    if os.path.exists(daily_path):
        with open(daily_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                daily_rows.append({
                    "date": row["date"],
                    "requests": int(row["requests"]),
                    "input_tokens": int(row["input_tokens"]),
                    "cached_tokens": int(row["cached_tokens"]),
                    "output_tokens": int(row["output_tokens"]),
                    "total_tokens": int(row["total_tokens"]),
                    "input_cost": float(row["input_cost"]),
                    "cached_cost": float(row["cached_cost"]),
                    "output_cost": float(row["output_cost"]),
                    "total_cost": float(row["total_cost"]),
                })

    # -- Accumulator --
    def _empty() -> dict[str, float | int]:
        return {"requests": 0, "input_tokens": 0, "cached_tokens": 0,
                "output_tokens": 0, "input_cost": 0.0, "cached_cost": 0.0,
                "output_cost": 0.0, "total_cost": 0.0}

    def _add(acc: dict[str, float | int], *, inp: int, cached: int, out: int,
             ic: float, cc: float, oc: float, tc: float) -> None:
        acc["requests"] += 1
        acc["input_tokens"] += inp
        acc["cached_tokens"] += cached
        acc["output_tokens"] += out
        acc["input_cost"] += ic
        acc["cached_cost"] += cc
        acc["output_cost"] += oc
        acc["total_cost"] += tc

    totals_all = _empty()
    totals_today = _empty()
    totals_7d = _empty()
    totals_30d = _empty()

    # Hourly buckets for today
    hourly_req: list[int] = [0] * 24
    hourly_cost: list[float] = [0.0] * 24
    hourly_inp: list[int] = [0] * 24
    hourly_cache: list[int] = [0] * 24
    hourly_out: list[int] = [0] * 24

    recent: list[dict[str, Any]] = []

    with open(requests_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        all_rows: list[list[str]] = []
        for row in reader:
            if len(row) < 11:
                continue
            all_rows.append(row)

    for row in all_rows:
        ts_str = row[0]
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            ts = today_start

        try:
            inp = int(row[2])
            cached = int(row[3])
            out = int(row[4])
            ic = float(row[6])
            cc = float(row[7])
            oc = float(row[8])
            tc = float(row[9])
        except (ValueError, IndexError):
            continue

        _add(totals_all, inp=inp, cached=cached, out=out, ic=ic, cc=cc, oc=oc, tc=tc)
        if ts >= seven_days_ago_start:
            _add(totals_7d, inp=inp, cached=cached, out=out, ic=ic, cc=cc, oc=oc, tc=tc)
        if ts >= thirty_days_ago_start:
            _add(totals_30d, inp=inp, cached=cached, out=out, ic=ic, cc=cc, oc=oc, tc=tc)
        if ts >= today_start:
            _add(totals_today, inp=inp, cached=cached, out=out, ic=ic, cc=cc, oc=oc, tc=tc)
            try:
                h = ts.hour
                hourly_req[h] += 1
                hourly_cost[h] += tc
                hourly_inp[h] += inp
                hourly_cache[h] += cached
                hourly_out[h] += out
            except Exception:
                pass

    for row in all_rows[-30:]:
        recent.append({
            "timestamp": row[0],
            "model": row[1],
            "input_tokens": int(row[2]),
            "cached_tokens": int(row[3]),
            "output_tokens": int(row[4]),
            "total_cost": float(row[9]),
            "status": int(row[10]) if len(row) > 10 else 200,
        })

    hourly_series: list[dict[str, Any]] = []
    for h in range(24):
        if hourly_req[h] > 0:
            hourly_series.append({
                "hour": h,
                "requests": hourly_req[h],
                "total_cost": round(hourly_cost[h], 8),
                "input_tokens": hourly_inp[h],
                "cached_tokens": hourly_cache[h],
                "output_tokens": hourly_out[h],
            })

    def _round_totals(acc: dict[str, float | int]) -> dict[str, Any]:
        acc["total_tokens"] = acc["input_tokens"] + acc["output_tokens"]
        for k in ("input_cost", "cached_cost", "output_cost", "total_cost"):
            acc[k] = round(acc[k], 8)
        return acc

    return {
        "all": _round_totals(totals_all),
        "today": _round_totals(totals_today),
        "last7d": _round_totals(totals_7d),
        "last30d": _round_totals(totals_30d),
        "daily": daily_rows,
        "hourly": hourly_series,
        "recent": recent,
        "pricing": {
            "input": cost_input_price,
            "cached": cost_cached_price,
            "output": cost_output_price,
        },
        "generated_at": datetime.now().isoformat(),
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def render_dashboard_html(data: dict[str, Any]) -> bytes:
    """Generate a self-contained HTML dashboard page from collected data."""
    import html as html_module

    # Prefer the explicit model_name in the provided data (proxy passes the
    # repo/.env value).  Fall back to reading the repo .env file only if the
    # data dict doesn't include a model name — this avoids inconsistencies when
    # the running process has an older cached module.
    model_candidate = data.get("model_name") if isinstance(data.get("model_name"), str) else None
    if not model_candidate:
        try:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        if k.strip() == "MODEL_ALIAS":
                            model_candidate = v.strip().strip('"\'')
                            break
        except Exception:
            model_candidate = None

    model_name = html_module.escape(str(model_candidate or "PAQ_LLAMACPP_SERVER"))

    def fcost(v: float) -> str:
        if v == 0:
            return "$0.00"
        if v < 0.01:
            return f"${v:.4f}"
        return f"${v:.2f}"

    def ftok(v: int) -> str:
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v/1_000:.0f}K"
        return str(v)

    all_ = data.get("all", {})
    today = data.get("today", {})
    last7d = data.get("last7d", {})
    last30d = data.get("last30d", {})
    daily = data.get("daily", [])
    hourly = data.get("hourly", [])
    recent = data.get("recent", [])
    pricing = data.get("pricing", {})

    avg_input = int(all_.get("input_tokens", 0)) // max(1, int(all_.get("requests", 1)))
    avg_output = int(all_.get("output_tokens", 0)) // max(1, int(all_.get("requests", 1)))
    avg_cached = int(all_.get("cached_tokens", 0)) // max(1, int(all_.get("requests", 1)))
    total_input_all = int(all_.get("input_tokens", 0)) + int(all_.get("cached_tokens", 0))
    cache_ratio = (int(all_.get("cached_tokens", 0)) / max(1, total_input_all) * 100)

    ga = data.get("generated_at", "")
    ga_display = ga[:16].replace("T", " ") if len(ga) >= 16 else ga

    # Overview cards
    overview_cards = _overview_cards(all_, today, last7d, last30d, fcost)
    # Token stats
    token_stats = _token_stats(all_, avg_input, avg_cached, avg_output, cache_ratio, ftok, fcost)
    # Pricing
    pricing_section = _pricing_section(pricing)
    # Charts data as embedded JSON
    daily_json = json.dumps(daily, ensure_ascii=True)
    hourly_json = json.dumps(hourly, ensure_ascii=True)
    # Recent requests table
    recent_table = _recent_requests_table(recent, fcost)

    return _build_full_html(model_name, ga_display, overview_cards, token_stats, pricing_section,
                daily_json, hourly_json, recent_table)


def _overview_cards(all_, today, last7d, last30d, fcost):
    return f'''    <div class="cards" id="overview">
      <div class="card total">
        <div class="card-label">Total Cost</div>
        <div class="card-value">{fcost(all_.get("total_cost", 0))}</div>
        <div class="card-sub">{int(all_.get("requests", 0)):,} requests</div>
      </div>
      <div class="card">
        <div class="card-label">Today</div>
        <div class="card-value">{fcost(today.get("total_cost", 0))}</div>
        <div class="card-sub">{int(today.get("requests", 0))} requests</div>
      </div>
      <div class="card">
        <div class="card-label">Last 7 Days</div>
        <div class="card-value">{fcost(last7d.get("total_cost", 0))}</div>
        <div class="card-sub">{int(last7d.get("requests", 0))} requests</div>
      </div>
      <div class="card">
        <div class="card-label">Last 30 Days</div>
        <div class="card-value">{fcost(last30d.get("total_cost", 0))}</div>
        <div class="card-sub">{int(last30d.get("requests", 0))} requests</div>
      </div>
    </div>'''


def _token_stats(all_, avg_input, avg_cached, avg_output, cache_ratio, ftok, fcost):
    inp = int(all_.get("input_tokens", 0))
    cached = int(all_.get("cached_tokens", 0))
    out = int(all_.get("output_tokens", 0))
    ic = all_.get("input_cost", 0)
    cc = all_.get("cached_cost", 0)
    oc = all_.get("output_cost", 0)
    return f'''    <div class="section card">
      <h2>Token Statistics (All-Time)</h2>
      <div class="stats-grid">
        <div class="stat"><div class="stat-label">Total Input Tokens</div><div class="stat-value">{inp:,}</div></div>
        <div class="stat"><div class="stat-label">Total Cached Tokens</div><div class="stat-value">{cached:,}</div></div>
        <div class="stat"><div class="stat-label">Total Output Tokens</div><div class="stat-value">{out:,}</div></div>
        <div class="stat"><div class="stat-label">Avg Input/Request</div><div class="stat-value">{ftok(avg_input)}</div></div>
        <div class="stat"><div class="stat-label">Avg Cached/Request</div><div class="stat-value">{ftok(avg_cached)}</div></div>
        <div class="stat"><div class="stat-label">Avg Output/Request</div><div class="stat-value">{ftok(avg_output)}</div></div>
        <div class="stat"><div class="stat-label">Cache Hit Ratio</div><div class="stat-value">{cache_ratio:.1f}%</div></div>
        <div class="stat"><div class="stat-label">Cost (Input)</div><div class="stat-value">{fcost(ic)}</div></div>
        <div class="stat"><div class="stat-label">Cost (Cached)</div><div class="stat-value">{fcost(cc)}</div></div>
        <div class="stat"><div class="stat-label">Cost (Output)</div><div class="stat-value">{fcost(oc)}</div></div>
      </div>
    </div>'''


def _pricing_section(pricing):
    inp_p = pricing.get("input", 0)
    cache_p = pricing.get("cached", 0)
    out_p = pricing.get("output", 0)
    return f'''    <div class="section card">
      <h2>Current Pricing Settings</h2>
      <div class="stats-grid">
        <div class="stat"><div class="stat-label">Input (fresh)</div><div class="stat-value">${inp_p:.4f}/1M tokens</div></div>
        <div class="stat"><div class="stat-label">Input (cached)</div><div class="stat-value">${cache_p:.4f}/1M tokens</div></div>
        <div class="stat"><div class="stat-label">Output</div><div class="stat-value">${out_p:.4f}/1M tokens</div></div>
      </div>
    </div>'''


def _recent_requests_table(recent, fcost):
    rows_html = ""
    for r in recent:
        ts = r["timestamp"]
        ts_display = ts[:16].replace("T", " ") if len(ts) >= 16 else ts
        rows_html += f'<tr><td>{ts_display}</td><td>{r["model"]}</td><td>{int(r["input_tokens"]):,}</td><td>{int(r["cached_tokens"]):,}</td><td>{int(r["output_tokens"]):,}</td><td>{fcost(r["total_cost"])}</td><td>{r["status"]}</td></tr>\n'
    return f'''    <div class="section card">
      <h2>Recent Requests (Last {len(recent)})</h2>
      <table class="recent-table">
        <thead>
          <tr><th>Time</th><th>Model</th><th>Input</th><th>Cached</th><th>Output</th><th>Cost</th><th>Status</th></tr>
        </thead>
        <tbody>
        {rows_html.rstrip()}
        </tbody>
      </table>
    </div>'''


def _build_full_html(model_name, ga_display, overview_cards, token_stats, pricing_section,
                     daily_json, hourly_json, recent_table) -> bytes:
    # Build the Canvas chart JavaScript
    charts_js = _build_charts_js(daily_json, hourly_json)
    css = _build_css()
    escaped_json_daily = daily_json.replace("\\", "\\\\").replace("'", "\\'")
    escaped_json_hourly = hourly_json.replace("\\", "\\\\").replace("'", "\\'")

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
  <title>""" + model_name + """ — Cost Dashboard</title>
<style>
""" + css + """
</style>
</head>
<body>

  <h1>""" + model_name + """ — Cost Dashboard</h1>
<div class="subtitle">Last updated: """ + ga_display + """ · <a href="/cost-calculator" style="color:var(--accent);text-decoration:none;">⚡ Cost Calculator</a></div>

""" + overview_cards + """
""" + token_stats + """
""" + pricing_section + """

<!-- Daily Cost Chart -->
<div class="chart-container">
  <div class="chart-title">Daily Cost</div>
  <canvas id="dailyCostChart"></canvas>
</div>

<!-- Daily Tokens Chart -->
<div class="chart-container">
  <div class="chart-title">Daily Tokens Breakdown</div>
  <div class="legend">
    <span class="legend-item"><span class="legend-dot" style="background:#58a6ff"></span> Input (fresh)</span>
    <span class="legend-item"><span class="legend-dot" style="background:#3fb950"></span> Cached</span>
    <span class="legend-item"><span class="legend-dot" style="background:#d29922"></span> Output</span>
  </div>
  <canvas id="dailyTokenChart"></canvas>
</div>

<!-- Hourly Activity (Today) -->
<div class="chart-container">
  <div class="chart-title">Today — Requests &amp; Cost by Hour</div>
  <canvas id="hourlyChart" class="hourly-chart"></canvas>
</div>

""" + recent_table + """

<script>
var dailyData = """ + escaped_json_daily + """;
var hourlyData = """ + escaped_json_hourly + """;

""" + charts_js + """
</script>

</body>
</html>"""
    return html_template.encode("utf-8")


def _build_css() -> str:
    return """:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
  --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --orange: #d29922; --red: #f85149;
  --border: #30363d;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text); padding: 20px;
  max-width: 1400px; margin: 0 auto;
}
h1 { font-size: 1.6rem; margin-bottom: 6px; }
h2 { font-size: 1.1rem; margin-bottom: 12px; color: var(--text2); }
.subtitle { color: var(--text2); font-size: 0.85rem; margin-bottom: 20px; }
.cards {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; margin-bottom: 20px;
}
.card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px;
}
.card-label { font-size: 0.75rem; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }
.card-value { font-size: 1.8rem; font-weight: 700; margin: 4px 0; }
.card-sub { font-size: 0.8rem; color: var(--text2); }
.card.total .card-value { color: var(--accent); }
.section { margin-bottom: 20px; }
.stats-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.stat-label { font-size: 0.75rem; color: var(--text2); }
.stat-value { font-size: 1.1rem; font-weight: 600; }
.section.card { padding: 20px; }
.chart-container {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; margin-bottom: 20px;
}
.chart-title { font-size: 0.9rem; color: var(--text2); margin-bottom: 10px; }
canvas { width: 100%; height: 280px; display: block; }
canvas.hourly-chart { height: 200px; }
.recent-table {
  width: 100%; border-collapse: collapse; font-size: 0.8rem;
}
.recent-table th {
  text-align: left; padding: 8px 6px; border-bottom: 2px solid var(--border);
  color: var(--text2); font-weight: 600; white-space: nowrap;
}
.recent-table td {
  padding: 6px; border-bottom: 1px solid var(--border); white-space: nowrap;
}
.recent-table tr:hover { background: var(--bg3); }
.legend {
  display: flex; gap: 16px; font-size: 0.75rem; color: var(--text2);
  margin-bottom: 8px;
}
.legend-item { display: flex; align-items: center; gap: 4px; }
.legend-dot {
  width: 10px; height: 10px; border-radius: 2px; display: inline-block;
}
@media (max-width: 600px) {
  .cards { grid-template-columns: repeat(2, 1fr); }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
}"""


def _build_charts_js(daily_json: str, hourly_json: str) -> str:
    return """
// --- Utility ---
function fmtCost(v) { return v < 0.01 ? "$"+v.toFixed(4) : "$"+v.toFixed(2); }
function fmtTok(v) { if(v>=1e6) return (v/1e6).toFixed(1)+"M"; if(v>=1e3) return (v/1e3).toFixed(0)+"K"; return v; }
function setDPI(cv, w, h) { var d=window.devicePixelRatio||1; cv.width=w*d; cv.height=h*d; cv.style.width=w+"px"; cv.style.height=h+"px"; return cv.getContext("2d"); }

// --- Daily Cost Chart ---
(function(){
  var c = document.getElementById("dailyCostChart");
  if(!c) return;
  var rect = c.getBoundingClientRect();
  var ctx = setDPI(c, rect.width||800, 280);
  var data = dailyData.slice(-60);
  if(!data.length) return;
  var pad = {t:10,r:15,b:40,l:60};
  var W = c.width/window.devicePixelRatio; var H = c.height/window.devicePixelRatio;
  var cw = W-pad.l-pad.r; var ch = H-pad.t-pad.b;
  var maxV = Math.max.apply(null, data.map(function(d){return d.total_cost;}));
  if(maxV < 0.001) maxV = 0.001;
  var barW = Math.max(1, (cw/data.length)-1);

  ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
  for(var i=0;i<=4;i++){
    var y = pad.t + ch*(1-i/4);
    ctx.fillText(fmtCost(maxV*i/4), pad.l-5, y+3);
    ctx.strokeStyle = "#21262d"; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
  }
  data.forEach(function(d,idx){
    var x = pad.l + (cw/data.length)*idx + (cw/data.length - barW)/2;
    var bh = (d.total_cost/maxV)*ch;
    var y = pad.t + ch - bh;
    ctx.fillStyle = "#58a6ff";
    ctx.fillRect(x, y, barW, bh);
  });
  ctx.fillStyle = "#8b949e"; ctx.textAlign = "center"; ctx.font = "9px sans-serif";
  var step = Math.max(1, Math.floor(data.length/10));
  data.forEach(function(d,idx){
    if(idx%step===0||idx===data.length-1){
      var x = pad.l + (cw/data.length)*idx + (cw/data.length)/2;
      ctx.fillText(d.date.slice(5), x, H-pad.b+14);
    }
  });
})();

// --- Daily Tokens Stacked Chart ---
(function(){
  var c = document.getElementById("dailyTokenChart");
  if(!c) return;
  var rect = c.getBoundingClientRect();
  var ctx = setDPI(c, rect.width||800, 280);
  var data = dailyData.slice(-60);
  if(!data.length) return;
  var pad = {t:10,r:15,b:40,l:70};
  var W = c.width/window.devicePixelRatio; var H = c.height/window.devicePixelRatio;
  var cw = W-pad.l-pad.r; var ch = H-pad.t-pad.b;
  var freshArr = data.map(function(d){return Math.max(0, d.input_tokens - d.cached_tokens);});
  var maxV = Math.max.apply(null, data.map(function(d,i){return freshArr[i]+d.cached_tokens+d.output_tokens;}));
  if(maxV < 1) maxV = 1;
  var barW = Math.max(1, (cw/data.length)-1);

  ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
  for(var i=0;i<=4;i++){
    var y = pad.t + ch*(1-i/4);
    ctx.fillText(fmtTok(maxV*i/4), pad.l-5, y+3);
    ctx.strokeStyle="#21262d"; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
  }
  var colors = ["#58a6ff","#3fb950","#d29922"];
  data.forEach(function(d,idx){
    var x = pad.l + (cw/data.length)*idx;
    var yOff = pad.t + ch;
    var layers = [freshArr[idx], d.cached_tokens, d.output_tokens];
    for(var li=0;li<layers.length;li++){
      var bh = (layers[li]/maxV)*ch;
      yOff -= bh;
      ctx.fillStyle = colors[li];
      ctx.fillRect(x+1, yOff, barW-2, bh);
    }
  });
  ctx.fillStyle = "#8b949e"; ctx.textAlign = "center"; ctx.font = "9px sans-serif";
  var step = Math.max(1, Math.floor(data.length/10));
  data.forEach(function(d,idx){
    if(idx%step===0||idx===data.length-1){
      var x = pad.l + (cw/data.length)*idx + (cw/data.length)/2;
      ctx.fillText(d.date.slice(5), x, H-pad.b+14);
    }
  });
})();

// --- Hourly Chart ---
(function(){
  var c = document.getElementById("hourlyChart");
  if(!c) return;
  var rect = c.getBoundingClientRect();
  var ctx = setDPI(c, rect.width||800, 200);
  if(!hourlyData.length) return;
  var pad = {t:10,r:15,b:30,l:60};
  var W = c.width/window.devicePixelRatio; var H = c.height/window.devicePixelRatio;
  var cw = W-pad.l-pad.r; var ch = H-pad.t-pad.b;
  var maxR = Math.max.apply(null, hourlyData.map(function(d){return d.requests;}));
  if(maxR < 1) maxR = 1;
  var colW = cw/24;

  ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
  for(var i=0;i<=3;i++){
    var y = pad.t + ch*(1-i/3);
    ctx.fillText(Math.round(maxR*i/3)+" req", pad.l-5, y+3);
    ctx.strokeStyle="#21262d"; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
  }
  hourlyData.forEach(function(d){
    var x = pad.l + d.hour * colW;
    var bh = (d.requests/maxR)*ch;
    ctx.fillStyle = "#3fb950";
    ctx.fillRect(x+1, pad.t+ch-bh, (colW-2), bh);
  });
  ctx.fillStyle = "#8b949e"; ctx.textAlign = "center"; ctx.font = "9px sans-serif";
  for(var h=0;h<24;h+=2){
    ctx.fillText(h+":00", pad.l + h*colW + colW/2, H-pad.b+14);
  }
  ctx.fillStyle = "#d29922"; ctx.font = "8px sans-serif";
  hourlyData.forEach(function(d){
    var x = pad.l + d.hour * colW + colW/2;
    var bh = (d.requests/maxR)*ch;
    var y = pad.t + ch - bh - 4;
    ctx.fillText(fmtCost(d.total_cost), x, y);
  });
})();
"""
