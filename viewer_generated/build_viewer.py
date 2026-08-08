"""
build_viewer.py  —  GENERATED HELPER (not hand-written by the project author)

Reads every negotiation JSON in ../runs/ and writes a single self-contained
HTML file (negotiation_viewer.html) you can open in a browser: batch stats,
a per-negotiation price-trajectory chart against the hidden ZOPA band, the
full turn-by-turn transcript including each agent's private reasoning, and
a fair-value read-out comparing each agent's stated estimate to the hidden
ground truth (plus whether that estimate drifted over the negotiation).

Run from inside this folder:   python build_viewer.py
Then open:                     negotiation_viewer.html

It writes nothing except that one HTML file, and only reads ../runs/.
"""

import os
import json
import html
import re

# --- Locate the runs/ folder relative to THIS script, not the cwd ---
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "..", "runs")
OUT_FILE = os.path.join(HERE, "negotiation_viewer.html")


def load_runs():
    """Read every .json file in ../runs/ into a list of records, tagging each
    with its source filename so the viewer can label them."""
    records = []
    if not os.path.isdir(RUNS_DIR):
        return records
    for name in sorted(os.listdir(RUNS_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(RUNS_DIR, name)
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            record["_filename"] = name
            records.append(record)
        except (json.JSONDecodeError, OSError):
            # Skip anything unreadable rather than crashing the whole build.
            continue
    return records


def seller_share_of_zopa(record):
    """For a deal, what fraction of the bargaining zone the seller captured.
    0.5 = split evenly, 1.0 = seller took everything, 0.0 = buyer took everything.
    Returns None for non-deals."""
    if record.get("outcome") != "deal" or record.get("final_price") is None:
        return None
    sc = record["scenario"]
    zopa = sc["buyer_reservation"] - sc["seller_reservation"]
    if zopa == 0:
        return None
    return (record["final_price"] - sc["seller_reservation"]) / zopa


def extract_dollar_number(text):
    """Pull the car's estimated fair-value price out of free text like
    'I estimate fair market value at approximately $13,200' -> 13200.0.

    Requires a literal '$' before a number to count as a price candidate at
    all — bare numbers in these responses are usually the model year ("this
    2018 Civic") or mileage ("79,670 miles"), NOT a price, and grabbing the
    first number >= 1000 with no '$' check used to misfire on the year.

    Among the '$'-prefixed numbers found (there may be several, since comps
    like "$13,197" and "$13,481" are often cited before the actual estimate),
    prefer one immediately following a keyword like "estimate" or "value" —
    that's the model's actual conclusion, not a cited comparable sale. If no
    keyword match is found, fall back to the LAST '$' figure in the text,
    since comps are typically listed before the summary judgment.

    Returns None if no '$'-prefixed number is found at all."""
    if not text:
        return None

    def to_float(s):
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return None

    dollar_matches = re.findall(r'\$\s?([\d]{1,3}(?:,\d{3})*(?:\.\d+)?)', text)
    dollar_values = [v for v in (to_float(m) for m in dollar_matches) if v is not None and v >= 1000]

    if dollar_values:
        keyword_matches = re.findall(
            r'(?:estimate|fair (?:market )?value|worth|valu(?:e|ed) at)[^.$]{0,40}\$\s?([\d]{1,3}(?:,\d{3})*(?:\.\d+)?)',
            text, re.IGNORECASE,
        )
        keyword_values = [v for v in (to_float(m) for m in keyword_matches) if v is not None and v >= 1000]
        if keyword_values:
            return keyword_values[-1]
        return dollar_values[-1]

    return None


def compute_fv_summary(turns, fair_value, speaker):
    """For one side of one negotiation: pull every fair_value_estimate this
    speaker gave, in turn order, parse the dollar figure out of each, and
    summarize: first estimate, whether later estimates ever differ from the
    first (drift), and how far that first estimate was from ground truth."""
    estimates = []
    for t in turns:
        if t.get("speaker") != speaker:
            continue
        num = extract_dollar_number(t.get("fair_value_estimate"))
        if num is not None:
            estimates.append(num)

    if not estimates:
        return {"first": None, "changed": False, "diff": None, "pct_off": None, "count": 0}

    first = estimates[0]
    changed = any(e != first for e in estimates[1:])
    diff = (first - fair_value) if fair_value else None
    pct_off = (100.0 * diff / fair_value) if (diff is not None and fair_value) else None
    return {
        "first": first,
        "changed": changed,
        "diff": diff,
        "pct_off": pct_off,
        "count": len(estimates),
    }


def compute_stats(records):
    outcomes = {}
    shares = []
    deal_turns = []
    buyer_drift_count = 0
    seller_drift_count = 0
    fv_tracked = 0
    for r in records:
        outcomes[r.get("outcome", "unknown")] = outcomes.get(r.get("outcome", "unknown"), 0) + 1
        s = seller_share_of_zopa(r)
        if s is not None:
            shares.append(s)
            if isinstance(r.get("num_turns"), int):
                deal_turns.append(r["num_turns"])

        sc = r.get("scenario", {})
        fv = sc.get("fair_value")
        turns = r.get("turns", [])
        buyer_fv = compute_fv_summary(turns, fv, "buyer")
        seller_fv = compute_fv_summary(turns, fv, "seller")
        if buyer_fv["count"] or seller_fv["count"]:
            fv_tracked += 1
        if buyer_fv["changed"]:
            buyer_drift_count += 1
        if seller_fv["changed"]:
            seller_drift_count += 1

    avg_share = sum(shares) / len(shares) if shares else None
    avg_turns = sum(deal_turns) / len(deal_turns) if deal_turns else None
    return {
        "total": len(records),
        "outcomes": outcomes,
        "deals": len(shares),
        "avg_seller_share": avg_share,
        "avg_deal_turns": avg_turns,
        "shares": shares,
        "fv_tracked": fv_tracked,
        "buyer_drift_count": buyer_drift_count,
        "seller_drift_count": seller_drift_count,
    }


# ---------------------------------------------------------------------------
# The design: a "convergence" view. Each negotiation's numeric offers are
# plotted as two lines (buyer rising, seller falling) over an invisible-until-
# revealed ZOPA band. The band is ground truth the agents never saw, so showing
# it is the whole analytical payoff — you can SEE whether they landed high or
# low in the zone. This is the signature element.
# ---------------------------------------------------------------------------

def esc(x):
    return html.escape(str(x), quote=True)


def build_html(records, stats):
    # Pre-serialize the data the client-side JS needs. We keep it compact:
    # only what the charts/transcripts render.
    payload = []
    for i, r in enumerate(records):
        sc = r["scenario"]
        turns_raw = r.get("turns", [])
        fair_value = sc.get("fair_value")

        buyer_fv = compute_fv_summary(turns_raw, fair_value, "buyer")
        seller_fv = compute_fv_summary(turns_raw, fair_value, "seller")

        payload.append({
            "id": i,
            "filename": r.get("_filename", f"run {i}"),
            "outcome": r.get("outcome"),
            "final_price": r.get("final_price"),
            "accepted_by": r.get("accepted_by"),
            "num_turns": r.get("num_turns"),
            "fair_value": fair_value,
            "seller_reservation": sc.get("seller_reservation"),
            "buyer_reservation": sc.get("buyer_reservation"),
            "asking_price": sc.get("asking_price"),
            "seller_share": seller_share_of_zopa(r),
            "car": sc.get("car_facts", {}),
            "fv_summary": {"buyer": buyer_fv, "seller": seller_fv},
            "turns": [
                {
                    "n": t.get("turn_number"),
                    "speaker": t.get("speaker"),
                    "action": t.get("action"),
                    "number": t.get("number"),
                    # normalized so trajectories from different-priced scenarios
                    # can be overlaid and compared on one axis
                    "pct_fair": (round(100.0 * t["number"] / sc["fair_value"], 2)
                                 if isinstance(t.get("number"), (int, float)) and sc.get("fair_value")
                                 else None),
                    "message": t.get("message"),
                    "thoughts": t.get("thoughts_on_counterpart"),
                    "reasoning": t.get("reasoning"),
                    "fair_value_estimate": t.get("fair_value_estimate"),
                    "fv_number": extract_dollar_number(t.get("fair_value_estimate")),
                }
                for t in turns_raw
            ],
        })

    data_json = json.dumps(payload)
    stats_json = json.dumps(stats)

    # NOTE: the CSS/JS below is intentionally one self-contained file so a
    # screenshot-friendly artifact travels as a single .html with no assets.
    return TEMPLATE.replace("/*__DATA__*/", data_json).replace("/*__STATS__*/", stats_json)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Haggle — negotiation viewer</title>
<style>
  :root {
    --ink: #14210f;
    --paper: #f3f0e7;
    --panel: #fbfaf5;
    --line: #d8d2c2;
    --buyer: #2f6f4f;
    --seller: #b4451f;
    --zopa: #e7c65a;
    --muted: #6b6656;
    --deal: #2f6f4f;
    --nodeal: #9a3b2a;
    --warn: #a3781e;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--paper);
    color: var(--ink);
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 40px 24px 120px; }

  header.masthead {
    border-bottom: 3px solid var(--ink);
    padding-bottom: 14px;
    margin-bottom: 6px;
  }
  .kicker {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    font-size: 11px;
    color: var(--seller);
    margin: 0 0 8px;
  }
  h1 {
    font-size: clamp(38px, 7vw, 68px);
    line-height: 0.98;
    margin: 0;
    font-weight: 800;
    letter-spacing: -0.01em;
  }
  .sub { color: var(--muted); margin: 10px 0 0; font-size: 15px; max-width: 60ch; }

  .stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1px;
    background: var(--line);
    border: 1px solid var(--line);
    margin: 34px 0 10px;
  }
  .stat { background: var(--panel); padding: 16px 18px; }
  .stat .label {
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 6px;
  }
  .stat .value { font-size: 30px; font-weight: 800; letter-spacing: -0.02em; }
  .stat .value small { font-size: 14px; font-weight: 600; color: var(--muted); }

  .share-meter {
    margin: 18px 0 8px; border: 1px solid var(--line); background: var(--panel);
    padding: 16px 18px;
  }
  .share-meter .label {
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 10px; display:flex; justify-content:space-between;
  }
  .track { position: relative; height: 26px; background:
      linear-gradient(90deg, var(--buyer) 0%, #cbd9cd 50%, var(--seller) 100%);
      border-radius: 2px; opacity: 0.9; }
  .track .mid { position:absolute; top:-4px; bottom:-4px; left:50%; width:2px; background:var(--ink); }
  .track .needle { position:absolute; top:-6px; bottom:-6px; width:3px; background:var(--ink);
      box-shadow: 0 0 0 2px var(--panel); }
  .track .caption { position:absolute; top:30px; font-size:11px; color:var(--muted);
      font-family: ui-monospace, Menlo, monospace; transform: translateX(-50%); white-space:nowrap; }
  .endlabels { display:flex; justify-content:space-between; font-size:11px; color:var(--muted);
      font-family: ui-monospace, Menlo, monospace; margin-top:26px; }

  h2.section {
    font-size: 13px; font-family: ui-monospace, Menlo, Consolas, monospace;
    text-transform: uppercase; letter-spacing: 0.18em; color: var(--muted);
    border-bottom: 1px solid var(--line); padding-bottom: 8px; margin: 48px 0 0;
  }

  .neg {
    border: 1px solid var(--line);
    border-top: none;
    background: var(--panel);
  }
  .neg:first-of-type { border-top: 1px solid var(--line); }
  .neg > summary {
    list-style: none; cursor: pointer; padding: 18px 20px;
    display: grid; grid-template-columns: 44px 1fr auto; gap: 16px; align-items: center;
  }
  .neg > summary::-webkit-details-marker { display: none; }
  .neg[open] > summary { border-bottom: 1px solid var(--line); background: var(--panel); }
  .idx {
    font-family: ui-monospace, Menlo, monospace; font-size: 22px; font-weight: 700;
    color: var(--muted);
  }
  .neg-title { font-weight: 700; font-size: 17px; }
  .neg-meta { font-size: 12.5px; color: var(--muted); margin-top: 2px;
    font-family: ui-monospace, Menlo, monospace; }
  .pill {
    font-family: ui-monospace, Menlo, monospace; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; padding: 5px 10px; border-radius: 2px;
    white-space: nowrap;
  }
  .pill.deal { background: rgba(47,111,79,0.14); color: var(--deal); }
  .pill.nodeal { background: rgba(154,59,42,0.13); color: var(--nodeal); }
  .pill.drift { background: rgba(163,120,30,0.16); color: var(--warn); }

  .neg-body { padding: 8px 20px 24px; }

  .chart-wrap { margin: 12px 0 6px; }
  svg.chart { width: 100%; height: auto; display: block; }
  .legend { display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:var(--muted);
     font-family: ui-monospace, Menlo, monospace; margin: 6px 2px 0; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .swatch { width:14px; height:3px; display:inline-block; }
  .swatch.band { height:12px; width:12px; opacity:0.5; }

  .fv-panel {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1px; background: var(--line); border: 1px solid var(--line); margin: 14px 0 4px;
  }
  .fv-card { background: var(--panel); padding: 14px 16px; }
  .fv-card .who {
    font-family: ui-monospace, Menlo, monospace; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 700; margin-bottom: 8px; display:flex; align-items:center; gap:8px;
  }
  .fv-card.buyer .who { color: var(--buyer); }
  .fv-card.seller .who { color: var(--seller); }
  .fv-card .row { display:flex; justify-content:space-between; font-size:13px; margin:4px 0; }
  .fv-card .row .k { color: var(--muted); }
  .fv-card .row .v { font-weight: 700; font-family: ui-monospace, Menlo, monospace; }
  .fv-card .row .v.off { color: var(--warn); }
  .drift-badge {
    font-family: ui-monospace, Menlo, monospace; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em; padding: 2px 7px; border-radius: 2px;
    background: rgba(163,120,30,0.16); color: var(--warn);
  }
  .nodrift-badge {
    font-family: ui-monospace, Menlo, monospace; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em; padding: 2px 7px; border-radius: 2px;
    background: rgba(47,111,79,0.12); color: var(--deal);
  }

  .turns { margin-top: 18px; display: flex; flex-direction: column; gap: 10px; }
  .turn { border-left: 3px solid var(--line); padding: 2px 0 2px 14px; }
  .turn.buyer { border-color: var(--buyer); }
  .turn.seller { border-color: var(--seller); }
  .turn .who {
    font-family: ui-monospace, Menlo, monospace; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.12em; font-weight: 700; margin-bottom: 3px;
  }
  .turn.buyer .who { color: var(--buyer); }
  .turn.seller .who { color: var(--seller); }
  .turn .who .act { color: var(--muted); font-weight: 500; }
  .turn .msg { font-size: 15px; }
  .turn details { margin-top: 6px; }
  .turn details summary {
    cursor: pointer; font-family: ui-monospace, Menlo, monospace; font-size: 11px;
    color: var(--muted); letter-spacing: 0.06em;
  }
  .turn .private {
    margin-top: 6px; padding: 10px 12px; background: var(--paper);
    border: 1px dashed var(--line); font-size: 13.5px; color: #40402f;
  }
  .turn .private .lbl {
    font-family: ui-monospace, Menlo, monospace; font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.14em; color: var(--muted); margin: 8px 0 3px;
  }
  .turn .private .lbl:first-child { margin-top: 0; }

  .empty { padding: 40px; text-align: center; color: var(--muted);
     border: 1px dashed var(--line); background: var(--panel); margin-top: 24px; }

  footer { margin-top: 60px; color: var(--muted); font-size: 12px;
     font-family: ui-monospace, Menlo, monospace; border-top: 1px solid var(--line); padding-top: 14px; }

  @media (max-width: 560px) {
    .neg > summary { grid-template-columns: 32px 1fr; }
    .neg > summary .pill { grid-column: 2; justify-self: start; margin-top: 6px; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header class="masthead">
    <p class="kicker">AI negotiation harness &middot; batch readout</p>
    <h1>Haggle</h1>
    <p class="sub">Two language-model agents &mdash; a buyer and a seller &mdash; bargain over a used 2018 Civic.
       Each chart plots their offers converging against the <em>hidden</em> zone of possible agreement
       neither agent could see. Where they land inside that band is who won.</p>
  </header>

  <div class="stats" id="stats"></div>
  <div class="share-meter" id="shareMeter"></div>

  <h2 class="section">All negotiations, normalized</h2>
  <p class="sub" style="margin:14px 0 0">Every offer as a percentage of that scenario's hidden fair value, so
     negotiations with different dollar values can be compared on one axis. The band around 100% is the
     average bargaining zone. Deals are marked where they closed.</p>
  <div id="aggregate"></div>

  <h2 class="section">Negotiations</h2>
  <div id="list"></div>

  <footer>Generated viewer &middot; reads runs/*.json &middot; one self-contained HTML file.</footer>
</div>

<script>
const DATA = /*__DATA__*/;
const STATS = /*__STATS__*/;

const money = n => (n === null || n === undefined) ? "\u2014" : "$" + Number(n).toLocaleString(undefined, {maximumFractionDigits:0});
const esc = s => (s === null || s === undefined) ? "" :
  String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

(function renderStats(){
  const el = document.getElementById('stats');
  const o = STATS.outcomes || {};
  const cells = [];
  cells.push(`<div class="stat"><div class="label">Negotiations</div><div class="value">${STATS.total}</div></div>`);
  cells.push(`<div class="stat"><div class="label">Deals closed</div><div class="value">${STATS.deals}<small> / ${STATS.total}</small></div></div>`);
  const share = STATS.avg_seller_share;
  cells.push(`<div class="stat"><div class="label">Avg seller share</div><div class="value">${share===null?'\u2014':(share*100).toFixed(0)+'%'}</div></div>`);
  const at = STATS.avg_deal_turns;
  cells.push(`<div class="stat"><div class="label">Avg turns to deal</div><div class="value">${at===null?'\u2014':at.toFixed(1)}</div></div>`);
  cells.push(`<div class="stat"><div class="label">Buyer FV drifted</div><div class="value">${STATS.buyer_drift_count}<small> / ${STATS.fv_tracked}</small></div></div>`);
  cells.push(`<div class="stat"><div class="label">Seller FV drifted</div><div class="value">${STATS.seller_drift_count}<small> / ${STATS.fv_tracked}</small></div></div>`);
  el.innerHTML = cells.join('');
})();

(function renderMeter(){
  const el = document.getElementById('shareMeter');
  const share = STATS.avg_seller_share;
  if (share === null) { el.style.display = 'none'; return; }
  const pct = Math.max(0, Math.min(1, share)) * 100;
  el.innerHTML =
    `<div class="label"><span>Who captured the bargaining zone</span><span>${(share*100).toFixed(1)}% seller</span></div>
     <div class="track">
        <div class="mid"></div>
        <div class="needle" style="left:${pct}%"></div>
        <div class="caption" style="left:${pct}%">avg landing</div>
     </div>
     <div class="endlabels"><span>&#9664; buyer captures all</span><span>even split</span><span>seller captures all &#9654;</span></div>`;
})();

const tip = document.createElement('div');
tip.id = 'tip';
tip.style.cssText = 'position:fixed;z-index:50;display:none;background:var(--ink);color:var(--paper);'
  + 'font-family:ui-monospace,Menlo,monospace;font-size:12px;padding:8px 10px;border-radius:3px;'
  + 'pointer-events:none;max-width:240px;line-height:1.4;box-shadow:0 4px 14px rgba(0,0,0,0.25)';
document.body.appendChild(tip);
function showTip(evt, htmlStr){
  tip.innerHTML = htmlStr;
  tip.style.display = 'block';
  const x = evt.clientX, y = evt.clientY;
  tip.style.left = Math.min(x + 14, window.innerWidth - 250) + 'px';
  tip.style.top = (y + 14) + 'px';
}
function hideTip(){ tip.style.display = 'none'; }
document.addEventListener('click', e => { if (!e.target.closest('[data-tip]')) hideTip(); });

function aggregateChart(){
  const W = 900, H = 360, padL = 56, padR = 24, padT = 20, padB = 64;

  const speakerAt = n => (n % 2 === 0) ? 'buyer' : 'seller';

  const byTurn = {};
  const discussAt = {};
  let maxTurn = 0;
  DATA.forEach(neg => {
    neg.turns.forEach(t => {
      maxTurn = Math.max(maxTurn, t.n);
      if (t.pct_fair !== null) {
        (byTurn[t.n] = byTurn[t.n] || []).push(t.pct_fair);
      } else if (t.action === 'discuss') {
        discussAt[t.n] = (discussAt[t.n] || 0) + 1;
      }
    });
  });
  if (maxTurn === 0 && !Object.keys(byTurn).length)
    return '<div class="empty">No negotiations to summarize yet.</div>';

  const totalNegs = DATA.length;
  const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

  const series = { buyer: [], seller: [] };
  for (let n = 0; n <= maxTurn; n++){
    if (!byTurn[n] || !byTurn[n].length) continue;
    const who = speakerAt(n);
    series[who].push({ n, mean: avg(byTurn[n]), count: byTurn[n].length,
                       lo: Math.min(...byTurn[n]), hi: Math.max(...byTurn[n]) });
  }

  const allMeans = [...series.buyer, ...series.seller];
  const vals = allMeans.flatMap(p=>[p.lo,p.hi]).concat([100]);
  let ymin = Math.floor(Math.min(...vals)/5)*5 - 2;
  let ymax = Math.ceil(Math.max(...vals)/5)*5 + 2;

  const X = n => padL + (maxTurn ? n/maxTurn : 0.5) * (W - padL - padR);
  const Y = v => padT + (1 - (v - ymin)/(ymax - ymin)) * (H - padT - padB);

  let grid = '';
  for (let v = Math.ceil(ymin/5)*5; v <= ymax; v += 5){
    grid += `<line x1="${padL}" x2="${W-padR}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="var(--line)"/>
             <text x="${padL-8}" y="${Y(v).toFixed(1)}" dy="4" text-anchor="end" fill="var(--muted)" font-size="11" font-family="ui-monospace,Menlo,monospace">${v}%</text>`;
  }

  let brs=[], srs=[];
  DATA.forEach(n=>{ if(n.fair_value){brs.push(100*n.buyer_reservation/n.fair_value); srs.push(100*n.seller_reservation/n.fair_value);} });
  const band = brs.length ? `<rect x="${padL}" y="${Y(avg(brs)).toFixed(1)}" width="${(W-padL-padR).toFixed(1)}"
       height="${Math.abs(Y(avg(srs))-Y(avg(brs))).toFixed(1)}" fill="var(--zopa)" opacity="0.22"/>` : '';
  const fairLine = `<line x1="${padL}" x2="${W-padR}" y1="${Y(100).toFixed(1)}" y2="${Y(100).toFixed(1)}"
       stroke="#8a6d17" stroke-width="1.5" stroke-dasharray="3 3"/>
     <text x="${W-padR}" y="${Y(100).toFixed(1)}" dy="-5" text-anchor="end" fill="#8a6d17" font-size="11"
       font-family="ui-monospace,Menlo,monospace">fair value (100%)</text>`;

  function draw(pts, color){
    if (!pts.length) return '';
    const ribbon = 'M' + pts.map(p=>`${X(p.n).toFixed(1)},${Y(p.hi).toFixed(1)}`).join(' L')
      + ' L' + pts.slice().reverse().map(p=>`${X(p.n).toFixed(1)},${Y(p.lo).toFixed(1)}`).join(' L') + ' Z';
    const meanLine = pts.map((p,i)=>`${i?'L':'M'}${X(p.n).toFixed(1)},${Y(p.mean).toFixed(1)}`).join(' ');
    const dots = pts.map(p=>{
      const opacity = 0.35 + 0.65*(p.count/totalNegs);
      const tipHtml = `<b>${speakerAt(p.n)} &middot; turn ${p.n}</b><br>avg ${p.mean.toFixed(1)}% of fair`
        + `<br>range ${p.lo.toFixed(1)}\u2013${p.hi.toFixed(1)}%<br>${p.count} of ${totalNegs} negotiations`;
      return `<circle cx="${X(p.n).toFixed(1)}" cy="${Y(p.mean).toFixed(1)}" r="5" fill="${color}"
        opacity="${opacity.toFixed(2)}" style="cursor:pointer" data-tip="1"
        onclick='showTip(event, ${JSON.stringify(tipHtml)})'></circle>`;
    }).join('');
    return `<path d="${ribbon}" fill="${color}" opacity="0.1"/>
            <path d="${meanLine}" fill="none" stroke="${color}" stroke-width="2.5"/>${dots}`;
  }

  let axis = '';
  for (let n = 0; n <= maxTurn; n++){
    const who = speakerAt(n);
    const c = who === 'buyer' ? 'var(--buyer)' : 'var(--seller)';
    axis += `<circle cx="${X(n).toFixed(1)}" cy="${H-padB+22}" r="4" fill="${c}"/>`;
    if (n % 2 === 0 || n === maxTurn)
      axis += `<text x="${X(n).toFixed(1)}" y="${H-padB+40}" text-anchor="middle" fill="var(--muted)" font-size="10" font-family="ui-monospace,Menlo,monospace">${n}</text>`;
    if (discussAt[n])
      axis += `<text x="${X(n).toFixed(1)}" y="${H-padB+8}" text-anchor="middle" fill="var(--muted)" font-size="9" font-family="ui-monospace,Menlo,monospace" opacity="0.8">talk</text>`;
  }
  axis += `<text x="${padL}" y="${H-6}" fill="var(--muted)" font-size="11" font-family="ui-monospace,Menlo,monospace">turn (buyer opens, then alternating)</text>`;

  return `<div class="chart-wrap">
    <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Average negotiation trajectory">
      ${grid}${band}${fairLine}
      ${draw(series.seller,'var(--seller)')}
      ${draw(series.buyer,'var(--buyer)')}
      ${axis}
    </svg></div>
    <div class="legend">
      <span><span class="swatch" style="background:var(--buyer)"></span>avg buyer offer</span>
      <span><span class="swatch" style="background:var(--seller)"></span>avg seller offer</span>
      <span><span class="swatch band" style="background:var(--zopa)"></span>avg bargaining zone</span>
      <span style="opacity:0.7">shaded = full range &middot; dots fade where fewer negotiations reach that turn</span>
    </div>`;
}

function trajectoryChart(neg){
  const pts = neg.turns.filter(t => typeof t.number === 'number')
                       .map(t => ({n:t.n, num:t.number, who:t.speaker}));
  const W = 900, H = 300, padL = 70, padR = 130, padT = 24, padB = 34;

  const ys = [neg.seller_reservation, neg.buyer_reservation, neg.fair_value,
              neg.asking_price, ...pts.map(p=>p.num)].filter(v => typeof v === 'number');
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  const padY = (ymax - ymin) * 0.08 || 500;
  ymin -= padY; ymax += padY;

  const xs = pts.map(p=>p.n);
  const xmin = pts.length ? Math.min(...xs) : 0;
  const xmax = pts.length ? Math.max(...xs) : 1;
  const X = n => padL + (xmax===xmin ? 0.5 : (n - xmin)/(xmax - xmin)) * (W - padL - padR);
  const Y = v => padT + (1 - (v - ymin)/(ymax - ymin)) * (H - padT - padB);

  const bandTop = Y(neg.buyer_reservation), bandBot = Y(neg.seller_reservation);

  const buyerPts = pts.filter(p=>p.who==='buyer');
  const sellerPts = pts.filter(p=>p.who==='seller');
  const line = arr => arr.map((p,i)=>`${i?'L':'M'}${X(p.n).toFixed(1)},${Y(p.num).toFixed(1)}`).join(' ');
  const fv = neg.fair_value;
  const dots = (arr,color) => arr.map(p=>{
    const pct = fv ? (100*p.num/fv) : null;
    const diff = fv ? (p.num - fv) : null;
    const sign = diff > 0 ? '+' : '';
    const tipHtml = `<b>${p.who} &middot; turn ${p.n}</b><br>${money(p.num)}`
      + (fv ? `<br>${pct.toFixed(1)}% of fair value<br>${sign}${money(diff)} vs fair` : '');
    return `<circle cx="${X(p.n).toFixed(1)}" cy="${Y(p.num).toFixed(1)}" r="6" fill="${color}"
      style="cursor:pointer" data-tip="1"
      onclick='showTip(event, ${JSON.stringify(tipHtml)})'></circle>`;
  }).join('');

  const refLine = (v,label,color,dash) => {
    if (typeof v !== 'number') return '';
    const y = Y(v).toFixed(1);
    return `<line x1="${padL}" x2="${W-padR}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="1.2" stroke-dasharray="${dash}" opacity="0.8"/>
            <text x="${W-padR+8}" y="${y}" dy="4" fill="${color}" font-size="12" font-family="ui-monospace,Menlo,monospace">${label} ${money(v)}</text>`;
  };

  const final = (neg.outcome === 'deal' && typeof neg.final_price === 'number')
    ? `<line x1="${padL}" x2="${W-padR}" y1="${Y(neg.final_price).toFixed(1)}" y2="${Y(neg.final_price).toFixed(1)}" stroke="var(--ink)" stroke-width="2"/>
       <text x="${W-padR+8}" y="${Y(neg.final_price).toFixed(1)}" dy="4" fill="var(--ink)" font-size="12" font-weight="700" font-family="ui-monospace,Menlo,monospace">DEAL ${money(neg.final_price)}</text>`
    : '';

  return `
  <div class="chart-wrap">
  <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Price trajectory">
    <rect x="${padL}" y="${Math.min(bandTop,bandBot).toFixed(1)}" width="${(W-padL-padR).toFixed(1)}"
          height="${Math.abs(bandBot-bandTop).toFixed(1)}" fill="var(--zopa)" opacity="0.28"/>
    <text x="${padL+6}" y="${Math.min(bandTop,bandBot)+16}" fill="#8a6d17" font-size="11"
          font-family="ui-monospace,Menlo,monospace" opacity="0.9">ZOPA (hidden from agents)</text>
    ${refLine(neg.fair_value,'fair', '#8a6d17','2 3')}
    ${refLine(neg.asking_price,'ask', 'var(--seller)','4 4')}
    <path d="${line(sellerPts)}" fill="none" stroke="var(--seller)" stroke-width="2.5"/>
    <path d="${line(buyerPts)}" fill="none" stroke="var(--buyer)" stroke-width="2.5"/>
    ${dots(sellerPts,'var(--seller)')}
    ${dots(buyerPts,'var(--buyer)')}
    ${final}
  </svg>
  </div>
  <div class="legend">
    <span><span class="swatch" style="background:var(--buyer)"></span>buyer offers</span>
    <span><span class="swatch" style="background:var(--seller)"></span>seller offers</span>
    <span><span class="swatch band" style="background:var(--zopa)"></span>bargaining zone</span>
    <span><span class="swatch" style="background:#8a6d17"></span>fair value</span>
  </div>`;
}

function fvPanel(neg){
  const fv = neg.fair_value;
  const card = (who, summary) => {
    const s = summary || {};
    if (s.first === null || s.first === undefined) {
      return `<div class="fv-card ${who}"><div class="who">${who}</div>
        <div class="row"><span class="k">estimate</span><span class="v">no estimate parsed</span></div></div>`;
    }
    const badge = s.changed
      ? `<span class="drift-badge">drifted</span>`
      : `<span class="nodrift-badge">held steady</span>`;
    const offSign = (s.diff !== null && s.diff > 0) ? '+' : '';
    const offClass = (s.pct_off !== null && Math.abs(s.pct_off) > 10) ? 'off' : '';
    return `<div class="fv-card ${who}">
      <div class="who">${who} ${badge}</div>
      <div class="row"><span class="k">their first estimate</span><span class="v">${money(s.first)}</span></div>
      <div class="row"><span class="k">actual fair value</span><span class="v">${money(fv)}</span></div>
      <div class="row"><span class="k">off by</span>
        <span class="v ${offClass}">${s.diff===null?'\u2014':offSign+money(s.diff)}${s.pct_off===null?'':' ('+(s.pct_off>0?'+':'')+s.pct_off.toFixed(1)+'%)'}</span></div>
      <div class="row"><span class="k">estimates given</span><span class="v">${s.count}</span></div>
    </div>`;
  };
  const fvs = neg.fv_summary || {};
  return `<div class="fv-panel">${card('buyer', fvs.buyer)}${card('seller', fvs.seller)}</div>`;
}

function transcript(neg){
  if (!neg.turns.length) return '';
  const rows = neg.turns.map(t => {
    const act = t.action + (typeof t.number==='number' ? ' &middot; ' + money(t.number) : '');
    const priv = (t.thoughts || t.reasoning || t.fair_value_estimate) ? `
      <details>
        <summary>show private reasoning</summary>
        <div class="private">
          ${t.fair_value_estimate ? `<div class="lbl">fair value estimate</div><div>${esc(t.fair_value_estimate)}</div>`:''}
          ${t.thoughts ? `<div class="lbl">thoughts on counterpart</div><div>${esc(t.thoughts)}</div>`:''}
          ${t.reasoning ? `<div class="lbl">reasoning</div><div>${esc(t.reasoning)}</div>`:''}
        </div>
      </details>` : '';
    return `<div class="turn ${t.speaker}">
       <div class="who">${esc(t.speaker)} <span class="act">&mdash; ${esc(act)}</span></div>
       <div class="msg">${esc(t.message)}</div>
       ${priv}
     </div>`;
  }).join('');
  return `<div class="turns">${rows}</div>`;
}

(function renderAggregate(){
  const el = document.getElementById('aggregate');
  if (el) el.innerHTML = aggregateChart();
})();

(function renderList(){
  const el = document.getElementById('list');
  if (!DATA.length) {
    el.innerHTML = `<div class="empty">No negotiations found in <code>runs/</code>. Run a batch first, then rebuild.</div>`;
    return;
  }
  el.innerHTML = DATA.map((neg, i) => {
    const isDeal = neg.outcome === 'deal';
    const pill = isDeal
      ? `<span class="pill deal">deal ${money(neg.final_price)}</span>`
      : `<span class="pill nodeal">${esc((neg.outcome||'').replace(/_/g,' '))}</span>`;
    const car = neg.car || {};
    const carline = `${car.year||''} ${car.model||''} &middot; ${car.mileage? Number(car.mileage).toLocaleString()+' mi':''} &middot; ${esc(car.damage||'')}`;
    const shareTxt = (neg.seller_share!==null && neg.seller_share!==undefined)
        ? ` &middot; seller took ${(neg.seller_share*100).toFixed(0)}% of zone` : '';
    const fvs = neg.fv_summary || {};
    const anyDrift = (fvs.buyer && fvs.buyer.changed) || (fvs.seller && fvs.seller.changed);
    const driftPill = anyDrift ? `<span class="pill drift">fv drifted</span>` : '';
    return `<details class="neg">
      <summary>
        <span class="idx">${String(i+1).padStart(2,'0')}</span>
        <span>
          <div class="neg-title">${esc(carline)}</div>
          <div class="neg-meta">${neg.num_turns??'?'} turns${shareTxt} &middot; ${esc(neg.filename)}</div>
        </span>
        <span style="display:flex; gap:8px; align-items:center;">${driftPill}${pill}</span>
      </summary>
      <div class="neg-body">
        ${trajectoryChart(neg)}
        ${fvPanel(neg)}
        ${transcript(neg)}
      </div>
    </details>`;
  }).join('');
})();
</script>
</body>
</html>
"""


def main():
    records = load_runs()
    stats = compute_stats(records)
    out = build_html(records, stats)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {OUT_FILE}")
    print(f"  {stats['total']} negotiations, {stats['deals']} deals")
    if stats["avg_seller_share"] is not None:
        print(f"  avg seller share of ZOPA: {stats['avg_seller_share']*100:.1f}%")
    print(f"  fair-value drift - buyer: {stats['buyer_drift_count']}/{stats['fv_tracked']}, "
          f"seller: {stats['seller_drift_count']}/{stats['fv_tracked']}")
    print("Open the file in a browser to view.")


if __name__ == "__main__":
    main()