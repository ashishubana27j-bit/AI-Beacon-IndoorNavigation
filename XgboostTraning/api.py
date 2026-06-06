"""
Indoor Navigation — Prediction API (Regressor Only)
====================================================
- Returns pure regressor coordinates (no node snap).
- Server‑side EMA applied to regressor position.
- NodeId / confidence / candidates still provided for client logic.
"""

import json, os, time, math
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

# ══════════════════════════════════════════════════════════
# COORDINATE MODE
# ══════════════════════════════════════════════════════════
COORDS_ARE_PIXELS = True
PIXELS_PER_METRE  = 40
ORIGIN_PX_X       = 0
ORIGIN_PX_Y       = 0

def coords_to_pixels(x: float, y: float) -> tuple[float, float]:
    if COORDS_ARE_PIXELS:
        return (x, y)
    return (
        ORIGIN_PX_X + x * PIXELS_PER_METRE,
        ORIGIN_PX_Y + y * PIXELS_PER_METRE,
    )

metres_to_pixels = coords_to_pixels

# ══════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════
app = FastAPI(title="Indoor Nav API", version="6.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

MODELS_DIR   = "models"
MISSING_RSSI = -100

# ══════════════════════════════════════════════════════════
# MODEL STORE
# ══════════════════════════════════════════════════════════
class _Store:
    node_model      = None
    x_model         = None
    y_model         = None
    label_encoder   = None
    all_beacons     = None
    feature_columns = None
    training_report = None
    node_coords: Dict[str, dict] = {}
    loaded = False
    error  = None

store = _Store()

def _load():
    required = [
        "node_classifier.pkl", "x_model.pkl", "y_model.pkl",
        "label_encoder.pkl",   "beacons.pkl", "feature_columns.pkl",
    ]
    missing = [f for f in required if not os.path.exists(f"{MODELS_DIR}/{f}")]
    if missing:
        store.error = f"Missing model files: {missing}. Run train_model.py first."
        return

    try:
        store.node_model      = joblib.load(f"{MODELS_DIR}/node_classifier.pkl")
        store.x_model         = joblib.load(f"{MODELS_DIR}/x_model.pkl")
        store.y_model         = joblib.load(f"{MODELS_DIR}/y_model.pkl")
        store.label_encoder   = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")
        store.all_beacons     = joblib.load(f"{MODELS_DIR}/beacons.pkl")
        store.feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
        store.loaded          = True

        rp = f"{MODELS_DIR}/training_report.json"
        if os.path.exists(rp):
            with open(rp) as f:
                store.training_report = json.load(f)

        fp_path = "fingerprints.json"
        if os.path.exists(fp_path):
            with open(fp_path) as f:
                fps = json.load(f)
            seen: Dict[str, dict] = {}
            for fp in fps:
                nid = fp["nodeId"]
                if nid not in seen:
                    raw_x, raw_y   = float(fp["x"]), float(fp["y"])
                    x_px, y_px     = coords_to_pixels(raw_x, raw_y)
                    seen[nid] = {
                        "x_raw": raw_x,  "y_raw": raw_y,
                        "x_px":  x_px,   "y_px":  y_px,
                    }
            store.node_coords = seen
            print(f"✓ {len(seen)} node coordinates loaded from {fp_path}")

        print(f"✓ Models ready — {len(store.all_beacons)} beacons, "
              f"{len(store.label_encoder.classes_)} nodes")
    except Exception as e:
        store.error = str(e)
        print(f"❌ {e}")

_load()

# ══════════════════════════════════════════════════════════
# SERVER‑SIDE EMA (on regressor position)
# ══════════════════════════════════════════════════════════
EMA_ALPHA = 0.40
_ema_state: Dict[str, dict] = {}

def _ema(client_id: str, x: float, y: float) -> dict:
    if client_id not in _ema_state:
        _ema_state[client_id] = {"x": x, "y": y}
        return _ema_state[client_id]
    prev = _ema_state[client_id]
    sx   = prev["x"] + EMA_ALPHA * (x - prev["x"])
    sy   = prev["y"] + EMA_ALPHA * (y - prev["y"])
    _ema_state[client_id] = {"x": sx, "y": sy}
    return _ema_state[client_id]

# ══════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    signals: Dict[str, float] = Field(..., description="beaconMinor → median RSSI")
    headingSin:  float          = Field(0.0)
    headingCos:  float          = Field(1.0)
    topK:        int            = Field(3, ge=1, le=10)
    client_id:   Optional[str]  = Field(None)

class NodeCandidate(BaseModel):
    nodeId:     str
    confidence: float
    x_m:        float
    y_m:        float
    x_px:       float
    y_px:       float

class Position(BaseModel):
    x_m:  float
    y_m:  float
    x_px: float
    y_px: float

class PredictResponse(BaseModel):
    position:        Position
    position_smooth: Position
    nodeId:          str
    confidence:      float
    topCandidates:   List[NodeCandidate]
    signalQuality:   str
    inferenceMs:     float
    beaconsUsed:     int
    beaconsMissing:  int

# ══════════════════════════════════════════════════════════
# CORE PREDICTION (REGRESSOR ONLY)
# ══════════════════════════════════════════════════════════
def _feature_vector(signals, hsin, hcos):
    row, used, missing = {}, 0, 0
    for b in store.all_beacons:
        key = f"b_{b}"
        if b in signals:
            row[key] = float(signals[b])
            used += 1
        else:
            row[key] = MISSING_RSSI
            missing += 1
    row["headingSin"] = hsin
    row["headingCos"] = hcos
    vec = [[row[c] for c in store.feature_columns]]
    return vec, used, missing

def _signal_quality(used: int, signals: Dict[str, float]) -> str:
    if used == 0: return "poor"
    avg = sum(signals.values()) / len(signals)
    if used >= 4 and avg > -75: return "strong"
    if used >= 3 and avg > -82: return "fair"
    if used >= 2:               return "weak"
    return "poor"

def _run_prediction(signals, hsin, hcos, topK, client_id=None) -> PredictResponse:
    vec, used, missing = _feature_vector(signals, hsin, hcos)
    t0 = time.perf_counter()

    # 1. Classifier
    probs    = store.node_model.predict_proba(vec)[0]
    top_idx  = np.argsort(probs)[::-1][:topK]
    best_idx = top_idx[0]
    conf     = float(probs[best_idx])
    node_id  = store.label_encoder.classes_[best_idx]

    candidates = []
    for idx in top_idx:
        nid = store.label_encoder.classes_[idx]
        c   = float(probs[idx])
        nc  = store.node_coords.get(nid, {})
        candidates.append(NodeCandidate(
            nodeId=nid, confidence=round(c, 4),
            x_m=nc.get("x_raw", 0), y_m=nc.get("y_raw", 0),
            x_px=nc.get("x_px", 0), y_px=nc.get("y_px", 0),
        ))

    # 2. Regressors – pure regressor coordinates (no node snap)
    reg_x_m = float(store.x_model.predict(vec)[0])
    reg_y_m = float(store.y_model.predict(vec)[0])
    reg_px_x, reg_px_y = coords_to_pixels(reg_x_m, reg_y_m)

    position = Position(
        x_m=round(reg_x_m, 4), y_m=round(reg_y_m, 4),
        x_px=round(reg_px_x, 2), y_px=round(reg_px_y, 2),
    )

    # 3. EMA smoothing (on regressor position)
    if client_id:
        sm = _ema(client_id, reg_px_x, reg_px_y)
        position_smooth = Position(
            x_m=round(reg_x_m, 4), y_m=round(reg_y_m, 4),
            x_px=round(sm["x"], 2), y_px=round(sm["y"], 2),
        )
    else:
        position_smooth = position

    ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        position=position,
        position_smooth=position_smooth,
        nodeId=node_id,
        confidence=round(conf, 4),
        topCandidates=candidates,
        signalQuality=_signal_quality(used, signals),
        inferenceMs=round(ms, 2),
        beaconsUsed=used,
        beaconsMissing=missing,
    )

# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════
@app.get("/health")
def health():
    if not store.loaded:
        return {"status": "error", "message": store.error, "models_loaded": False}
    r = store.training_report or {}
    return {
        "status": "ok", "models_loaded": True,
        "n_beacons": len(store.all_beacons),
        "n_nodes":   len(store.label_encoder.classes_),
        "pixels_per_metre": PIXELS_PER_METRE,
        "origin_px": {"x": ORIGIN_PX_X, "y": ORIGIN_PX_Y},
        "trained_at": r.get("trained_at"),
        "metrics":    r.get("metrics"),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not store.loaded:
        raise HTTPException(503, detail=store.error or "Models not loaded")
    if not req.signals:
        raise HTTPException(400, detail="signals dict is empty")
    return _run_prediction(req.signals, req.headingSin, req.headingCos, req.topK, req.client_id)

@app.get("/nodes")
def list_nodes():
    if not store.loaded:
        raise HTTPException(503)
    return {
        "nodes": [
            {
                "nodeId": nid,
                "x_px":   c.get("x_px", 0),
                "y_px":   c.get("y_px", 0),
                "x_raw":  c.get("x_raw", 0),
                "y_raw":  c.get("y_raw", 0),
            }
            for nid, c in store.node_coords.items()
        ],
        "beacons":           store.all_beacons,
        "coords_are_pixels": COORDS_ARE_PIXELS,
        "pixels_per_metre":  PIXELS_PER_METRE,
    }

@app.delete("/smooth/{client_id}")
def reset_smooth(client_id: str):
    _ema_state.pop(client_id, None)
    return {"reset": True, "client_id": client_id}

# ══════════════════════════════════════════════════════════
# WEB TEST UI
# ══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def web_ui():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Indoor Nav API</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#07070d;--s:#0f0f1a;--b:#18182c;--a:#4dffa8;--a2:#7c6fff;--r:#ff4d6d;--t:#e2e2f0;--m:#5a5a80;--mono:'JetBrains Mono',monospace;--sans:'Syne',sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t);font-family:var(--sans);min-height:100vh;display:flex;flex-direction:column}
header{border-bottom:1px solid var(--b);padding:16px 24px;display:flex;align-items:center;gap:12px}
.logo{width:32px;height:32px;background:linear-gradient(135deg,var(--a),var(--a2));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px}
header h1{font-size:16px;font-weight:800}
.v{font-family:var(--mono);font-size:10px;background:rgba(124,111,255,.15);color:var(--a2);border:1px solid rgba(124,111,255,.25);border-radius:4px;padding:1px 7px;margin-left:6px}
.hst{color:var(--m);font-size:11px;font-family:var(--mono);margin-left:auto;display:flex;align-items:center;gap:6px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--m)}
.dot.ok{background:var(--a);box-shadow:0 0 5px var(--a)}.dot.err{background:var(--r)}
main{display:grid;grid-template-columns:1fr 1fr;flex:1;overflow:hidden}
.panel{padding:22px 24px;border-right:1px solid var(--b);overflow-y:auto}
.pt{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--m);margin-bottom:14px}
label{display:block;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--m);margin-bottom:6px;margin-top:14px}
label:first-of-type{margin-top:0}
textarea,input[type=text],input[type=number]{width:100%;background:var(--s);border:1px solid var(--b);color:var(--t);font-family:var(--mono);font-size:12px;border-radius:7px;padding:9px 11px;outline:none;transition:border-color .2s;resize:vertical}
textarea:focus,input:focus{border-color:var(--a2)}
.row{display:flex;gap:8px}.row>div{flex:1}
.sub{font-size:10px;color:var(--m);font-family:var(--mono);margin-top:3px}
button{margin-top:14px;width:100%;padding:11px;background:linear-gradient(135deg,var(--a),var(--a2));color:#07070d;font-family:var(--sans);font-size:13px;font-weight:800;border:none;border-radius:7px;cursor:pointer;transition:opacity .2s}
button:hover{opacity:.88}button:disabled{opacity:.4;cursor:not-allowed}
.rbox{background:var(--s);border:1px solid var(--b);border-radius:7px;padding:12px;font-family:var(--mono);font-size:11.5px;line-height:1.7;white-space:pre;overflow:auto;min-height:80px;max-height:280px}
.mg{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}
.mc{background:var(--s);border:1px solid var(--b);border-radius:7px;padding:11px}
.mv{font-size:20px;font-weight:800;font-family:var(--mono);letter-spacing:-1px;color:var(--a)}
.mv.b{color:var(--a2)}.mv.sm{font-size:14px}
.ml{font-size:10px;color:var(--m);text-transform:uppercase;letter-spacing:1px;margin-top:2px}
.cbar{height:3px;background:var(--b);border-radius:2px;margin-top:6px;overflow:hidden}
.cfill{height:100%;background:linear-gradient(90deg,var(--a2),var(--a));transition:width .4s}
.cand{display:flex;align-items:center;gap:7px;padding:6px 0;border-bottom:1px solid var(--b);font-family:var(--mono);font-size:11px}
.cand:last-child{border-bottom:none}
.cn{color:var(--a2);font-weight:600;min-width:32px}.cp{color:var(--m);min-width:44px}
.cb{flex:1;height:2px;background:var(--b);border-radius:2px;overflow:hidden}
.cbf{height:100%;background:var(--a2)}
.sq{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-family:var(--mono)}
.strong{background:rgba(77,255,168,.12);color:var(--a);border:1px solid rgba(77,255,168,.2)}
.fair{background:rgba(255,200,0,.1);color:#ffc800;border:1px solid rgba(255,200,0,.18)}
.weak{background:rgba(255,150,0,.1);color:#ff9600;border:1px solid rgba(255,150,0,.18)}
.poor{background:rgba(255,77,109,.1);color:var(--r);border:1px solid rgba(255,77,109,.18)}
.err{color:var(--r)}
.pos-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:14px}
.pos-box{background:var(--s);border:1px solid var(--b);border-radius:7px;padding:10px}
.pos-label{font-size:10px;font-weight:700;letter-spacing:1px;color:var(--m);text-transform:uppercase;margin-bottom:6px}
.pos-vals{display:flex;gap:10px}
.pos-val{font-family:var(--mono);font-size:13px;font-weight:600}
.pos-val span{font-size:10px;color:var(--m);font-weight:400;margin-left:2px}
@media(max-width:720px){main{grid-template-columns:1fr}.panel{border-right:none;border-bottom:1px solid var(--b)}}
</style>
</head>
<body>
<header>
  <div class="logo">📍</div>
  <h1>Indoor Nav API<span class="v">v3</span></h1>
  <div class="hst"><div class="dot" id="sd"></div><span id="ss">checking…</span></div>
</header>
<main>
<div class="panel">
  <div class="pt">Send Prediction</div>
  <label>Beacon Signals (JSON — minor: RSSI)</label>
  <textarea id="sig" rows="5">{"1":-72,"2":-81,"3":-66,"4":-90}</textarea>
  <label>Heading</label>
  <div class="row">
    <div><input type="number" id="hs" value="0.99" step="0.01" min="-1" max="1"><div class="sub">headingSin</div></div>
    <div><input type="number" id="hc" value="0.01" step="0.01" min="-1" max="1"><div class="sub">headingCos</div></div>
  </div>
  <label>Client ID (for server EMA smoothing)</label>
  <input type="text" id="cid" value="device_001">
  <button id="btn" onclick="go()">▶ Predict Position</button>
  <div style="margin-top:18px"><div class="pt">Raw Response</div><div class="rbox" id="raw">— waiting —</div></div>
</div>
<div class="panel">
  <div class="pt">Result</div>
  <div class="mg">
    <div class="mc"><div class="mv" id="mn">—</div><div class="ml">Node</div></div>
    <div class="mc"><div class="mv" id="mconf">—</div><div class="ml">Confidence</div><div class="cbar"><div class="cfill" id="cb" style="width:0"></div></div></div>
  </div>
  <div class="pos-grid">
    <div class="pos-box">
      <div class="pos-label">Blended Position</div>
      <div class="pos-vals">
        <div class="pos-val" id="px_m">—<span>m</span></div>
        <div class="pos-val" id="py_m">—<span>m</span></div>
      </div>
      <div class="pos-vals" style="margin-top:4px">
        <div class="pos-val b sm" id="px_px">—<span>px</span></div>
        <div class="pos-val b sm" id="py_px">—<span>px</span></div>
      </div>
    </div>
    <div class="pos-box">
      <div class="pos-label">Smooth Position</div>
      <div class="pos-vals">
        <div class="pos-val" id="sx_m">—<span>m</span></div>
        <div class="pos-val" id="sy_m">—<span>m</span></div>
      </div>
      <div class="pos-vals" style="margin-top:4px">
        <div class="pos-val b sm" id="sx_px">—<span>px</span></div>
        <div class="pos-val b sm" id="sy_px">—<span>px</span></div>
      </div>
    </div>
  </div>
  <div style="display:flex;gap:10px;align-items:center;font-size:11px;color:var(--m);font-family:var(--mono);margin-bottom:14px">
    <span>⏱ <span id="ms">—</span>ms</span>
    <span>📡 <span id="bu">—</span> beacons seen</span>
    <span id="sq"></span>
  </div>
  <div class="pt">Top Candidates</div>
  <div id="cands" style="margin-bottom:14px"><div style="color:var(--m);font-size:12px;font-family:var(--mono)">— run a prediction —</div></div>
  <div class="pt">Server Health</div>
  <div class="rbox" id="mi">loading…</div>
</div>
</main>
<script>
async function checkH(){
  try{
    const d=await(await fetch('/health')).json();
    const ok=d.status==='ok';
    document.getElementById('sd').className='dot '+(ok?'ok':'err');
    document.getElementById('ss').textContent=ok?`${d.n_nodes} nodes · ${d.n_beacons} beacons · ${d.pixels_per_metre}px/m`:(d.message||'error');
    document.getElementById('mi').textContent=JSON.stringify(d,null,2);
  }catch{document.getElementById('ss').textContent='unreachable';}
}
function set(id,v){const e=document.getElementById(id);if(e)e.textContent=v;}
async function go(){
  const btn=document.getElementById('btn');btn.disabled=true;btn.textContent='⏳…';
  let sig;try{sig=JSON.parse(document.getElementById('sig').value);}catch{alert('Invalid JSON');btn.disabled=false;btn.textContent='▶ Predict Position';return;}
  const body={signals:sig,headingSin:+document.getElementById('hs').value,headingCos:+document.getElementById('hc').value,topK:5,client_id:document.getElementById('cid').value||undefined};
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    document.getElementById('raw').textContent=JSON.stringify(d,null,2);
    if(r.ok){
      set('mn',d.nodeId);
      set('mconf',(d.confidence*100).toFixed(1)+'%');
      document.getElementById('cb').style.width=(d.confidence*100)+'%';
      const p=d.position, s=d.position_smooth;
      set('px_m', p.x_m+' m'); set('py_m', p.y_m+' m');
      set('px_px',p.x_px+' px'); set('py_px',p.y_px+' px');
      set('sx_m', s.x_m+' m'); set('sy_m', s.y_m+' m');
      set('sx_px',s.x_px+' px'); set('sy_px',s.y_px+' px');
      set('ms',d.inferenceMs); set('bu',d.beaconsUsed);
      document.getElementById('sq').innerHTML=`<span class="sq ${d.signalQuality}">${d.signalQuality}</span>`;
      document.getElementById('cands').innerHTML=d.topCandidates.map(c=>`
        <div class="cand">
          <span class="cn">${c.nodeId}</span>
          <span class="cp">${(c.confidence*100).toFixed(1)}%</span>
          <div class="cb"><div class="cbf" style="width:${c.confidence*100}%"></div></div>
          <span style="color:var(--m);font-size:10px">(${c.x_m}m, ${c.y_m}m) = (${c.x_px}px, ${c.y_px}px)</span>
        </div>`).join('');
    }else{document.getElementById('raw').innerHTML='<span class="err">'+JSON.stringify(d,null,2)+'</span>';}
  }catch(e){document.getElementById('raw').textContent='Error: '+e.message;}
  btn.disabled=false;btn.textContent='▶ Predict Position';
}
checkH();
</script>
</body></html>"""