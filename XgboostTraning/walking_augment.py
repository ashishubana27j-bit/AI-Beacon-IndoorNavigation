# walking_augment.py
# ══════════════════════════════════════════════════════════════
# Generates synthetic "walking between nodes" fingerprints by
# linearly interpolating RSSI and position between adjacent nodes.
#
# Run AFTER collecting real fingerprints, BEFORE training:
#   python3 walking_augment.py
#   python3 train_model.py --data fingerprints_augmented.json
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDIT ADJACENT_PAIRS BELOW to match your floor layout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json, math, random

INPUT_FILE          = "fingerprints.json"
OUTPUT_FILE         = "fingerprints_augmented.json"
STEPS_BETWEEN_NODES = 3     # synthetic points inserted between each pair
NOISE_DBM           = 2.0   # ±dBm Gaussian noise added to each RSSI
RANDOM_SEED         = 42

random.seed(RANDOM_SEED)

# ── EDIT THIS to match your walkable connections ──────────────
# List pairs of nodeIds that are physically adjacent (you can walk
# directly between them without going through another node).
# The direction doesn't matter — (A, B) == (B, A).
ADJACENT_PAIRS = [
    ("A", "B"),
    ("B", "C"),
    ("B", "D"),
    ("D", "E"),
    ("E", "F"),
    ("F", "G"),
    ("G", "H"),
    ("H", "I"),
    ("I", "J"),
     # washroom spur
]
# ─────────────────────────────────────────────────────────────

# ── Load fingerprints ─────────────────────────────────────────
with open(INPUT_FILE) as f:
    fps = json.load(f)

print(f"Loaded {len(fps)} real fingerprints from {INPUT_FILE}")

# ── Build per-node average RSSI ───────────────────────────────
node_sums   = {}
node_counts = {}
node_pos    = {}

for fp in fps:
    nid = fp["nodeId"]
    if nid not in node_sums:
        node_sums[nid]   = {}
        node_counts[nid] = {}
        node_pos[nid]    = {"x": fp["x"], "y": fp["y"]}
    for b, v in fp["signals"].items():
        node_sums[nid][b]   = node_sums[nid].get(b, 0) + v
        node_counts[nid][b] = node_counts[nid].get(b, 0) + 1

node_avg = {}
for nid in node_sums:
    node_avg[nid] = {
        "x": node_pos[nid]["x"],
        "y": node_pos[nid]["y"],
        "signals": {
            b: node_sums[nid][b] / node_counts[nid][b]
            for b in node_sums[nid]
        }
    }

all_beacons = sorted(set(b for n in node_avg.values() for b in n["signals"]))

# ── Generate synthetic fingerprints ──────────────────────────
synthetic = []
skipped   = []

for nA, nB in ADJACENT_PAIRS:
    if nA not in node_avg:
        skipped.append(f"{nA} not found")
        continue
    if nB not in node_avg:
        skipped.append(f"{nB} not found")
        continue

    a, b = node_avg[nA], node_avg[nB]
    dx   = b["x"] - a["x"]
    dy   = b["y"] - a["y"]
    heading_rad = math.atan2(dy, dx)

    for step in range(1, STEPS_BETWEEN_NODES + 1):
        t  = step / (STEPS_BETWEEN_NODES + 1)   # 0 < t < 1
        ix = round(a["x"] + t * dx)
        iy = round(a["y"] + t * dy)

        signals = {}
        for beacon in all_beacons:
            va = a["signals"].get(beacon, -100)
            vb = b["signals"].get(beacon, -100)
            if va <= -100 and vb <= -100:
                continue   # beacon not visible from either endpoint
            interp = va + t * (vb - va)
            noise  = random.gauss(0, NOISE_DBM)
            rssi   = round(max(-100, min(-40, interp + noise)))
            signals[beacon] = rssi

        if len(signals) < 3:
            continue   # not enough beacons — skip this point

        # Label: use whichever endpoint is closer
        label_node = nA if t < 0.5 else nB

        synthetic.append({
            "nodeId":      label_node,
            "x":           ix,
            "y":           iy,
            "headingSin":  round(math.sin(heading_rad), 6),
            "headingCos":  round(math.cos(heading_rad), 6),
            "orientation": "walking",
            "timestamp":   0,
            "signals":     signals,
            "_synthetic":  True,   # flag so you can filter them out later if needed
        })

# ── Save ──────────────────────────────────────────────────────
augmented = fps + synthetic

with open(OUTPUT_FILE, "w") as f:
    json.dump(augmented, f, indent=2)

print(f"\nReal fingerprints:      {len(fps)}")
print(f"Synthetic fingerprints: {len(synthetic)}")
print(f"Total:                  {len(augmented)}")
print(f"\nSaved to: {OUTPUT_FILE}")
if skipped:
    print(f"Skipped pairs: {skipped}")
print(f"\nNext step:")
print(f"  python3 train_model.py --data {OUTPUT_FILE}")
