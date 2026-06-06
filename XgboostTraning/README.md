# Indoor Navigation — XGBoost Positioning System

WiFi/BLE fingerprint-based indoor positioning with a FastAPI prediction server.

## Project Structure

```
indoor-nav/
├── fingerprints.json       ← your collected scan data
├── train_model.py          ← training pipeline
├── api.py                  ← FastAPI prediction server
├── requirements.txt
└── models/                 ← generated after training
    ├── node_classifier.pkl
    ├── x_model.pkl
    ├── y_model.pkl
    ├── label_encoder.pkl
    ├── beacons.pkl
    ├── feature_columns.pkl
    └── training_report.json
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your fingerprint data
Edit `fingerprints.json` with your real scan data, or start with the sample file provided.

**Format:**
```json
[
  {
    "nodeId": "A",
    "x": 2.0,
    "y": 5.0,
    "headingSin": 1.0,
    "headingCos": 0.0,
    "signals": {
      "beacon_mac_1": { "median": -70 },
      "beacon_mac_2": { "median": -82 }
    }
  }
]
```

### 3. Train the models
```bash
python train_model.py
```

Options:
```bash
python train_model.py --data my_data.json     # custom file
python train_model.py --test-size 0.25        # 75/25 train/test split
python train_model.py --no-cv                 # skip cross-validation (faster)
```

### 4. Start the API
```bash
uvicorn api:app --reload --port 8000
```

Open http://localhost:8000 to use the web tester.

---

## API Endpoints

### `POST /predict`
Main prediction endpoint.

**Request:**
```json
{
  "signals": {
    "beacon_1": -72,
    "beacon_2": -81,
    "beacon_3": -66
  },
  "headingSin": 0.99,
  "headingCos": 0.01,
  "topK": 3
}
```

**Response:**
```json
{
  "nodeId": "B",
  "confidence": 0.94,
  "x": 5.3,
  "y": 4.8,
  "topCandidates": [
    { "nodeId": "B", "confidence": 0.94, "x": 5.3, "y": 4.8 },
    { "nodeId": "E", "confidence": 0.04, "x": 5.3, "y": 4.8 },
    { "nodeId": "C", "confidence": 0.02, "x": 5.3, "y": 4.8 }
  ],
  "inferenceMs": 12.4,
  "beaconsUsed": 3,
  "beaconsMissing": 1
}
```

### `GET /health`
Returns API status + model metrics.

### `GET /nodes`
Lists all trained nodes and beacons.

### `GET /training-report`
Full training report with accuracy metrics.

---

## React Native Integration

```javascript
// Call from your BLE/WiFi scanning loop
async function getPosition(beaconReadings, heading) {
  const headingRad = (heading * Math.PI) / 180;

  const response = await fetch('http://YOUR_SERVER_IP:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      signals: beaconReadings,   // { "mac_addr": rssiValue, ... }
      headingSin: Math.sin(headingRad),
      headingCos: Math.cos(headingRad),
      topK: 3,
    }),
  });

  const result = await response.json();
  // result.nodeId   → "B"
  // result.x, result.y → coordinates in meters
  // result.confidence → 0–1
  return result;
}
```

---

## Data Collection Guidelines

| Item              | Minimum | Target      |
|-------------------|---------|-------------|
| Nodes             | 50+     | 100–300     |
| Scans per node    | 4       | 4 (N/S/E/W) |
| Total fingerprints| 200+    | 400–1200    |
| Visible beacons   | 3+      | 4–8         |

**Tips:**
- Collect scans at consistent heights (waist level)
- Wait 2–3 seconds per orientation for signals to stabilize
- Use median RSSI over multiple samples per scan (reduces noise)
- Re-collect fingerprints if you move beacons
- Collect at different times of day if possible (people affect signals)

---

## Accuracy Targets

| Metric          | Good    | Excellent |
|-----------------|---------|-----------|
| Node accuracy   | > 85%   | > 95%     |
| X error (MAE)   | < 1.5 m | < 0.5 m   |
| Y error (MAE)   | < 1.5 m | < 0.5 m   |
| Inference time  | < 50 ms | < 20 ms   |
