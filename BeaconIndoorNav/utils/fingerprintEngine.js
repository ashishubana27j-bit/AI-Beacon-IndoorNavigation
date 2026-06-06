/**
 * fingerprintEngine.js
 * ══════════════════════════════════════════════════════════════
 *
 * This file does two separate things:
 *
 *  1. FILE I/O  — load / save / clear fingerprints.json
 *                 (used by setupScreen and the ML pipeline)
 *
 *  2. KNN MATCHER — used ONLY as a local fallback when the
 *                   ML API server is unreachable.  In normal
 *                   operation index.js calls the API; this
 *                   matcher is a safety net.
 *
 * FORMAT written by setupScreen_final.js:
 * ─────────────────────────────────────────
 * {
 *   nodeId:      "A",
 *   x:           120,        ← pixel x from floorData
 *   y:           340,        ← pixel y from floorData
 *   headingSin:  0.993,      ← sin(heading_radians)
 *   headingCos:  0.017,      ← cos(heading_radians)
 *   orientation: "north",    ← human label (north/east/south/west)
 *   timestamp:   1718000000000,
 *   signals: {
 *     "1": -72,              ← beacon minor: median RSSI (integer)
 *     "2": -81,
 *     "3": -66
 *   }
 * }
 *
 * This is ALSO the format train_model.py expects — so the file
 * can be copied straight to your Python project for training.
 */

import RNFS from 'react-native-fs';
import { FLOOR_DATA } from '../data/floorData';

/* ══════════════════════════════════════════════════════════
   CONFIG
══════════════════════════════════════════════════════════ */
const FILE_PATH            = RNFS.DocumentDirectoryPath + '/fingerprints.json';
const K_NEAREST            = 3;       // how many neighbours to blend
const MIN_BEACONS_REQUIRED = 3;       // reject fingerprints with fewer beacons
const ORIENTATION_PENALTY  = 2.0;     // extra distance added for wrong orientation
const HEADING_WEIGHT       = 8.0;     // how much heading sin/cos contributes to distance

/* ══════════════════════════════════════════════════════════
   FILE I/O
══════════════════════════════════════════════════════════ */

export async function loadFingerprints() {
  try {
    const exists = await RNFS.exists(FILE_PATH);
    if (!exists) return [];
    const raw = await RNFS.readFile(FILE_PATH, 'utf8');
    return JSON.parse(raw);
  } catch (e) {
    console.warn('loadFingerprints error:', e);
    return [];
  }
}

export async function saveFingerprints(data) {
  try {
    await RNFS.writeFile(FILE_PATH, JSON.stringify(data), 'utf8');
  } catch (e) {
    console.warn('saveFingerprints error:', e);
  }
}

export async function clearFingerprintsFile() {
  try {
    const exists = await RNFS.exists(FILE_PATH);
    if (exists) await RNFS.unlink(FILE_PATH);
  } catch (e) {
    console.warn('clearFingerprintsFile error:', e);
  }
}

/* ══════════════════════════════════════════════════════════
   KNN MATCHER  (local fallback — API is preferred)
══════════════════════════════════════════════════════════ */

/**
 * Normalise RSSI to [-100, -30] range.
 * Clips extreme values that indicate sensor noise.
 */
function normalizeRSSI(rssi) {
  if (rssi === undefined || rssi === null) return -100;
  return Math.max(-100, Math.min(-30, rssi));
}

/**
 * Weighted Euclidean distance between a live scan and a stored fingerprint.
 *
 * Distance formula:
 *   For each beacon in the fingerprint:
 *     diff = |live_rssi - fp_rssi|
 *     weight = 1 / |fp_rssi|   (stronger beacon → higher weight)
 *     contribution = diff² × weight
 *
 *   Plus heading contribution:
 *     heading_dist = (sin_diff² + cos_diff²) × HEADING_WEIGHT
 *
 * Returns Infinity if fewer than MIN_BEACONS_REQUIRED beacons match.
 */
function calculateDistance(liveSignals, fp, liveHeadingSin, liveHeadingCos) {
  let sum   = 0;
  let count = 0;

  // ── RSSI component ──────────────────────────────────────────
  Object.entries(fp.signals).forEach(([minor, fpRssiRaw]) => {
    const fpRssi   = normalizeRSSI(fpRssiRaw);
    const liveRssi = liveSignals[minor] !== undefined
      ? normalizeRSSI(liveSignals[minor])
      : -100;                          // treat missing beacon as very far

    if (fpRssi < -95) return;          // fp beacon too weak — skip

    const weight = 1 / Math.max(Math.abs(fpRssi), 1);
    const diff   = Math.abs(liveRssi - fpRssi);
    sum  += diff * diff * weight;
    count++;
  });

  if (count < MIN_BEACONS_REQUIRED) return Infinity;

  const rssiDist = Math.sqrt(sum / count);

  // ── Heading component ───────────────────────────────────────
  // Only used if the fingerprint has sin/cos (new format).
  // Falls back to 0 if not present (old format compatibility).
  let headingDist = 0;
  if (fp.headingSin !== undefined && liveHeadingSin !== undefined) {
    const dSin   = liveHeadingSin - fp.headingSin;
    const dCos   = liveHeadingCos - fp.headingCos;
    headingDist  = Math.sqrt(dSin * dSin + dCos * dCos) * HEADING_WEIGHT;
  }

  return rssiDist + headingDist;
}

/**
 * Match live beacon signals against stored fingerprints.
 * Returns { x, y, confidence, nodeId, matches } or null.
 *
 * @param {Object}  liveSignals       { "minor": rssi, ... }
 * @param {number}  liveHeadingSin    sin(compass heading in radians)
 * @param {number}  liveHeadingCos    cos(compass heading in radians)
 * @param {boolean} useOrientFilter   if true, only match same orientation
 * @returns {{ x, y, confidence, nodeId, nearestDistance, matches } | null}
 */
export function matchFingerprint(
  liveSignals,
  liveHeadingSin = 0,
  liveHeadingCos = 1,
  useOrientFilter = false,
) {
  if (!liveSignals) return null;

  const fps = FLOOR_DATA.fingerprints;
  if (!fps?.length) return null;

  // ── Derive current orientation from heading for optional filter ──
  const headingDeg = Math.round(
    (Math.atan2(liveHeadingSin, liveHeadingCos) * 180) / Math.PI + 360
  ) % 360;
  const currentOrientation =
    headingDeg >= 315 || headingDeg < 45  ? 'north' :
    headingDeg >= 45  && headingDeg < 135 ? 'east'  :
    headingDeg >= 135 && headingDeg < 225 ? 'south' : 'west';

  const results = [];

  fps.forEach(fp => {
    // Optional orientation filter
    if (useOrientFilter && fp.orientation && fp.orientation !== currentOrientation) {
      return;
    }

    const dist = calculateDistance(fp, liveSignals, liveHeadingSin, liveHeadingCos);
    if (!isFinite(dist)) return;

    results.push({ fp, dist });
  });

  if (!results.length) return null;

  results.sort((a, b) => a.dist - b.dist);
  const nearest = results.slice(0, K_NEAREST);

  // ── Weighted average of K nearest positions ──────────────────
  let x = 0, y = 0, totalWeight = 0;
  nearest.forEach(({ fp, dist }) => {
    const w = 1 / Math.max(dist, 0.0001);
    x           += fp.x * w;
    y           += fp.y * w;
    totalWeight += w;
  });

  if (totalWeight === 0) return null;

  x /= totalWeight;
  y /= totalWeight;

  // ── Confidence: inversely proportional to nearest distance ───
  // Range: 0 (far) → 1 (perfect match, dist ≈ 0)
  const confidence = 1 / (1 + nearest[0].dist);

  return {
    x,
    y,
    confidence,
    nodeId:          nearest[0].fp.nodeId,
    nearestDistance: nearest[0].dist,
    matches:         nearest.map(r => ({ nodeId: r.fp.nodeId, dist: r.dist })),
  };
}

/* ══════════════════════════════════════════════════════════
   UTILITY EXPORTS
══════════════════════════════════════════════════════════ */

/** Human-readable quality label from KNN confidence (0–1). */
export function getTrackingQuality(confidence) {
  if (confidence > 0.35) return 'EXCELLENT';
  if (confidence > 0.20) return 'GOOD';
  if (confidence > 0.10) return 'FAIR';
  return 'POOR';
}

/** Nearest walk graph node to a pixel position. */
export function findNearestFingerprintNode(position) {
  if (!position) return null;
  let nearest = null, minDist = Infinity;
  FLOOR_DATA.walkNodes.forEach(node => {
    const d = Math.hypot(node.x - position.x, node.y - position.y);
    if (d < minDist) { minDist = d; nearest = node; }
  });
  return nearest;
}
