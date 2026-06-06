/**
 * index.js — Indoor Navigation (Consensus Node, Regressor‑Based)
 *
 * - API returns pure regressor position (no snap).
 * - Client builds a consensus node from the last N predictions.
 * - Dot pulls gently toward the consensus node's surveyed position
 *   only when confident and nearby.
 * - Continuous movement between nodes via regressor interpolation.
 */

const API_BASE = 'http://192.168.110.111:8000';   // your server IP
const PIXELS_PER_METRE = 40;
const ORIGIN_PX = { x: 0, y: 0 };

import { Picker } from '@react-native-picker/picker';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Animated, DeviceEventEmitter, Platform,
  Text, TouchableOpacity, View,
} from 'react-native';
import Beacons from 'react-native-beacons-manager';
import CompassHeading from 'react-native-compass-heading';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

import { PermissionsAndroid } from 'react-native';
import { FLOOR_DATA } from '../../data/floorData';
import FloorMap from '../../utils/FloorMap';
import KalmanFilter from '../../utils/kalman';

// ══════════════════════════════════════════════════════════
// TUNING
// ══════════════════════════════════════════════════════════
const BEACON_THROTTLE_MS = 100;
const SCAN_INTERVAL_MS   = 400;
const RENDER_FPS_MS      = 20;
const BEACON_STALE_MS    = 6000;
const RSSI_WINDOW        = 7;
const ARRIVAL_RADIUS_M   = 2.5;
const ARRIVAL_RADIUS_PX  = ARRIVAL_RADIUS_M * PIXELS_PER_METRE;
const PATH_LOCK_RADIUS   = 60;

const LERP_FAR  = 0.8;
const LERP_MID  = 0.5;
const LERP_NEAR = 0.3;

// Consensus node settings
const NODE_HISTORY_SIZE = 5;        // how many recent predictions to consider
const CONFIDENCE_THRESHOLD = 0.6;   // minimum confidence to apply pull
const MAX_DIST_TO_NODE     = 80;    // pixels – only pull if within this radius
const NODE_SNAP_STRENGTH   = 0.85;  // how strongly to pull toward the node (0‑1)
const CLIENT_EMA_ALPHA     = 0.5;   // smoothing on final target position

// ── Stable device ID ────────────────────────────────────
const DEVICE_ID = `device_${Math.random().toString(36).slice(2, 9)}`;

// ── Kalman filters ──────────────────────────────────────
const kfX = new KalmanFilter({ R: 0.4, Q: 0.05 });
const kfY = new KalmanFilter({ R: 0.4, Q: 0.05 });

// ── RSSI buffer (outside React) ─────────────────────────
const rssiHistory = {};
const lastSeenAt  = {};

// ── Beacon Region ──────────────────────────────────────
const REGION = {
  identifier: 'IndoorRegion',
  uuid: '2f234454cf6d4a0fadf2f4911ba9ffa6',
};


function pushRssi(minor, rssi) {
  if (!rssiHistory[minor]) rssiHistory[minor] = [];
  const med = getMedianRssi(minor);
  if (med !== null && Math.abs(rssi - med) > 15) return;
  rssiHistory[minor].push(rssi);
  if (rssiHistory[minor].length > RSSI_WINDOW) rssiHistory[minor].shift();
  lastSeenAt[minor] = Date.now();
}
export async function requestBlePermissions() {
  if (Platform.OS === 'android') {
    try {
      const granted = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
      ]);

      const allGranted = Object.values(granted).every(
        v => v === PermissionsAndroid.RESULTS.GRANTED
      );

      return allGranted;
    } catch (err) {
      console.warn(err);
      return false;
    }
  }
   return true; // iOS handled separately
}

function getMedianRssi(minor) {
  const arr = rssiHistory[minor];
  if (!arr?.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function snapshotSignals() {
  const now = Date.now();
  const out = {};
  for (const minor of Object.keys(lastSeenAt)) {
    if (now - lastSeenAt[minor] > BEACON_STALE_MS) continue;
    const med = getMedianRssi(minor);
    if (med !== null) out[minor] = med;
  }
  return out;
}

// ── Coordinate helpers ─────────────────────────────────
function pixelToGeo(x, y) {
  const aX = 350, aY = 990;
  const aLat = 25.26417611, aLng = 55.38526176;
  const latPx = (25.26418020 - 25.26417611) / (1230 - 990);
  const lngPx = (55.38534943 - 55.38526176) / (480  - 350);
  return {
    latitude:  aLat + (y - aY) * latPx,
    longitude: aLng + (x - aX) * lngPx,
  };
}

// ── Path helpers ────────────────────────────────────────
function dist2D(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function projectToSegment(p, v, w) {
  const dx = w.x - v.x, dy = w.y - v.y;
  const l2 = dx * dx + dy * dy;
  if (l2 === 0) return { point: v, distFromP: dist2D(p, v) };
  const t     = Math.max(0, Math.min(1, ((p.x - v.x) * dx + (p.y - v.y) * dy) / l2));
  const point = { x: v.x + t * dx, y: v.y + t * dy };
  return { point, t, distFromP: dist2D(p, point) };
}

function closestPointOnPath(p, path) {
  if (!path || path.length < 2) return null;
  let best = null;
  for (let i = 0; i < path.length - 1; i++) {
    const r = projectToSegment(p, path[i], path[i + 1]);
    if (!best || r.distFromP < best.dist)
      best = { point: r.point, segIndex: i, dist: r.distFromP };
  }
  return best;
}

function constrainToWalkGraph(pos) {
  const nodes   = FLOOR_DATA.walkNodes;
  let bestPt    = pos, bestDist = Infinity;
  for (const node of nodes) {
    for (const linkId of node.links) {
      const linked = nodes.find(n => n.id === linkId);
      if (!linked) continue;
      const r = projectToSegment(pos, node, linked);
      if (r.distFromP < bestDist) { bestDist = r.distFromP; bestPt = r.point; }
    }
  }
  return bestDist < 80 ? bestPt : pos;
}

function bfsPath(startId, endId) {
  const queue = [[startId]], visited = new Set();
  while (queue.length) {
    const route  = queue.shift();
    const nodeId = route[route.length - 1];
    if (nodeId === endId) return route;
    if (!visited.has(nodeId)) {
      visited.add(nodeId);
      const node = FLOOR_DATA.walkNodes.find(n => n.id === nodeId);
      if (node) node.links.forEach(id => queue.push([...route, id]));
    }
  }
  return [];
}

function nearestNode(pos) {
  if (!pos) return null;
  return FLOOR_DATA.walkNodes.reduce((best, n) => {
    const d = dist2D(n, pos);
    return (!best || d < best._d) ? { ...n, _d: d } : best;
  }, null);
}

function remainingPathDistancePx(pos, path, segIndex) {
  if (!path || path.length < 2 || !pos) return 0;
  const idx  = Math.min(segIndex, path.length - 2);
  let total  = dist2D(pos, path[idx + 1]);
  for (let i = idx + 1; i < path.length - 1; i++)
    total += dist2D(path[i], path[i + 1]);
  return total;
}

function formatDistance(px) {
  const m = px / PIXELS_PER_METRE;
  if (m < 10)  return `${m.toFixed(1)} m`;
  if (m < 100) return `${Math.round(m)} m`;
  return `${(m / 1000).toFixed(2)} km`;
}

// ══════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════
export default function App() {

  const [userPosition,        setUserPosition]        = useState(null);
  const [geoPosition,         setGeoPosition]         = useState(null);
  const [heading,             setHeading]             = useState(0);
  const [beaconCount,         setBeaconCount]         = useState(0);
  const [status,              setStatus]              = useState('📡 Searching…');
  const [confidence,          setConfidence]          = useState(0);
  const [signalQuality,       setSignalQuality]       = useState('poor');
  const [predictedNode,       setPredictedNode]       = useState(null);
  const [apiError,            setApiError]            = useState(false);

  const [selectedDestination, setSelectedDestination] = useState(null);
  const [path,                setPath]                = useState([]);
  const [distanceText,        setDistanceText]        = useState('');
  const [hasArrived,          setHasArrived]          = useState(false);
  const [isNavigating,        setIsNavigating]        = useState(false);

  const arrivalOpacity = useRef(new Animated.Value(0)).current;
  const arrivalScale   = useRef(new Animated.Value(0.7)).current;

  const rawPos       = useRef(null);
  const smoothPos    = useRef(null);
  const headingRef   = useRef(0);
  const pathRef      = useRef([]);
  const segIndexRef  = useRef(0);
  const hasFixRef    = useRef(false);
  const arrivedRef   = useRef(false);
  const destPxRef    = useRef(null);
  const lastBeaconMs = useRef(0);

  // Consensus tracking
  const nodeHistory     = useRef([]);   // recent nodeIds
  const nodePositions   = useRef({});   // nodeId → {x_px, y_px}
  const emaPos          = useRef(null); // EMA-smoothed target

  useEffect(() => {
    pathRef.current = path;
    if (rawPos.current && path.length >= 2) {
      const snap = closestPointOnPath(rawPos.current, path);
      if (snap) {
        segIndexRef.current = snap.segIndex;
        rawPos.current      = snap.point;
      }
    }
  }, [path]);

  useEffect(() => {
    if (!selectedDestination) {
      destPxRef.current = null;
      setHasArrived(false);
      setIsNavigating(false);
      arrivedRef.current = false;
      setDistanceText('');
      return;
    }
    const dest = FLOOR_DATA.destinations.find(d => d.id === selectedDestination);
    if (!dest) return;
    const node = FLOOR_DATA.walkNodes.find(n => n.id === dest.nodeId);
    if (node) destPxRef.current = { x: node.x, y: node.y };
    setHasArrived(false);
    arrivedRef.current = false;
    setIsNavigating(true);
    arrivalOpacity.setValue(0);
    arrivalScale.setValue(0.7);
  }, [selectedDestination]);

  // Compass
  useEffect(() => {
    CompassHeading.start(3, ({ heading: h }) => {
      headingRef.current = h;
      setHeading(h);
    });
    return () => CompassHeading.stop();
  }, []);

  // Beacons
  useEffect(() => {
    async function init() {

      const ok = await requestBlePermissions();
    if (!ok) {
      setStatus('❌ Permissions not granted');
      return;
    }

    
      if (Platform.OS === 'android') await Beacons.detectIBeacons();
      else await Beacons.requestAlwaysAuthorization();
      await Beacons.startRangingBeaconsInRegion(REGION);
    }
    init();

    const sub = DeviceEventEmitter.addListener('beaconsDidRange', data => {
      const now = Date.now();
      if (now - lastBeaconMs.current < BEACON_THROTTLE_MS) return;
      lastBeaconMs.current = now;
      if (!data.beacons?.length) return;

      let count = 0;
      data.beacons.forEach(b => {
        if (!b.rssi || b.rssi === 0) return;
        const exists = FLOOR_DATA.beacons?.some(m => m.minor === String(b.minor));
        if (!exists) return;
        pushRssi(String(b.minor), b.rssi);
        count++;
      });
      setBeaconCount(count);
    });

    return () => {
      sub.remove();
      Beacons.stopRangingBeaconsInRegion(REGION);
    };
  }, []);

  // ── Get consensus node ────────────────────────────────
  const getConsensusNode = () => {
    if (nodeHistory.current.length === 0) return null;
    const freq = {};
    nodeHistory.current.forEach(id => { freq[id] = (freq[id] || 0) + 1; });
    let bestNode = null, bestCount = 0;
    for (const [id, count] of Object.entries(freq)) {
      if (count > bestCount || (count === bestCount && nodeHistory.current.lastIndexOf(id) > nodeHistory.current.lastIndexOf(bestNode))) {
        bestNode = id;
        bestCount = count;
      }
    }
    return bestNode;
  };

  // ── Prediction loop ───────────────────────────────────
  useEffect(() => {
    const timer = setInterval(async () => {
      const signals = snapshotSignals();
      if (Object.keys(signals).length === 0) {
        setStatus('📡 No beacons visible…');
        return;
      }

      const headingRad = (headingRef.current * Math.PI) / 180;

      try {
        const res = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            signals,
            headingSin: Math.sin(headingRad),
            headingCos: Math.cos(headingRad),
            topK: 3,
            client_id: DEVICE_ID,
          }),
        });

        if (!res.ok) {
          setApiError(true);
          setStatus('⚠️ Server error');
          return;
        }

        const data = await res.json();
        setApiError(false);

        // API now returns regressor-only position (no snap)
        const pos = data.position_smooth ?? data.position;
        const regPx = { x: pos.x_px, y: pos.y_px };

        // Update node history
        nodeHistory.current.push(data.nodeId);
        if (nodeHistory.current.length > NODE_HISTORY_SIZE) {
          nodeHistory.current.shift();
        }

        // Store surveyed positions
        const topCandidate = data.topCandidates.find(c => c.nodeId === data.nodeId);
        if (topCandidate) {
          nodePositions.current[data.nodeId] = {
            x: topCandidate.x_px,
            y: topCandidate.y_px,
          };
        }

        // Get consensus node
        const consensusNode = getConsensusNode();
        let targetPx = regPx;   // start with regressor position

        // Apply node pull only if consensus exists, confidence is high, and nearby
        if (consensusNode && data.confidence >= CONFIDENCE_THRESHOLD) {
          const nodePos = nodePositions.current[consensusNode];
          if (nodePos) {
            const d = dist2D(regPx, nodePos);
            if (d <= MAX_DIST_TO_NODE) {
              targetPx = {
                x: nodePos.x * NODE_SNAP_STRENGTH + regPx.x * (1 - NODE_SNAP_STRENGTH),
                y: nodePos.y * NODE_SNAP_STRENGTH + regPx.y * (1 - NODE_SNAP_STRENGTH),
              };
            }
          }
        }

        // Constrain to path / walk graph
        if (pathRef.current.length >= 2) {
          const snap = closestPointOnPath(targetPx, pathRef.current);
          if (snap && snap.dist <= PATH_LOCK_RADIUS) {
            segIndexRef.current = snap.segIndex;
            targetPx = snap.point;
          } else {
            targetPx = constrainToWalkGraph(targetPx);
          }
        } else {
          targetPx = constrainToWalkGraph(targetPx);
        }

        // Client-side EMA on target position
        if (!emaPos.current) {
          emaPos.current = targetPx;
        } else {
          emaPos.current = {
            x: emaPos.current.x * (1 - CLIENT_EMA_ALPHA) + targetPx.x * CLIENT_EMA_ALPHA,
            y: emaPos.current.y * (1 - CLIENT_EMA_ALPHA) + targetPx.y * CLIENT_EMA_ALPHA,
          };
        }

        rawPos.current = emaPos.current;

        if (!hasFixRef.current) {
          hasFixRef.current = true;
          smoothPos.current = emaPos.current;
          setStatus('✅ Location acquired');
        }

        setConfidence(data.confidence);
        setSignalQuality(data.signalQuality);
        setPredictedNode(data.nodeId);

      } catch {
        setApiError(true);
        setStatus('⚠️ API unreachable — retrying…');
      }
    }, SCAN_INTERVAL_MS);

    return () => clearInterval(timer);
  }, []);

  // ── Render loop ───────────────────────────────────────
  useEffect(() => {
    const timer = setInterval(() => {
      if (!rawPos.current) return;

      const filtered = {
        x: kfX.filter(rawPos.current.x),
        y: kfY.filter(rawPos.current.y),
      };

      const prev = smoothPos.current || filtered;
      const d = dist2D(filtered, prev);
      const lerp = d > 40 ? LERP_FAR : d > 15 ? LERP_MID : LERP_NEAR;

      smoothPos.current = {
        x: prev.x + (filtered.x - prev.x) * lerp,
        y: prev.y + (filtered.y - prev.y) * lerp,
      };

      const pos = { ...smoothPos.current };
      setUserPosition(pos);
      setGeoPosition(pixelToGeo(pos.x, pos.y));

      if (destPxRef.current && pathRef.current.length >= 2 && !arrivedRef.current) {
        const distPx = remainingPathDistancePx(pos, pathRef.current, segIndexRef.current);
        setDistanceText(formatDistance(distPx));
        if (dist2D(pos, destPxRef.current) <= ARRIVAL_RADIUS_PX) {
          arrivedRef.current = true;
          setHasArrived(true);
          setIsNavigating(false);
          setDistanceText('0 m');
          setStatus('🏁 Arrived!');
          Animated.parallel([
            Animated.spring(arrivalOpacity, { toValue: 1, useNativeDriver: true, tension: 60, friction: 8 }),
            Animated.spring(arrivalScale,   { toValue: 1, useNativeDriver: true, tension: 60, friction: 8 }),
          ]).start();
        }
      }
    }, RENDER_FPS_MS);
    return () => clearInterval(timer);
  }, []);

  // ── Path computation ──────────────────────────────────
  const computePath = useCallback(() => {
    if (!selectedDestination) { setPath([]); return; }
    const dest = FLOOR_DATA.destinations.find(d => d.id === selectedDestination);
    if (!dest) return;
    const origin    = rawPos.current || smoothPos.current;
    const startNode = nearestNode(origin);
    const endNode   = FLOOR_DATA.walkNodes.find(n => n.id === dest.nodeId);
    if (!startNode || !endNode) return;
    const ids = bfsPath(startNode.id, endNode.id);
    if (!ids.length) return;
    const coords = ids.map(id => {
      const n = FLOOR_DATA.walkNodes.find(n => n.id === id);
      return { x: n.x, y: n.y };
    });
    segIndexRef.current = 0;
    setPath(coords);
    if (!arrivedRef.current) setStatus(`🧭 Navigating to ${dest.name}`);
  }, [selectedDestination]);

  useEffect(() => {
    computePath();
    const reroute = setInterval(() => {
      if (selectedDestination && hasFixRef.current && !arrivedRef.current) computePath();
    }, 5000);
    return () => clearInterval(reroute);
  }, [computePath, selectedDestination]);

  const dismissArrival = () => {
    Animated.timing(arrivalOpacity, { toValue: 0, duration: 250, useNativeDriver: true })
      .start(() => {
        setHasArrived(false);
        setSelectedDestination(null);
        setPath([]);
        setDistanceText('');
        destPxRef.current  = null;
        arrivedRef.current = false;
        fetch(`${API_BASE}/smooth/${DEVICE_ID}`, { method: 'DELETE' }).catch(() => {});
      });
  };

  const confPct = Math.round(confidence * 100);
  const signalEmoji =
    signalQuality === 'strong' ? '🟢' :
    signalQuality === 'fair'   ? '🟡' :
    signalQuality === 'weak'   ? '🟠' : '🔴';
  const confColor =
    confPct >= 80 ? '#16A34A' :
    confPct >= 60 ? '#D97706' : '#DC2626';
  const confBg =
    confPct >= 80 ? '#F0FFF4' :
    confPct >= 60 ? '#FFFBEB' : '#FFF0F0';

  return (
    <View style={{ flex: 1, backgroundColor: '#f0f0f0' }}>
      {/* Destination Picker */}
      <View style={{
        position: 'absolute', top: 40, left: 10, right: 10, zIndex: 999,
        backgroundColor: '#fff', borderRadius: 14,
        elevation: 8, shadowColor: '#000', shadowOpacity: 0.15,
        shadowRadius: 8, shadowOffset: { width: 0, height: 3 },
      }}>
        <Picker
          selectedValue={selectedDestination}
          onValueChange={v => setSelectedDestination(v)}
          style={{ height: 50 }}
        >
          <Picker.Item label="📍 Select Destination" value={null} />
          {FLOOR_DATA.destinations.map(d => (
            <Picker.Item key={d.id} label={d.name} value={d.id} />
          ))}
        </Picker>
      </View>

      <GestureHandlerRootView style={{ flex: 1 }}>
        {!hasFixRef.current || !userPosition ? (
          <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', gap: 12 }}>
            <Text style={{ fontSize: 18, color: '#333' }}>📡 Acquiring indoor position…</Text>
            <Text style={{ fontSize: 13, color: '#888' }}>
              Ensure Bluetooth is on and you are near a beacon.
            </Text>
            <Text style={{ fontSize: 13, color: '#555' }}>
              Detected beacons: {beaconCount}
            </Text>
            {apiError && (
              <Text style={{ fontSize: 12, color: '#DC2626', marginTop: 4 }}>
                ⚠️ Cannot reach server at {API_BASE}
              </Text>
            )}
          </View>
        ) : (
          <FloorMap userPosition={userPosition} heading={heading} path={path} />
        )}

        {isNavigating && selectedDestination && distanceText !== '' && !hasArrived && (
          <View style={{
            position: 'absolute', top: 110, alignSelf: 'center',
            flexDirection: 'row', alignItems: 'center',
            backgroundColor: '#1A73E8', borderRadius: 24,
            paddingHorizontal: 20, paddingVertical: 10,
            elevation: 8, shadowColor: '#1A73E8', shadowOpacity: 0.45,
            shadowRadius: 10, shadowOffset: { width: 0, height: 4 }, gap: 8,
          }}>
            <Text style={{ fontSize: 18 }}>⬆️</Text>
            <Text style={{ color: '#fff', fontWeight: '800', fontSize: 22, letterSpacing: 0.5 }}>
              {distanceText}
            </Text>
            <Text style={{ color: 'rgba(255,255,255,0.8)', fontSize: 13, fontWeight: '500' }}>
              remaining
            </Text>
          </View>
        )}

        {hasArrived && (
          <Animated.View pointerEvents="box-none" style={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            justifyContent: 'center', alignItems: 'center',
            backgroundColor: 'rgba(0,0,0,0.45)', opacity: arrivalOpacity,
          }}>
            <Animated.View style={{
              backgroundColor: '#fff', borderRadius: 24, padding: 36,
              alignItems: 'center', width: '78%', elevation: 20,
              shadowColor: '#000', shadowOpacity: 0.25, shadowRadius: 20,
              shadowOffset: { width: 0, height: 8 },
              transform: [{ scale: arrivalScale }], gap: 8,
            }}>
              <View style={{
                width: 80, height: 80, borderRadius: 40,
                backgroundColor: '#34A853',
                justifyContent: 'center', alignItems: 'center', marginBottom: 6,
              }}>
                <Text style={{ fontSize: 38 }}>✓</Text>
              </View>
              <Text style={{ fontSize: 22, fontWeight: '800', color: '#1a1a1a', textAlign: 'center' }}>
                You have arrived!
              </Text>
              <Text style={{ fontSize: 15, color: '#555', textAlign: 'center', lineHeight: 22 }}>
                {FLOOR_DATA.destinations.find(d => d.id === selectedDestination)?.name ?? 'Destination'}
              </Text>
              <Text style={{ fontSize: 13, color: '#aaa', marginTop: 2 }}>0 m · You're here</Text>
              <TouchableOpacity onPress={dismissArrival} style={{
                marginTop: 18, backgroundColor: '#1A73E8', borderRadius: 50,
                paddingHorizontal: 40, paddingVertical: 14,
                width: '100%', alignItems: 'center',
              }}>
                <Text style={{ color: '#fff', fontWeight: '700', fontSize: 15 }}>Done</Text>
              </TouchableOpacity>
            </Animated.View>
          </Animated.View>
        )}

        {/* Status Bar */}
        <View style={{
          position: 'absolute', bottom: 20, left: 10, right: 10,
          backgroundColor: '#fff', borderRadius: 14, padding: 14,
          elevation: 6, shadowColor: '#000', shadowOpacity: 0.10,
          shadowRadius: 8, shadowOffset: { width: 0, height: 2 }, gap: 4,
        }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ fontWeight: '700', fontSize: 14, color: '#1a1a1a' }}>{status}</Text>
            <Text style={{ fontSize: 12 }}>{signalEmoji} {signalQuality} · {beaconCount}B</Text>
          </View>

          {hasFixRef.current && predictedNode && (
            <View style={{
              flexDirection: 'row', alignItems: 'center', gap: 8,
              backgroundColor: confBg, borderRadius: 8,
              paddingHorizontal: 10, paddingVertical: 6, marginTop: 2,
            }}>
              <Text style={{ fontSize: 12 }}>🤖</Text>
              <Text style={{ fontSize: 13, fontWeight: '600', color: confColor }}>
                Node {predictedNode} · {confPct}% confidence
              </Text>
            </View>
          )}

          {isNavigating && distanceText !== '' && (
            <View style={{
              flexDirection: 'row', alignItems: 'center', gap: 6,
              backgroundColor: '#F1F8FF', borderRadius: 8,
              paddingHorizontal: 10, paddingVertical: 6, marginTop: 2,
            }}>
              <Text style={{ fontSize: 13 }}>📏</Text>
              <Text style={{ fontSize: 13, fontWeight: '600', color: '#1A73E8' }}>
                {distanceText} to destination
              </Text>
            </View>
          )}

          <Text style={{ fontSize: 11, color: '#444', marginTop: 2 }}>
            Lat: {geoPosition?.latitude?.toFixed(7)}
            {'  '}Lng: {geoPosition?.longitude?.toFixed(7)}
          </Text>
          <Text style={{ fontSize: 11, color: '#666' }}>
            X: {userPosition?.x?.toFixed(1)} px{'  '}
            Y: {userPosition?.y?.toFixed(1)} px{'   '}
            Beacons: {beaconCount}
          </Text>
        </View>
      </GestureHandlerRootView>
    </View>
  );
}