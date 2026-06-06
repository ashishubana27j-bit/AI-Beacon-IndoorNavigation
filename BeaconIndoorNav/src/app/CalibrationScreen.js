/**
 * setupScreen.js — Fingerprint Calibration (Final)
 * ══════════════════════════════════════════════════════════════
 *
 * WHAT THIS FIXED vs the original
 * ─────────────────────────────────────────────────────────────
 * 1. signals format: saves { "minor": -70 } (flat int) ✓
 *    train_model.py accepts both flat and { median } — confirmed.
 *
 * 2. headingSin / headingCos added to every fingerprint.
 *    The ML model uses these as features. Without them, the
 *    model gets 0.0 / 1.0 defaults, which degrades accuracy
 *    whenever orientation matters (near walls, corners).
 *
 * 3. Export button: taps → shows the full JSON file path AND
 *    opens the Share sheet so you can AirDrop / email it to
 *    your Python training machine directly from the phone.
 *
 * 4. Format written is exactly what train_model.py expects:
 *    {
 *      nodeId:     "A",
 *      x:          120,          ← pixel x from FLOOR_DATA
 *      y:          340,          ← pixel y from FLOOR_DATA
 *      headingSin: 0.99,         ← sin(headingRadians)
 *      headingCos: 0.01,         ← cos(headingRadians)
 *      orientation: "north",     ← human label (not used by model)
 *      timestamp:  1718000000000,
 *      signals: {
 *        "1": -72,               ← beacon minor: median RSSI
 *        "2": -81,
 *        "3": -66
 *      }
 *    }
 *
 * 5. Live sample-count bar shows per-beacon sample counts so
 *    you can see exactly which beacons are building up data.
 *
 * SETUP — one thing to configure
 * ─────────────────────────────────────────────────────────────
 *    COLLECT_DURATION_MS  default 15 s — increase to 20 s for
 *                         noisy environments.
 */

import { Picker } from '@react-native-picker/picker';
import { useEffect, useRef, useState } from 'react';
import {
    Alert, DeviceEventEmitter, Platform,
    Pressable, ScrollView, Share, Text, View,
} from 'react-native';
import Beacons from 'react-native-beacons-manager';
import CompassHeading from 'react-native-compass-heading';
import RNFS from 'react-native-fs';

import { FLOOR_DATA } from '../../data/floorData';
import {
    clearFingerprintsFile,
    loadFingerprints,
    saveFingerprints,
} from '../../utils/fingerprintEngine';

/* ═══════════════════════════════════════════════════════════
   CONSTANTS
═══════════════════════════════════════════════════════════ */
const COLLECT_DURATION_MS  = 15000;   // 15 s per capture
const MIN_SAMPLES_REQUIRED = 8;       // per beacon — filters transient beacons
const MIN_BEACONS_REQUIRED = 3;       // minimum stable beacons to accept a scan

const ORIENTATIONS = ['north', 'east', 'south', 'west'];

const FILE_PATH = RNFS.DocumentDirectoryPath + '/fingerprints.json';

/* ═══════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════ */
function getOrientationName(heading) {
  if (heading >= 315 || heading < 45)  return 'north';
  if (heading >= 45  && heading < 135) return 'east';
  if (heading >= 135 && heading < 225) return 'south';
  return 'west';
}

function median(arr) {
  if (!arr.length) return -100;
  const s   = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 !== 0 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function orientationStatus(o, captured, isCollecting, currentOrientation) {
  if (captured.includes(o)) return 'done';
  if (isCollecting && currentOrientation === o) return 'active';
  return 'pending';
}

const STATUS_COLOR = { done: '#22c55e', active: '#f59e0b', pending: '#6b7280' };
const STATUS_ICON  = { done: '✅',      active: '🔄',      pending: '⬜' };

/* ═══════════════════════════════════════════════════════════
   COMPONENT
═══════════════════════════════════════════════════════════ */
export default function FingerprintCalibrationScreen() {

  const [heading,              setHeading]              = useState(0);
  const [liveSignals,          setLiveSignals]          = useState({});
  const [fingerprints,         setFingerprints]         = useState([]);
  const [isCollecting,         setIsCollecting]         = useState(false);
  const [selectedNode,         setSelectedNode]         = useState(FLOOR_DATA.walkNodes[0]?.id || 'A');
  const [debugMsg,             setDebugMsg]             = useState('');
  const [countdown,            setCountdown]            = useState(0);
  const [capturedOrientations, setCapturedOrientations] = useState([]);
  const [sampleCount,          setSampleCount]          = useState(0);
  const [perBeaconCounts,      setPerBeaconCounts]      = useState({});  // minor → count

  const headingRef     = useRef(0);
  const sampleRef      = useRef({});       // minor → rssi[]
  const collectingRef  = useRef(false);
  const countdownTimer = useRef(null);

  /* ── reset when node changes ── */
  useEffect(() => {
    setCapturedOrientations([]);
    setDebugMsg('');
  }, [selectedNode]);

  /* ══════════════════════════════════════════════════════
     INIT: beacons + compass
  ══════════════════════════════════════════════════════ */
  useEffect(() => {
    async function init() {
      if (Platform.OS === 'android') Beacons.detectIBeacons();
      else Beacons.requestAlwaysAuthorization();
      await Beacons.startRangingBeaconsInRegion({
        identifier: 'IndoorRegion',
        uuid: '2f234454cf6d4a0fadf2f4911ba9ffa6',
      });
    }
    init();

    CompassHeading.start(3, ({ heading: h }) => {
      headingRef.current = h;
      setHeading(Math.round(h));
    });

    const sub = DeviceEventEmitter.addListener('beaconsDidRange', data => {
      if (!data.beacons?.length) return;

      const live   = {};
      const sample = { ...sampleRef.current };

      data.beacons.forEach(b => {
        if (!b.rssi || b.rssi === 0) return;
        live[String(b.minor)] = b.rssi;

        if (collectingRef.current) {
          const key = String(b.minor);
          if (!sample[key]) sample[key] = [];
          sample[key].push(b.rssi);
        }
      });

      setLiveSignals(live);
      sampleRef.current = sample;

      if (collectingRef.current) {
        const total  = Object.values(sample).reduce((s, a) => s + a.length, 0);
        const counts = {};
        Object.entries(sample).forEach(([k, v]) => { counts[k] = v.length; });
        setSampleCount(total);
        setPerBeaconCounts(counts);
      }
    });

    return () => {
      sub.remove();
      CompassHeading.stop();
      Beacons.stopRangingBeaconsInRegion({
        identifier: 'IndoorRegion',
        uuid: '2f234454cf6d4a0fadf2f4911ba9ffa6',
      });
      clearInterval(countdownTimer.current);
    };
  }, []);

  /* ══════════════════════════════════════════════════════
     LOAD saved fingerprints on mount
  ══════════════════════════════════════════════════════ */
  useEffect(() => {
    (async () => {
      const data = await loadFingerprints();
      FLOOR_DATA.fingerprints = data;
      setFingerprints(data);
    })();
  }, []);

  /* ══════════════════════════════════════════════════════
     START COLLECTION
  ══════════════════════════════════════════════════════ */
  function startCollecting() {
    if (isCollecting) return;
    collectingRef.current = true;
    sampleRef.current     = {};
    setSampleCount(0);
    setPerBeaconCounts({});
    setIsCollecting(true);
    setDebugMsg('');
    setCountdown(Math.round(COLLECT_DURATION_MS / 1000));

    let remaining = Math.round(COLLECT_DURATION_MS / 1000);
    countdownTimer.current = setInterval(() => {
      remaining -= 1;
      setCountdown(remaining);
      if (remaining <= 0) clearInterval(countdownTimer.current);
    }, 1000);

    setTimeout(stopCollecting, COLLECT_DURATION_MS);
  }

  /* ══════════════════════════════════════════════════════
     STOP + SAVE
     ─────────────────────────────────────────────────────
     Writes the exact format train_model.py expects:
       signals:    { "minor": medianRssi }   ← flat integer
       headingSin: sin(heading_rad)           ← NEW — required by ML
       headingCos: cos(heading_rad)           ← NEW — required by ML
  ══════════════════════════════════════════════════════ */
  async function stopCollecting() {
    clearInterval(countdownTimer.current);
    collectingRef.current = false;
    setIsCollecting(false);
    setCountdown(0);

    /* ── Median per beacon, only if ≥ MIN_SAMPLES_REQUIRED ── */
    const signals = {};
    Object.entries(sampleRef.current).forEach(([minor, arr]) => {
      const clean = arr.filter(x => x !== 0 && x > -100);
      if (clean.length < MIN_SAMPLES_REQUIRED) return;
      signals[minor] = Math.round(median(clean));
    });

    const beaconCount = Object.keys(signals).length;

    if (beaconCount < MIN_BEACONS_REQUIRED) {
      setDebugMsg(
        `❌ Only ${beaconCount} stable beacon(s) detected — need ≥ ${MIN_BEACONS_REQUIRED}.\n` +
        `Stand still, check Bluetooth is on, and retry.`
      );
      return;
    }

    const node = FLOOR_DATA.walkNodes.find(n => n.id === selectedNode);
    if (!node) return;

    const orientation = getOrientationName(headingRef.current);

    /* ── sin/cos encoding of heading ──────────────────────────
       Why sin/cos instead of raw degrees?
       Degrees have a discontinuity at 0/360 — a model treating
       359° and 1° as far apart makes large errors near north.
       sin/cos wraps smoothly: sin(359°) ≈ sin(1°), no jump.
    ── */
    const headingRad = (headingRef.current * Math.PI) / 180;

    const fingerprint = {
      nodeId:      selectedNode,
      x:           node.x,           // pixel coords from floorData
      y:           node.y,
      headingSin:  parseFloat(Math.sin(headingRad).toFixed(6)),
      headingCos:  parseFloat(Math.cos(headingRad).toFixed(6)),
      orientation,                   // human label, kept for display
      timestamp:   Date.now(),
      signals,                       // { "minor": medianRssi }
    };

    /* ── Deduplication: replace same nodeId + orientation ── */
    const existing     = FLOOR_DATA.fingerprints || [];
    const deduplicated = existing.filter(
      fp => !(fp.nodeId === selectedNode && fp.orientation === orientation)
    );
    const updated = [...deduplicated, fingerprint];

    FLOOR_DATA.fingerprints = updated;
    setFingerprints(updated);
    await saveFingerprints(updated);

    /* ── Track orientation progress ── */
    setCapturedOrientations(prev =>
      prev.includes(orientation) ? prev : [...prev, orientation]
    );

    const totalSamples = Object.values(sampleRef.current).reduce((s, a) => s + a.length, 0);
    const remaining    = ORIENTATIONS.filter(
      o => ![...capturedOrientations, orientation].includes(o)
    );
    const nextMsg = remaining.length > 0
      ? `Next: face ${remaining[0].toUpperCase()} and capture again`
      : '🎉 All 4 orientations done for this node!';

    setDebugMsg(
      `✅ Saved (${orientation.toUpperCase()}) — ${beaconCount} beacons · ${totalSamples} samples\n` +
      `headingSin: ${fingerprint.headingSin.toFixed(3)}  headingCos: ${fingerprint.headingCos.toFixed(3)}\n` +
      nextMsg
    );
  }

  /* ══════════════════════════════════════════════════════
     EXPORT — share the JSON file via OS share sheet
     The file path is also shown so you can use adb pull
     or iTunes File Sharing to copy it to your PC.
  ══════════════════════════════════════════════════════ */
  async function exportFingerprints() {
    try {
      const exists = await RNFS.exists(FILE_PATH);
      if (!exists || fingerprints.length === 0) {
        Alert.alert('Nothing to export', 'Capture some fingerprints first.');
        return;
      }

      // Show the file path for adb pull / iTunes
      Alert.alert(
        '📤 Export fingerprints.json',
        `File saved at:\n\n${FILE_PATH}\n\nTap "Share" to send via AirDrop, email, or any app.`,
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Share',
            onPress: async () => {
              try {
                await Share.share({
                  title:   'fingerprints.json',
                  message: JSON.stringify(fingerprints, null, 2),
                  // On iOS, 'url' with a file:// path attaches the actual file
                  url:     `file://${FILE_PATH}`,
                });
              } catch (e) {
                // Fallback: share JSON as text
                await Share.share({
                  title:   'fingerprints.json',
                  message: JSON.stringify(fingerprints, null, 2),
                });
              }
            },
          },
        ]
      );
    } catch (e) {
      Alert.alert('Export error', e.message);
    }
  }

  /* ══════════════════════════════════════════════════════
     CLEAR
  ══════════════════════════════════════════════════════ */
  async function clearFingerprints() {
    Alert.alert(
      'Clear all fingerprints?',
      `This will delete all ${fingerprints.length} fingerprints. Cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear', style: 'destructive',
          onPress: async () => {
            FLOOR_DATA.fingerprints = [];
            setFingerprints([]);
            setCapturedOrientations([]);
            setDebugMsg('🗑️ All fingerprints cleared.');
            await clearFingerprintsFile();
          },
        },
      ]
    );
  }

  /* ══════════════════════════════════════════════════════
     DERIVED VALUES
  ══════════════════════════════════════════════════════ */
  const currentOrientation  = getOrientationName(heading);
  const nodeFingerprints    = fingerprints.filter(fp => fp.nodeId === selectedNode);
  const allOrientationsDone = ORIENTATIONS.every(o => capturedOrientations.includes(o));

  // Progress toward recommended dataset size
  const totalNodes    = FLOOR_DATA.walkNodes.length;
  const nodesWithAll4 = (() => {
    const grouped = {};
    fingerprints.forEach(fp => {
      if (!grouped[fp.nodeId]) grouped[fp.nodeId] = new Set();
      grouped[fp.nodeId].add(fp.orientation);
    });
    return Object.values(grouped).filter(s => s.size === 4).length;
  })();
  const progressPct = Math.round((nodesWithAll4 / Math.max(totalNodes, 1)) * 100);

  /* ══════════════════════════════════════════════════════
     RENDER
  ══════════════════════════════════════════════════════ */
  return (
    <ScrollView style={{ flex: 1, backgroundColor: '#0a0a0f' }}>

      {/* ── HEADER ── */}
      <View style={{ padding: 16, backgroundColor: '#13131a', borderBottomWidth: 1, borderBottomColor: '#1e1e2e' }}>
        <Text style={{ color: '#fff', fontSize: 20, fontWeight: '800', letterSpacing: -0.3 }}>
          📍 Fingerprint Calibration
        </Text>
        <Text style={{ color: '#6b6b8a', marginTop: 4, fontSize: 13, lineHeight: 19 }}>
          Stand still at each node. Capture one scan per direction (N/E/S/W).
          All 4 orientations × every node = best ML accuracy.
        </Text>
      </View>

      {/* ── DATASET PROGRESS ── */}
      <View style={{ marginHorizontal: 10, marginTop: 10, backgroundColor: '#13131a', borderRadius: 10, padding: 12, borderWidth: 1, borderColor: '#1e1e2e' }}>
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 }}>
          <Text style={{ color: '#ccc', fontWeight: '600', fontSize: 13 }}>Dataset Progress</Text>
          <Text style={{ color: '#5cefb0', fontWeight: '700', fontFamily: 'monospace' }}>
            {nodesWithAll4}/{totalNodes} nodes complete
          </Text>
        </View>
        {/* Progress bar */}
        <View style={{ height: 6, backgroundColor: '#1e1e2e', borderRadius: 3, overflow: 'hidden' }}>
          <View style={{ height: '100%', width: `${progressPct}%`, backgroundColor: '#5cefb0', borderRadius: 3 }} />
        </View>
        <Text style={{ color: '#6b6b8a', fontSize: 11, marginTop: 6 }}>
          {fingerprints.length} fingerprints total · Target: {totalNodes * 4} ({totalNodes} nodes × 4 orientations)
        </Text>
      </View>

      {/* ── NODE SELECT ── */}
      <View style={{ marginHorizontal: 10, marginTop: 8, backgroundColor: '#13131a', borderRadius: 10, padding: 12, borderWidth: 1, borderColor: '#1e1e2e' }}>
        <Text style={{ color: '#ccc', marginBottom: 6, fontWeight: '600', fontSize: 13 }}>Select Node</Text>
        <View style={{ backgroundColor: '#0f0f18', borderRadius: 8, borderWidth: 1, borderColor: '#1e1e2e' }}>
          <Picker
            selectedValue={selectedNode}
            onValueChange={setSelectedNode}
            style={{ color: '#fff' }}
            dropdownIconColor="#5cefb0"
          >
            {FLOOR_DATA.walkNodes.map(node => {
              // Show ✅ next to nodes with all 4 orientations done
              const done = fingerprints.filter(fp => fp.nodeId === node.id);
              const orientsDone = new Set(done.map(fp => fp.orientation)).size;
              const label = orientsDone === 4
                ? `✅ Node ${node.id}`
                : `Node ${node.id}  (${orientsDone}/4)`;
              return <Picker.Item key={node.id} label={label} value={node.id} />;
            })}
          </Picker>
        </View>
      </View>

      {/* ── ORIENTATION CHECKLIST ── */}
      <View style={{ marginHorizontal: 10, marginTop: 8, backgroundColor: '#13131a', borderRadius: 10, padding: 12, borderWidth: 1, borderColor: '#1e1e2e' }}>
        <Text style={{ color: '#ccc', fontWeight: '600', fontSize: 13, marginBottom: 10 }}>
          Orientation Progress — Node {selectedNode}
        </Text>
        <View style={{ flexDirection: 'row', justifyContent: 'space-around' }}>
          {ORIENTATIONS.map(o => {
            const st    = orientationStatus(o, capturedOrientations, isCollecting, currentOrientation);
            const isCur = currentOrientation === o;
            return (
              <View key={o} style={{ alignItems: 'center', gap: 4 }}>
                <Text style={{ fontSize: 22 }}>{STATUS_ICON[st]}</Text>
                <Text style={{
                  color: STATUS_COLOR[st],
                  fontWeight: isCur ? '800' : '500',
                  fontSize: isCur ? 14 : 12,
                  textTransform: 'uppercase',
                }}>
                  {o}
                </Text>
                {isCur && <Text style={{ color: '#f59e0b', fontSize: 10 }}>← YOU</Text>}
              </View>
            );
          })}
        </View>
      </View>

      {/* ── COMPASS + HEADING ── */}
      <View style={{ marginHorizontal: 10, marginTop: 8, backgroundColor: '#13131a', borderRadius: 10, padding: 12, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', borderWidth: 1, borderColor: '#1e1e2e' }}>
        <Text style={{ color: '#6b6b8a', fontSize: 13 }}>Compass heading</Text>
        <View style={{ alignItems: 'flex-end' }}>
          <Text style={{ color: '#fff', fontWeight: '700', fontFamily: 'monospace', fontSize: 15 }}>
            {heading}° — {currentOrientation.toUpperCase()}
          </Text>
          <Text style={{ color: '#6b6b8a', fontSize: 11, marginTop: 2 }}>
            sin: {Math.sin((heading * Math.PI) / 180).toFixed(3)}  cos: {Math.cos((heading * Math.PI) / 180).toFixed(3)}
          </Text>
        </View>
      </View>

      {/* ── LIVE RSSI ── */}
      <View style={{ marginHorizontal: 10, marginTop: 8, backgroundColor: '#13131a', borderRadius: 10, padding: 12, borderWidth: 1, borderColor: '#1e1e2e' }}>
        <Text style={{ color: '#ccc', fontWeight: '600', fontSize: 13, marginBottom: 8 }}>
          Live Beacons ({Object.keys(liveSignals).length} visible)
        </Text>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8 }}>
          {Object.keys(liveSignals).length === 0 ? (
            <Text style={{ color: '#555', fontSize: 13 }}>No beacons detected…</Text>
          ) : (
            Object.entries(liveSignals)
              .sort((a, b) => b[1] - a[1])
              .map(([minor, rssi]) => {
                const sampleN = perBeaconCounts[minor] || 0;
                const bgColor = rssi > -70 ? '#14532d' : rssi > -85 ? '#713f12' : '#3f1919';
                const ready   = sampleN >= MIN_SAMPLES_REQUIRED;
                return (
                  <View key={minor} style={{
                    backgroundColor: bgColor,
                    borderRadius: 8,
                    paddingHorizontal: 10, paddingVertical: 6,
                    borderWidth: 1,
                    borderColor: ready ? '#22c55e' : '#444',
                    minWidth: 80, alignItems: 'center',
                  }}>
                    <Text style={{ color: '#fff', fontSize: 12, fontWeight: '700' }}>
                      B{minor}: {rssi} dBm
                    </Text>
                    {isCollecting && (
                      <Text style={{ color: ready ? '#86efac' : '#f59e0b', fontSize: 10, marginTop: 2 }}>
                        {sampleN} samples {ready ? '✓' : '…'}
                      </Text>
                    )}
                  </View>
                );
              })
          )}
        </View>
      </View>

      {/* ── CAPTURE BUTTON ── */}
      <Pressable
        onPress={startCollecting}
        disabled={isCollecting}
        style={{
          marginHorizontal: 10, marginTop: 12,
          padding: 18, borderRadius: 14,
          backgroundColor: isCollecting
            ? '#7f1d1d'
            : allOrientationsDone
            ? '#14532d'
            : '#1e3a8a',
          alignItems: 'center',
        }}
      >
        {isCollecting ? (
          <View style={{ alignItems: 'center', gap: 5 }}>
            <Text style={{ color: '#fff', fontSize: 17, fontWeight: '800' }}>
              🔴 Collecting… {countdown}s
            </Text>
            <Text style={{ color: '#fca5a5', fontSize: 13 }}>
              Stay completely still — {sampleCount} samples
            </Text>
            <Text style={{ color: '#fca5a5', fontSize: 11 }}>
              facing {currentOrientation.toUpperCase()} · need ≥{MIN_SAMPLES_REQUIRED} per beacon
            </Text>
          </View>
        ) : (
          <View style={{ alignItems: 'center', gap: 4 }}>
            <Text style={{ color: '#fff', fontSize: 17, fontWeight: '800' }}>
              {allOrientationsDone
                ? '🔄 Recapture any direction'
                : `📡 Capture — ${currentOrientation.toUpperCase()} (${capturedOrientations.length}/4)`}
            </Text>
            <Text style={{ color: 'rgba(255,255,255,0.65)', fontSize: 12 }}>
              15 seconds · stand still · face {currentOrientation.toUpperCase()}
            </Text>
          </View>
        )}
      </Pressable>

      {/* ── STATUS / DEBUG MESSAGE ── */}
      {debugMsg !== '' && (
        <View style={{ marginHorizontal: 10, marginTop: 8, backgroundColor: '#13131a', borderRadius: 10, padding: 12, borderWidth: 1, borderColor: '#1e1e2e' }}>
          <Text style={{ color: '#a3e635', fontSize: 13, lineHeight: 20, fontFamily: 'monospace' }}>
            {debugMsg}
          </Text>
        </View>
      )}

      {/* ── SAVED FINGERPRINTS FOR THIS NODE ── */}
      <View style={{ marginHorizontal: 10, marginTop: 8, backgroundColor: '#13131a', borderRadius: 10, padding: 12, borderWidth: 1, borderColor: '#1e1e2e' }}>
        <Text style={{ color: '#ccc', fontWeight: '600', fontSize: 13, marginBottom: 8 }}>
          Node {selectedNode} fingerprints ({nodeFingerprints.length}/4)
        </Text>
        {nodeFingerprints.length === 0 ? (
          <Text style={{ color: '#555', fontSize: 13 }}>None yet — capture all 4 orientations above.</Text>
        ) : (
          nodeFingerprints.map((fp, i) => (
            <View key={i} style={{
              flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
              paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: '#1e1e2e',
            }}>
              <Text style={{ color: '#86efac', fontSize: 13, fontWeight: '700', minWidth: 60 }}>
                {fp.orientation.toUpperCase()}
              </Text>
              <Text style={{ color: '#ccc', fontSize: 12 }}>
                {Object.keys(fp.signals).length} beacons
              </Text>
              <Text style={{ color: '#6b6b8a', fontSize: 11, fontFamily: 'monospace' }}>
                sin:{fp.headingSin?.toFixed(2) ?? 'n/a'}  cos:{fp.headingCos?.toFixed(2) ?? 'n/a'}
              </Text>
              <Text style={{ color: '#555', fontSize: 11 }}>
                {new Date(fp.timestamp).toLocaleTimeString()}
              </Text>
            </View>
          ))
        )}
      </View>

      {/* ── EXPORT BUTTON ── */}
      <Pressable
        onPress={exportFingerprints}
        style={{
          marginHorizontal: 10, marginTop: 10,
          padding: 14, borderRadius: 12,
          backgroundColor: '#0f172a',
          borderWidth: 1, borderColor: '#334155',
          alignItems: 'center', flexDirection: 'row', justifyContent: 'center', gap: 8,
        }}
      >
        <Text style={{ color: '#94a3b8', fontSize: 15 }}>📤</Text>
        <View>
          <Text style={{ color: '#e2e8f0', fontWeight: '700', fontSize: 14 }}>
            Export fingerprints.json
          </Text>
          <Text style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>
            {fingerprints.length} fingerprints — share to PC for training
          </Text>
        </View>
      </Pressable>

      {/* ── FILE PATH INFO ── */}
      <View style={{ marginHorizontal: 10, marginTop: 6, padding: 10, backgroundColor: '#0f0f18', borderRadius: 8 }}>
        <Text style={{ color: '#4b5563', fontSize: 10, fontFamily: 'monospace' }}>
          File: {FILE_PATH}
        </Text>
        <Text style={{ color: '#374151', fontSize: 10, marginTop: 3 }}>
          Android: adb pull {FILE_PATH} ./fingerprints.json
        </Text>
        <Text style={{ color: '#374151', fontSize: 10 }}>
          iOS: Xcode → Devices → App container → fingerprints.json
        </Text>
      </View>

      {/* ── CLEAR ALL ── */}
      <Pressable
        onPress={clearFingerprints}
        style={{
          marginHorizontal: 10, marginTop: 10, marginBottom: 40,
          padding: 14, borderRadius: 12,
          backgroundColor: '#1a0000',
          borderWidth: 1, borderColor: '#7f1d1d',
          alignItems: 'center',
        }}
      >
        <Text style={{ color: '#f87171', fontWeight: '600', fontSize: 14 }}>
          🗑️ Clear All Fingerprints ({fingerprints.length})
        </Text>
      </Pressable>

    </ScrollView>
  );
}
