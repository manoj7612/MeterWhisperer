# MeterWhisperer — UI Integration Plan

This document captures the analysis of the proposed user interface at
`https://meterwhisp-dgcfaj9b.manus.space/` and proposes how to integrate it with
the existing Flask backend (`app.py` + `src/features.py` + the trained model).

> **Status:** Planning only. No code changes have been made.

---

## 1. What the UI actually is

`https://meterwhisp-dgcfaj9b.manus.space/` is a **React + TypeScript + Vite
single-page app** (source maps point to `client/src/pages/Home.tsx`). It is a
marketing/demo landing page titled *"The Meter Whisperer — Listen to the meter"*,
**not yet wired to any backend**.

Key facts observed from the bundled JavaScript:

- **It is fully simulated.** The only `fetch` calls are Vite asset preloads —
  there is no API call to any backend. The page shows the disclaimer:
  *"Browser demonstration only — no live accelerometer data connected."*
- **It renders 5 hardcoded "scenarios"** in an object (`ep`) that the user can
  switch between via buttons.
- Each scenario has: `label`, `severity`, `confidence`, `explanation`, `action`,
  `pattern` (an array of ~16 bar heights used only for a bar-chart visual), and
  `tone` (a colour).
- **The `pattern` is not real sensor data** — it is a hardcoded array such as
  `[18,17,18,17,...]` used only to draw bars.
- It has 4 "Signal Lab" view modes: `raw`, `processed`, `frequency`,
  `interpretation`.
- It has a roadmap: **NOW** (AI interface + demonstration) → **NEXT** (collect
  real data) → **TRAIN** → **CONNECT** (integrate ESP32 + accelerometer) →
  **FIELD TEST** → **REFINE**.

### UI scenarios vs. backend classes

| UI key | Label | Severity | Confidence | Backend class? |
|--------|-------|----------|------------|----------------|
| `normal` | No abnormal activity | `NORMAL` | `—` | ✅ `normal` |
| `tap` | Possible faucet / tap | `LOW` | `76%*` | ✅ `tap_water_leak` |
| `washing` | Possible washing machine | `MODERATE` | `81%*` | ✅ `washing_machine_leak` |
| `toilet` | Possible toilet leak | `HIGH` | `94%*` | ❌ none |
| `pipe` | Pipe / fitting anomaly | `MODERATE` | `68%*` | ❌ none |

The `*` on the confidence values marks them as simulated/placeholder.

---

## 2. Gap analysis (what needs to be bridged)

| # | Gap | Detail |
|---|-----|--------|
| 1 | **No network layer** | The UI never calls a backend; everything is static. |
| 2 | **Class-name mismatch** | UI uses `tap`/`washing`/`toilet`/`pipe`; backend uses `tap_water_leak`/`washing_machine_leak` and has no `toilet`/`pipe`. |
| 3 | **Data-shape mismatch** | UI has `pattern` bar heights; backend needs `{time_s, acc_x_m_s2, acc_y_m_s2, acc_z_m_s2}` arrays (≥200 samples). |
| 4 | **Confidence source** | UI confidence is hardcoded with a `*` placeholder; backend computes real `confidence_percent`. |
| 5 | **Extra UI fields** | UI shows `explanation` and `action` text; the backend currently returns only `note` (no per-class explanation/action). |
| 6 | **CORS** | Flask already has `CORS(app)` (open), so a separately-hosted frontend can call it — this part is already solved. |

---

## 3. Proposed integration architecture

Adopt a **3-phase approach**. Phase 1 is the smallest change that proves the
wiring end-to-end.

### Recommended deployment topology (for now)

Keep two processes connected over HTTP (CORS is already enabled):

```
React SPA (any static host / dev server)
        │  POST /api/predict
        ▼
Flask backend (127.0.0.1:5000)  →  model + src/features.py
```

Alternative (cleaner for production): **Flask serves the built frontend** — put
the Vite `dist/` output into the Flask project and serve it as static files.
This gives a single origin, no CORS configuration needed, and one
`python3 app.py` command to run everything.

### Data-flow design

The core problem is bridging UI "scenario" → real time-series → backend → rich
result. The cleanest approach is a **contract layer** that maps a scenario key to
a synthetic accelerometer waveform generated client-side (so the demo stays
interactive), then sends real arrays to the real backend:

```
UI scenario (normal/tap/washing/...)
   → waveform generator (client, produces time_s + acc_x/y/z arrays at ~100 Hz, 8 s)
   → POST /api/predict {time_s, acc_x_m_s2, acc_y_m_s2, acc_z_m_s2}
   → backend: features.py → model.predict_proba
              → {predicted_class, confidence_percent, severity, average_vibration_rms}
   → UI renders the result in the existing "AI interpretation" card
```

### Class mapping (contract)

```js
const UI_TO_BACKEND = {
  normal:  "normal",
  tap:     "tap_water_leak",
  washing: "washing_machine_leak",
  toilet:  null,   // not supported yet → retrain needed
  pipe:    null,   // not supported yet → retrain needed
};
```

The backend already returns the raw class (`predicted_class`) plus a display name
(`source`). Add a reverse mapping in the UI (backend class → UI scenario key) so
the UI can still colour/label the result correctly.


---

## 4. Concrete changes needed

### Backend (small, additive)

1. *(Optional, recommended)* Add per-class `explanation` and `action` text to the
   `/api/predict` response (or a lookup the UI uses), so the existing
   "explanation"/"action" UI fields can be filled from real predictions instead
   of hardcoded strings.
2. Add a `GET /api/classes` (or reuse `/api/health`) so the UI can discover
   supported classes dynamically instead of hardcoding the mapping.
3. *(Optional)* Add `POST /api/predict` support for `toilet`/`pipe` — **requires
   retraining** the model with those two extra classes (data does not exist yet,
   so this is Phase 3).
4. Everything else (CORS, validation, feature extraction) already exists and
   needs no change.

### Frontend (the actual integration work)

1. Add a configurable API base URL via `VITE_API_URL`
   (defaults to `http://127.0.0.1:5000`).
2. Add a small `api.ts` client with `getHealth()` and `predict(payload)` using
   `fetch`.
3. Add a client-side **waveform generator** that turns a scenario key into ~8 s
   of 100 Hz accelerometer data (deterministic per scenario: e.g. normal =
   low-amplitude noise; tap = short high-amplitude bursts; washing = cyclical
   20–40 Hz; etc.).
4. Replace the hardcoded `ep.confidence`/`ep.severity` with a real call to
   `/api/predict` when a scenario is selected (keep the static `pattern` only as
   the visual preview, or derive it from the generated waveform).
5. Map the backend response into the existing `demo-output` card:
   `predicted_class` → scenario key, `confidence_percent` → `Confidence`,
   `severity` → `Severity`, `source` → label.
6. Handle loading/error states (backend down → fall back to the current
   "simulated" disclaimer).

---

## 5. Phased rollout

- **Phase 1 — Prove the wiring (smallest change):** Add a single hidden "Live
  inference" call: pick a scenario, generate a synthetic waveform, call the real
  `/api/predict`, and show the real `confidence_percent`/`severity` in place of
  the `*` values. No model retraining. This validates CORS, data shape, and the
  class mapping.
- **Phase 2 — Real data:** Replace synthetic waveforms with the real recordings
  in `data/raw/*.csv` (serve them as JSON or let the UI upload/select a CSV),
  giving genuine end-to-end predictions on actual data.
- **Phase 3 — Full parity:** Add `toilet` and `pipe` classes by collecting
  labelled data, retraining the 5-class model, and update the backend + UI
  mapping so all 5 UI scenarios are backed by real inference.
- **Phase 4 — Live sensor (ESP32):** WebSocket/streaming from an ESP32 +
  accelerometer to the backend, then push results to the UI (replaces the
  simulation entirely).

---

## 6. Key risks / open questions

1. **Class mismatch (`toilet`/`pipe`):** These two UI scenarios cannot be backed
   by the current 3-class model. Decide whether to (a) hide/disable them,
   (b) map them to the nearest existing class, or (c) collect data and retrain.
2. **Waveform realism:** A synthetic client-side generator will not reproduce the
   exact distributions the model learned from `data/raw/`, so Phase-1 predictions
   may be unreliable — fine for wiring, but Phase 2 (real CSVs) is what actually
   validates correctness.
3. **Deployment/URL config:** Confirm whether the SPA will call `127.0.0.1:5000`
   directly (dev) or be served by Flask (prod). The API base URL must be
   injectable.
4. **`pattern` bar visualization:** It is cosmetic; decide if it should be
   derived from the generated waveform or kept static during integration.

---

## 7. Suggested next step

The most natural next step (when ready to make changes) is **Phase 1**:

- Draft the frontend `api.ts` client.
- Draft the waveform generator.
- Draft the scenario→class mapping.
- Add the minimal backend response additions (per-class `explanation`/`action`,
  optional `/api/classes`) needed to make the existing "Interactive AI
  demonstration" card show real predictions.

