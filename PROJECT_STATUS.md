# DRISHTIQ — Project Status, Architecture & Next Steps (AI + Team Handoff)

> **Purpose of this document:** Give both team members and any AI coding assistant an exact, truthful snapshot of what DRISHTIQ has built and improved on the path to `main`, and what must be built next without drifting from the core goal.

---

## 1. North-Star Goal (Do Not Drift)

From [PRINCIPLES.md](PRINCIPLES.md) and [README.md](README.md):

> **Develop an intelligent video streaming system that identifies visually important regions (ROIs), preserves their quality while reducing quality in less important background regions, and intelligently allocates available bandwidth to deliver a better viewing experience with minimal buffering (and lower/efficient data use as a secondary benefit).**

**What success looks like in a live demo:**
1. At the **same throttled bandwidth** (and especially during a live drop like `3 Mbps → 300 kbps`), DRISHTIQ keeps playback smooth and keeps the important region (face/subject/text/action) noticeably sharper than Conventional uniform ABR.
2. At **healthy bandwidth** (`3 Mbps`), DRISHTIQ must look **at least as crisp and clean** as Conventional (never blurry on high-res rungs) while still using fewer bytes via smart background compression.
3. Quality adaptation, buffer state, and ROI metrics update visibly and verifiably on screen.

---

## 2. Repo Conventions & Verification Gates (Read Before Editing)

- **Package manager:** `uv` only (Python `3.12.16` pinned in `.python-version` and `uv.lock`). Never use `pip install`.
- **Strict data contracts:** Shared Pydantic models live in `src/drishtiq/core/schemas.py`. Avoid passing loose ad-hoc structures across module boundaries.
- **Keep code concise:** Follow `PRINCIPLES.md` — small, focused functions without boilerplate wrappers. Keep `ruff` happy (`line-length = 100`, rules `E, F, I, W` in `pyproject.toml`).
- **Required checks before pushing:**
  ```bash
  uv run --no-sync ruff check --no-cache .
  uv run --no-sync pytest -v
  ```
- **Branch note:** `design/ui-prototype` holds an experimental UI styling prototype separate from current working demo code.

---

## 3. What Was Added & Improved So Far

### Member 1 — Content Analysis & Saliency (`src/drishtiq/analysis/analyzer.py`, `src/drishtiq/core/schemas.py`)
- Replaced basic chunk-level heuristics with a **multi-cue spatial saliency map** (`compute_saliency_map` at `160x90`):
  - `40%` CIE LAB color contrast + `35%` temporal motion diff + `25%` Sobel edge density, weighted by a gentle center prior (`self._center_prior`).
- Added tight ROI bounding-box extraction (`detect_rois`) combining OpenCV Haar face detection (when available in build) with thresholded morphological saliency regions (`BoundingBox` in `schemas.py`).
- Added a `24x14` normalized **`saliency_heatmap`** grid per `ChunkAnalysis` in `schemas.py` and exported it in `data/manifests/*_analysis.{json,csv}`.
- Pre-generated updated manifests in `data/manifests/avengers_15s_analysis.{json,csv}`.

### Member 2 — Spatial ROI Encoding & Measurement (`src/drishtiq/encoding/pipeline.py`, `scripts/compare.py`)
- Expanded ladder in `LADDER` (`144p`, `240p`, `360p`, `480p`, `720p`, `1080p`) so low-bandwidth drops (`300 kbps`) have realistic rungs while maintaining `aac` audio (`-b:a 64k`).
- Implemented real spatial ROI encoding (`encode_chunk`):
  - **Even-pixel-aligned ROI crop & overlay** (`trunc(.../2)*2`) to eliminate `yuv420p` chroma alignment seams that previously caused random `roi_ssim` drops.
  - **Resolution-scaled background pre-filter** (`_bg_blur_sigma`: `0.5` at `>=720p`, `1.0` at `360p–480p`, `1.3` at `<=240p`) so HD streams remain visually sharp instead of looking smeared.
  - **Bit-focused ROI quantization**: `BASE_CRF` (`21–24`), `BG_CRF_PENALTY = 2`, strong `ROI_QOFFSET = "-0.50"` via FFmpeg `addroi`, and tighter VBV (`CA_MAXRATE_FACTOR = 0.72`, `bufsize = mr // 2`) so DRISHTIQ rungs stay ~15–35% smaller in bitrate than Conventional rungs.
- Added common display-resolution quality evaluation in `measure_ssim()` measuring **both full-frame `ssim` and cropped `roi_ssim`**.
- **Verified benchmark on rebuilt `avengers_15s.mp4` chunks (chunks 0, 4, 6):**
  - Every rung `144p–720p` achieved higher `roi_ssim` than Conventional (e.g., Chunk 0 `360p`: Conventional `689.8 kbps` / `0.9872` vs DRISHTIQ `578.8 kbps` / `0.9915`; Chunk 4 `720p`: Conventional `1431.8 kbps` / `0.9906` vs DRISHTIQ `1219.1 kbps` / `0.9964`).
  - Batch trace (`3000 -> 300 -> 3000 kbps` via `run_stream`): Conventional (`2.40 MB`, `roi_ssim = 0.98062`) vs DRISHTIQ (`2.15 MB`, `roi_ssim = 0.99105`).

### Member 3 — Viewer-Aware Adaptive Streaming (`src/drishtiq/streaming/selector.py`, `engine.py`, `report.py`)
- Updated `select_budget_aware()` to optimize ROI clarity under bandwidth + secondary budget constraints:
  - Uses `safety_factor = 0.78`, dropping to `0.60` when `buffer_s < 2.0` to prevent rebuffer stalls.
  - Supports `viewer_focus` (`people`, `action`, `text`, `balanced`).
- Extended stream simulation logs in `engine.py` with chunk file `"path"` and `"roi_ssim"`.

### Member 4 — Streamlit Frontend & Live Dual Player (`src/drishtiq/demo/`)
Built an end-to-end visual verification & presentation app across 4 files:
- `src/drishtiq/demo/helpers.py`: Video discovery (`discover_videos`), cached pipeline orchestration (`ensure_encoded_assets`), CORS chunk HTTP server (`start_chunk_server` on port `8502`), stream stitching (`stitch_stream_video`), and zoomed ROI/background visual extraction (`extract_visual_inspection`).
- `src/drishtiq/demo/app.py`: Streamlit 1.67 dashboard (`st.iframe`, `width="stretch"`) with 3 tabs:
  1. `1. Live Stream & Network Drop`: Live split-screen player + static zoomed ROI vs Background inspector.
  2. `2. Comparison & Proof`: Stitched Conventional vs DRISHTIQ comparison, KPI deltas, Plotly ROI SSIM & Bitrate charts.
  3. `3. How It Works & Raw Data`: Pipeline breakdown, streaming decision log table, and raw JSON manifests.
- `src/drishtiq/demo/live_player.py` & `src/drishtiq/demo/live_player.html`:
  - **Iframe-safe blob chunk loader** using parent host resolution (`st.context.headers["Host"]` + `window.parent.location.hostname`) to prevent `about://` CORS black screens.
  - **A/B double-buffered `<video>` elements (`cVidA`/`cVidB`, `pVidA`/`pVidB`)** with `preloadStandby()` and `25 ms` simulation ticks to eliminate single-element `.src`-swap micro-stuttering between 2s chunks.
  - **DRISHTIQ-only audio** (`pVid` unmutes on Play or `[M]`, `cVid` stays muted to prevent echo).
  - **DRISHTIQ-only overlay** (`drawOverlay` restricted to `pState.mode === "prop"` so Conventional ABR stays untouched).
  - **Live bandwidth responsiveness**: Reduced buffer lookahead cap to 2 chunks, added continuous EWMA (`25 ms` tick), fixed `selectProp` byte penalty (`0.0002` / `0.0008`) so high-BW playback chooses `720p` instead of self-downgrading to `480p`, and added mid-chunk smart abandonment on DRISHTIQ during severe drops.

### Tests Added/Passing
- `tests/test_analysis.py`, `tests/test_encoding.py`, `tests/test_streaming.py`, `tests/test_demo.py`: **16/16 passing** (`pytest`), `ruff` clean, Node JS syntax verified, Streamlit 1.67 headless agent-API check passing.

---

## 4. What Needs Improvement Next (Prioritized Roadmap)

### P0 — ROI Detection & Tracking Accuracy (`src/drishtiq/analysis/analyzer.py`)
**Gap:** Classical LAB contrast + Sobel + optical diff + optional Haar cascade is fast, but can still miss off-center faces in dark/stylized movie scenes, draw static per-chunk rectangles across 2s when subjects move fast, and `viewer_focus` currently adjusts ABR weights rather than steering **which visual region gets protected**.
**Next actions:**
1. Upgrade `detect_rois()` with reliable lightweight semantic detectors (e.g., OpenCV YuNet / MediaPipe face+person detection and text-region detection alongside motion/saliency cues) without slowing chunk throughput excessively.
2. Add **multi-ROI + feathered/temporal tracking** across frames within each 2s chunk (or sub-second smoothing) so moving subjects remain inside the sharp ROI throughout the chunk.
3. Connect `viewer_focus` (`people` / `action` / `text` / `balanced`) directly into Member 1 ROI ranking so user focus actually shifts which region `encode_chunk` protects.
4. Visually audit ROI boxes and heatmaps on `avengers_30s.mp4` and `avengers_doomsday_full.mp4`.

### P1 — Frontend Controls: Bitrate, Buffer & Live Showcasing (`src/drishtiq/demo/live_player.html`, `live_player.py`, `app.py`)
**Gap:** Several simulation parameters are currently hardcoded in `live_player.html`, and presenters need richer live controls during demos to isolate and prove DRISHTIQ's advantage cleanly.
**Next actions:**
1. **Bitrate & Ladder Controls:**
   - Expose **Max Bitrate Cap / Manual Rung Lock** (e.g., lock *both* players to `360p` or a fixed `400 kbps` ceiling so viewers can visually compare Conventional vs DRISHTIQ at identical low bitrate, side-by-side).
   - Make `Drop [D]` and `Recover [R]` target kbps customizable in the UI, plus add a **1-click Automated Demo Scenario** (e.g., `3000 kbps → hold 4s → drop 300 kbps for 6s → recover 3000 kbps`) synced cleanly with both players.
2. **Buffer & ABR Controls:**
   - Expose **Buffer Target (`1–6 chunks` / `2–12 s`)**, **Low-Buffer Threshold (`buffer_s`)**, **ABR Adaptation Speed / Abandonment toggle**, and live **Download Queue / Buffer Bar** telemetry inside the player UI instead of hardcoding `queuedCount >= 2` in JS.
3. **Showcase & Playback UX Controls:**
   - Add **Play/Pause toggle `[Space]`**, **Chunk Scrubber / Step Back & Replay**, and **Drift Sync Guard** between Conventional and DRISHTIQ timelines when neither is stalled.
   - Add a **Live ROI vs Background Loupe / Zoom Toggle** (or split slider / synchronized freeze-frame comparison) right inside Tab 1, plus a **Separate ROI Box Only vs Full Saliency Heatmap** toggle.
   - Show a live **Comparison Delta Banner** during playback (`+ROI SSIM gain`, `Data saved %`, `Rebuffers avoided`).
   - *(Aligns with `README.md` Idea #3)* Add an optional toggleable **Personalization / Overlay Layer** (e.g., HD subtitles or graphic overlay) that adapts/sheds cleanly before degrading core ROI video quality under severe drops.

### P2 — Encoding Refinement & Cache Versioning (`src/drishtiq/encoding/pipeline.py`, `src/drishtiq/demo/helpers.py`, `app.py`)
**Gap:**
1. `encode_chunk` currently uses only `_primary_roi()` (1 sharp rectangle pasted over softened background via `overlay`), which can leave a faint seam in flat gradients and ignores secondary important regions (e.g., two people talking, or speaker + text).
2. Pre-cached manifests (like `avengers_30s.mp4`) don't automatically know when `pipeline.py` encoding settings change — presenters have to remember to click rebuild.
3. Tab 3 text in `app.py` (`line 285`) still states `qoffset=-1/3` instead of `-0.50`, and `selectProp` in `live_player.html` vs `select_budget_aware` in `selector.py` should share consistent utility coefficients.

**Next actions:**
1. Support **multi-ROI (`addroi` multiple boxes + soft edge feathering)** on the background/foreground mask so transitions are invisible to the viewer while protecting up to 2–3 ROIs.
2. Add a `pipeline_version` stamp to analysis + chunk manifests so `helpers.py` / `app.py` detects outdated cached chunks and prompts/rebuilds cleanly.
3. Keep `selector.py` and `live_player.html` ABR logic aligned and update stale UI copy in `app.py`.

### P3 — Full-Length Trailer Testing & Benchmark Suite (`scripts/compare.py`)
**Gap:** Most iterative testing so far focused on `avengers_15s.mp4`. Full trailers (`avengers_doomsday_full.mp4`) need timed benchmarks for chunk analysis, parallel encoding speed, chunk server memory/cleanup, and multi-trace evaluation (`ROI SSIM`, `VMAF` if available, `stalls`, `MB saved`).
**Next actions:**
1. Add a `--max-duration` / preview-window option or fast progress/resume feedback for full-length movies in `compare.py` and `app.py`.
2. Verify long-playback blob memory cleanup and smooth chunk transitions over 30–120s+ sessions in the live player.

---

## 5. Quick Start Commands

```bash
# Run full test & lint suite
uv run --no-sync ruff check --no-cache .
uv run --no-sync pytest -v

# Launch interactive visual demo
uv run --no-sync streamlit run src/drishtiq/demo/app.py

# Run 2-laptop / LAN presentation mode
uv run --no-sync streamlit run src/drishtiq/demo/app.py --server.address 0.0.0.0 --server.port 8501
```
