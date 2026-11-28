#!/usr/bin/env python3
"""
Member 2 - Content-Aware Encoding pipeline.

Input : source video + Member 1's per-chunk metadata (JSON/CSV/list/DataFrame)
Output: data/chunks/{mode}/{resolution}/chunk_###.mp4 + manifest.json (+ manifest.csv)

Modes
  conventional  : every chunk uses the SAME fixed bitrate per rung (the baseline)
  content_aware : every chunk gets its own CRF derived from complexity + priority
                  (hard / important chunks -> lower CRF -> more bits)

Public API (for Member 3 / 4):
  encode_video(path, metadata, mode="both", out_dir="data/chunks") -> manifest path(s)

CLI:
  uv run python -m drishtiq.encoding.pipeline --video input.mp4 --metadata analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. CONFIGURATION (tune these, document your choices in the report)
# ----------------------------------------------------------------------------
LADDER = [  # name, output height, nominal bitrate (kbps) for the conventional set
    {"name": "144p", "height": 144, "bitrate_kbps": 200},
    {"name": "240p", "height": 240, "bitrate_kbps": 400},
    {"name": "360p", "height": 360, "bitrate_kbps": 800},
    {"name": "480p", "height": 480, "bitrate_kbps": 1200},
    {"name": "720p", "height": 720, "bitrate_kbps": 2500},
    {"name": "1080p", "height": 1080, "bitrate_kbps": 5000},
]
# Base CRF per rung for the content-aware set (lower = better quality / more bytes)
BASE_CRF = {"144p": 24, "240p": 23, "360p": 22, "480p": 22, "720p": 21, "1080p": 21}
CRF_SWING = 3  # content-aware CRF = base +/- up to this many steps
CRF_MIN, CRF_MAX = 16, 34
W_COMPLEXITY, W_PRIORITY = 0.5, 0.5  # how the two scores combine
ROI_QOFFSET = "-0.50"  # strong negative QP offset concentrates bits into sharp ROI
BG_CRF_PENALTY = 2  # light CRF lift for non-ROI areas; addroi offsets ROI back down
CONV_MAXRATE_FACTOR = 1.2  # conventional: maxrate = bitrate * factor
CA_MAXRATE_FACTOR = 0.72  # keep content-aware noticeably leaner than conventional per rung
GOP_SECONDS = 2  # fixed GOP / forced keyframe interval


def _bg_blur_sigma(height: int) -> float:
    """Resolution-scaled background pre-filter (invisible at HD, bit-saving at low res)."""
    if height >= 720:
        return 0.5
    if height >= 360:
        return 1.0
    return 1.3

LABEL_TO_SCORE = {"low": 0.2, "medium": 0.5, "med": 0.5, "high": 0.8}

# Accepted column names in Member 1's output (edit if their names differ)
KEYS = {  # Member 1's DrishtiQ names come first; others kept as fallbacks
    "id": ["chunk_index", "chunk_id", "id", "index", "chunk"],
    "start": ["start_time_s", "start", "start_time", "start_s", "start_sec"],
    "duration": ["duration_s", "duration", "dur", "length"],
    "complexity": ["combined_score", "complexity_score", "visual_complexity", "complexity"],
    "priority": ["priority_score", "perceptual_priority", "priority"],
    "label": ["complexity", "class", "complexity_class", "label"],
}


# ----------------------------------------------------------------------------
# 2. SMALL HELPERS
# ----------------------------------------------------------------------------
def run(cmd: list[str]) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n{p.stderr[-1500:]}")
    return p


def check_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found. Install FFmpeg and add it to PATH.")


def probe_video(path: str) -> dict:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
    ).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return {
        "width": s["width"],
        "height": s["height"],
        "fps": fps,
        "duration": float(d["format"]["duration"]),
    }


def probe_chunk(path: Path) -> dict:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    ).stdout
    d = json.loads(out)
    return {
        "width": d["streams"][0]["width"],
        "height": d["streams"][0]["height"],
        "duration": float(d["format"]["duration"]),
    }


def _pick(row: dict, keys: list[str], default=None):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ----------------------------------------------------------------------------
# 3. LOAD + NORMALISE MEMBER 1's METADATA
# ----------------------------------------------------------------------------
def _score(value):
    """Number -> clamped 0-1.  'low'/'medium'/'high' -> 0.2/0.5/0.8.  Else None."""
    if value is None or value == "":
        return None
    try:
        return _clamp01(float(value))
    except (TypeError, ValueError):
        return LABEL_TO_SCORE.get(str(value).strip().lower())


def _primary_roi(roi_boxes) -> dict | None:
    """Pick the clearest normalized ROI bounding box (preferring faces over saliency)."""
    if not isinstance(roi_boxes, list) or not roi_boxes:
        return None
    valid = []
    for b in roi_boxes:
        if not isinstance(b, dict):
            continue
        x = max(0.0, min(0.9, float(b.get("x", 0.0))))
        y = max(0.0, min(0.9, float(b.get("y", 0.0))))
        w = max(0.0, min(1.0 - x, float(b.get("width", 0.0))))
        h = max(0.0, min(1.0 - y, float(b.get("height", 0.0))))
        if w >= 0.08 and h >= 0.08:
            valid.append({
                "x": round(x, 4),
                "y": round(y, 4),
                "width": round(w, 4),
                "height": round(h, 4),
                "label": str(b.get("label", "roi")),
            })
    if not valid:
        return None
    return max(valid, key=lambda b: (b["label"] == "face", b["width"] * b["height"]))


def load_metadata(metadata, segment_len: float = 2.0, normalize: bool = True) -> list[dict]:
    """
    Accepts a path (.json/.csv), a list of dicts, or a pandas DataFrame.
    Reads Member 1's DrishtiQ report: chunks[] with chunk_index, start_time_s,
    duration_s, combined_score (0-1), complexity/priority labels.

    normalize=True stretches the complexity scores to the full 0-1 range across the
    video, because Member 1's raw scores usually sit low (0.0-0.4).
    """
    if hasattr(metadata, "to_dict"):  # DataFrame
        rows = metadata.to_dict(orient="records")
    elif isinstance(metadata, (str, Path)):
        p = Path(metadata)
        if p.suffix.lower() == ".csv":
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):  # VideoAnalysisReport JSON
                data = data.get("chunks") or next(
                    (v for v in data.values() if isinstance(v, list)), []
                )
            rows = data
    else:
        rows = list(metadata)

    chunks = []
    for i, r in enumerate(rows):
        label = _pick(r, KEYS["label"])
        cx = _score(_pick(r, KEYS["complexity"]))
        pr = _score(_pick(r, KEYS["priority"]))
        cx = cx if cx is not None else 0.5
        pr = pr if pr is not None else cx
        chunks.append(
            {
                "index": i,
                "id": _pick(r, KEYS["id"], i),
                "start": float(_pick(r, KEYS["start"], i * segment_len)),
                "duration": float(_pick(r, KEYS["duration"], segment_len)),
                "complexity": cx,
                "priority": pr,
                "label": str(label) if label is not None else None,
                "primary_roi": _primary_roi(r.get("roi_boxes")),
            }
        )
    if not chunks:
        raise ValueError("No chunks found in metadata.")

    if normalize:
        lo = min(c["complexity"] for c in chunks)
        hi = max(c["complexity"] for c in chunks)
        if hi - lo > 1e-6:
            for c in chunks:
                c["complexity"] = round((c["complexity"] - lo) / (hi - lo), 4)
    return chunks


def default_video_from_metadata(metadata) -> str | None:
    """Member 1's JSON stores the analysed video path; use it if it exists here."""
    try:
        data = json.loads(Path(metadata).read_text(encoding="utf-8"))
        fp = data["metadata"]["file_path"]
        return fp if Path(fp).exists() else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 4. CONTENT -> ENCODER PARAMETERS
# ----------------------------------------------------------------------------
def chunk_score(c: dict) -> float:
    return W_COMPLEXITY * c["complexity"] + W_PRIORITY * c["priority"]


def crf_for(chunk: dict, rung_name: str) -> int:
    """High score -> lower CRF (more bits). score 0.5 -> base CRF."""
    offset = round((0.5 - chunk_score(chunk)) * 2 * CRF_SWING)
    return max(CRF_MIN, min(CRF_MAX, BASE_CRF[rung_name] + offset))


# ----------------------------------------------------------------------------
# 5. ENCODE ONE CHUNK / ONE RUNG
# ----------------------------------------------------------------------------
def encode_chunk(
    src: str, chunk: dict, rung: dict, mode: str, out_path: Path, fps: float, preset: str
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gop = max(1, round(fps * GOP_SECONDS))
    br = rung["bitrate_kbps"]
    vf = f"scale=-2:{rung['height']}:flags=lanczos"
    roi = chunk.get("primary_roi")
    roi_applied = False

    if mode == "conventional":
        crf = None
        rate = [
            "-b:v",
            f"{br}k",
            "-maxrate",
            f"{int(br * CONV_MAXRATE_FACTOR)}k",
            "-bufsize",
            f"{int(br * 2)}k",
        ]
    else:
        # Soften non-ROI background cleanly with even-pixel alignment + strong ROI QP offset
        base_crf = crf_for(chunk, rung["name"])
        crf = min(CRF_MAX, base_crf + BG_CRF_PENALTY) if roi else base_crf
        if roi:
            rx, ry, rw, rh = roi["x"], roi["y"], roi["width"], roi["height"]
            bg_sigma = _bg_blur_sigma(rung["height"])
            px_in, py_in = f"trunc(iw*{rx:.4f}/2)*2", f"trunc(ih*{ry:.4f}/2)*2"
            pw_in = f"max(8\\,trunc(iw*{rw:.4f}/2)*2)"
            ph_in = f"max(8\\,trunc(ih*{rh:.4f}/2)*2)"
            px_ov, py_ov = f"trunc(W*{rx:.4f}/2)*2", f"trunc(H*{ry:.4f}/2)*2"
            vf = (
                f"scale=-2:{rung['height']}:flags=lanczos[s];"
                f"[s]split[bg][fg];"
                f"[bg]gblur=sigma={bg_sigma}[bgb];"
                f"[fg]crop={pw_in}:{ph_in}:{px_in}:{py_in}[fgc];"
                f"[bgb][fgc]overlay={px_ov}:{py_ov},"
                f"addroi=x={px_in}:y={py_in}:w={pw_in}:h={ph_in}:qoffset={ROI_QOFFSET}"
            )
            roi_applied = True
        mr = int(br * CA_MAXRATE_FACTOR)
        bs = max(32, mr // 2)
        rate = ["-crf", str(crf), "-maxrate", f"{mr}k", "-bufsize", f"{bs}k"]

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{chunk['start']:.3f}",
        "-i",
        src,
        "-t",
        f"{chunk['duration']:.3f}",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-force_key_frames",
        f"expr:gte(t,n_forced*{GOP_SECONDS})",
        *rate,
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    if not out_path.exists():  # resume support: skip finished chunks
        run(cmd)
    return {"crf": crf, "roi_applied": roi_applied}


# ----------------------------------------------------------------------------
# 6. QUALITY MEASUREMENT (Common display-resolution Full-frame + ROI SSIM)
# ----------------------------------------------------------------------------
def measure_ssim(
    src: str, chunk: dict, enc_path: Path, ref_width: int, ref_height: int, roi: dict | None = None
) -> tuple[float | None, float | None]:
    """Compare encoded chunk against original source at common display resolution."""
    if roi:
        crop = (
            f"crop=max(8\\,iw*{roi['width']:.4f}):max(8\\,ih*{roi['height']:.4f})"
            f":iw*{roi['x']:.4f}:ih*{roi['y']:.4f}"
        )
        flt = (
            f"[0:v]scale={ref_width}:{ref_height}:flags=bicubic,setpts=PTS-STARTPTS,split[d1][d2];"
            f"[1:v]scale={ref_width}:{ref_height}:flags=bicubic,setpts=PTS-STARTPTS,split[r1][r2];"
            f"[d1][r1]ssim;[d2]{crop}[dc];[r2]{crop}[rc];[dc][rc]ssim"
        )
    else:
        flt = (
            f"[0:v]scale={ref_width}:{ref_height}:flags=bicubic,setpts=PTS-STARTPTS[d];"
            f"[1:v]scale={ref_width}:{ref_height}:flags=bicubic,setpts=PTS-STARTPTS[r];"
            f"[d][r]ssim"
        )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(enc_path),
        "-ss",
        f"{chunk['start']:.3f}",
        "-t",
        f"{chunk['duration']:.3f}",
        "-i",
        src,
        "-lavfi",
        flt,
        "-f",
        "null",
        "-",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if not roi:
        m = re.search(r"All:([0-9.]+)", p.stderr)
        return (float(m.group(1)) if m else None, None)

    matches = re.findall(r"Parsed_ssim_(\d+).*?All:([0-9.]+)", p.stderr)
    if len(matches) >= 2:
        ordered = sorted(matches, key=lambda pair: int(pair[0]))
        return float(ordered[0][1]), float(ordered[1][1])
    vals = re.findall(r"All:([0-9.]+)", p.stderr)
    return (float(vals[0]) if vals else None, float(vals[1]) if len(vals) > 1 else None)


# ----------------------------------------------------------------------------
# 7. MAIN ENCODE FUNCTION
# ----------------------------------------------------------------------------
def _encode_one_mode(src, chunks, mode, out_dir, info, rungs, preset, workers, do_ssim):
    mode_dir = Path(out_dir) / mode
    ref_h = rungs[-1]["height"]
    ref_w = max(2, int(round(info["width"] * ref_h / max(info["height"], 1) / 2)) * 2)
    jobs = []
    for c in chunks:
        for r in rungs:
            path = mode_dir / r["name"] / f"chunk_{c['index']:03d}.mp4"
            jobs.append((c, r, path))

    def work(job):
        c, r, path = job
        extra = encode_chunk(src, c, r, mode, path, info["fps"], preset)
        meta = probe_chunk(path)
        size = path.stat().st_size
        ssim, roi_ssim = (
            measure_ssim(src, c, path, ref_w, ref_h, c.get("primary_roi"))
            if do_ssim
            else (None, None)
        )
        return c["index"], {
            "name": r["name"],
            "height": meta["height"],
            "width": meta["width"],
            "path": str(path.relative_to(mode_dir)).replace("\\", "/"),
            "size_bytes": size,
            "bitrate_kbps": round(size * 8 / max(meta["duration"], 1e-6) / 1000, 1),
            "ssim": ssim,
            "roi_ssim": roi_ssim,
            "roi_applied": extra["roi_applied"],
            "crf": extra["crf"],
        }

    results: dict[int, list] = {c["index"]: [] for c in chunks}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            idx, rend = f.result()
            results[idx].append(rend)
            done += 1
            print(f"\r[{mode}] {done}/{len(jobs)} encodes", end="", flush=True)
    print()

    manifest = {
        "source": str(src),
        "mode": mode,
        "gop_seconds": GOP_SECONDS,
        "ladder": [{k: r[k] for k in ("name", "height", "bitrate_kbps")} for r in rungs],
        "chunks": [],
    }
    for c in chunks:
        rends = sorted(results[c["index"]], key=lambda x: x["height"])
        manifest["chunks"].append(
            {
                "index": c["index"],
                "id": c["id"],
                "start": c["start"],
                "duration": c["duration"],
                "complexity": c["complexity"],
                "priority": c["priority"],
                "label": c["label"],
                "roi_box": c.get("primary_roi"),
                "renditions": rends,
            }
        )

    mpath = mode_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with open(mode_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "chunk",
                "start",
                "duration",
                "complexity",
                "priority",
                "rendition",
                "crf",
                "roi_applied",
                "size_bytes",
                "bitrate_kbps",
                "ssim",
                "roi_ssim",
                "path",
            ]
        )
        for c in manifest["chunks"]:
            for r in c["renditions"]:
                w.writerow(
                    [
                        mode,
                        c["index"],
                        c["start"],
                        c["duration"],
                        c["complexity"],
                        c["priority"],
                        r["name"],
                        r["crf"],
                        r["roi_applied"],
                        r["size_bytes"],
                        r["bitrate_kbps"],
                        r["ssim"],
                        r["roi_ssim"],
                        r["path"],
                    ]
                )
    return str(mpath)


def encode_video(
    path: str,
    metadata,
    mode: str = "both",
    out_dir: str = "data/chunks",
    segment_len: float = 2.0,
    preset: str = "veryfast",
    workers: int = 2,
    compute_ssim: bool = True,
    normalize: bool = True,
):
    """
    Encode `path` into aligned multi-bitrate chunks.
    mode: "conventional" | "content_aware" | "both"
    Returns manifest path (str) or {"conventional": ..., "content_aware": ...}.
    """
    check_tools()
    info = probe_video(path)
    chunks = load_metadata(metadata, segment_len, normalize)

    # Clamp chunks to the real video length; drop empty tails
    valid = []
    for c in chunks:
        c["duration"] = min(c["duration"], info["duration"] - c["start"])
        if c["duration"] > 0.05:
            valid.append(c)
    chunks = valid

    # Never upscale: keep only rungs <= source height
    rungs = [r for r in LADDER if r["height"] <= info["height"]] or [LADDER[0]]

    modes = ["conventional", "content_aware"] if mode == "both" else [mode]
    out = {
        m: _encode_one_mode(path, chunks, m, out_dir, info, rungs, preset, workers, compute_ssim)
        for m in modes
    }
    return out if mode == "both" else out[mode]


# ----------------------------------------------------------------------------
# 8. VERIFY KEYFRAME ALIGNMENT (run before handing off to Member 3)
# ----------------------------------------------------------------------------
def verify_alignment(manifest_path: str) -> bool:
    mp = Path(manifest_path)
    man = json.loads(mp.read_text(encoding="utf-8"))
    ok = True
    for c in man["chunks"]:
        for r in c["renditions"]:
            f = mp.parent / r["path"]
            out = run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "packet=flags",
                    "-read_intervals",
                    "%+#1",
                    "-of",
                    "csv=p=0",
                    str(f),
                ]
            ).stdout.strip()
            if not out.startswith("K"):
                print(f"NOT starting with keyframe: {f}")
                ok = False
    print("Alignment OK" if ok else "Alignment PROBLEMS found")
    return ok


# ----------------------------------------------------------------------------
# 9. CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Content-aware encoding pipeline")
    ap.add_argument("--video", help="source video (default: path stored in metadata JSON)")
    ap.add_argument("--metadata", required=True, help="Member 1's *_analysis.json / .csv")
    ap.add_argument("--mode", default="both", choices=["conventional", "content_aware", "both"])
    ap.add_argument("--out", default="data/chunks")
    ap.add_argument("--segment-len", type=float, default=2.0)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-ssim", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    video = a.video or default_video_from_metadata(a.metadata)
    if not video:
        raise SystemExit("Give --video (could not find the video path inside the metadata).")
    res = encode_video(
        video,
        a.metadata,
        a.mode,
        a.out,
        a.segment_len,
        a.preset,
        a.workers,
        not a.skip_ssim,
        not a.no_normalize,
    )
    print(json.dumps(res, indent=2) if isinstance(res, dict) else res)
    if a.verify:
        for p in res.values() if isinstance(res, dict) else [res]:
            verify_alignment(p)


if __name__ == "__main__":
    main()
