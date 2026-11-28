"""Helper utilities for Member 4's DRISHTIQ visual demo & verification app."""

from __future__ import annotations

import functools
import http.server
import json
import socketserver
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from drishtiq.analysis import ContentAnalyzer
from drishtiq.encoding import encode_video

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        return  # keep Streamlit terminal clean


def start_chunk_server(root_dir: str | Path = "data/chunks", port: int = 8502) -> int:
    """Start a lightweight background HTTP server for live chunk streaming."""
    root = str(Path(root_dir).resolve())
    handler = functools.partial(_CORSHandler, directory=root)
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer(("0.0.0.0", port), handler)
    except OSError:
        httpd = socketserver.TCPServer(("0.0.0.0", 0), handler)
    actual_port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return actual_port


def discover_videos(raw_dir: str | Path = "data/raw") -> list[Path]:
    """Return sorted list of available input videos."""
    root = Path(raw_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def ensure_encoded_assets(
    video_path: str | Path,
    manifests_dir: str | Path = "data/manifests",
    chunks_root: str | Path = "data/chunks",
    chunk_duration_s: float = 2.0,
    preset: str = "veryfast",
    workers: int = 4,
) -> tuple[Path, Path, Path]:
    """Ensure analysis JSON and both conventional/content_aware manifests exist for a video."""
    vp = Path(video_path)
    stem = vp.stem
    analysis_json = Path(manifests_dir) / f"{stem}_analysis.json"
    analysis_csv = Path(manifests_dir) / f"{stem}_analysis.csv"
    out_dir = Path(chunks_root) / stem
    conv_manifest = out_dir / "conventional" / "manifest.json"
    ca_manifest = out_dir / "content_aware" / "manifest.json"

    if not analysis_json.exists():
        report = ContentAnalyzer(chunk_duration_s=chunk_duration_s).analyze(vp)
        report.to_json(analysis_json)
        report.to_csv(analysis_csv)

    if not conv_manifest.exists() or not ca_manifest.exists():
        encode_video(
            str(vp),
            str(analysis_json),
            mode="both",
            out_dir=str(out_dir),
            segment_len=chunk_duration_s,
            preset=preset,
            workers=workers,
            compute_ssim=True,
        )

    return analysis_json, conv_manifest, ca_manifest


def _has_audio(path: Path) -> bool:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return bool(out.stdout.strip())


def stitch_stream_video(
    stream_result: dict,
    manifest_path: str | Path,
    out_video: str | Path,
    target_width: int = 854,
) -> Path | None:
    """Concatenate downloaded stream chunks into one playable MP4 preserving chunk quality."""
    mp = Path(manifest_path)
    chunk_paths = [
        mp.parent / entry["path"]
        for entry in stream_result.get("logs", [])
        if entry.get("status") == "downloaded" and entry.get("path")
    ]
    chunk_paths = [p for p in chunk_paths if p.exists()]
    if not chunk_paths:
        return None

    out = Path(out_video)
    out.parent.mkdir(parents=True, exist_ok=True)
    with_audio = all(_has_audio(p) for p in chunk_paths)

    inputs: list[str] = []
    filters: list[str] = []
    concat_inputs: list[str] = []
    for idx, cp in enumerate(chunk_paths):
        inputs.extend(["-i", str(cp)])
        filters.append(f"[{idx}:v]scale={target_width}:-2:flags=bilinear,setsar=1[v{idx}]")
        concat_inputs.append(f"[v{idx}][{idx}:a]" if with_audio else f"[v{idx}]")

    a_flag = 1 if with_audio else 0
    concat_expr = f"{''.join(concat_inputs)}concat=n={len(chunk_paths)}:v=1:a={a_flag}[v]"
    if with_audio:
        concat_expr += "[a]"
    filter_complex = ";".join(filters) + ";" + concat_expr

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
    ]
    if with_audio:
        cmd.extend(["-map", "[a]", "-c:a", "aac", "-b:a", "96k"])
    cmd.extend([
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-movflags", "+faststart", str(out)
    ])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed stitching video:\n{proc.stderr[-1000:]}")
    return out


def _read_mid_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open chunk video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Failed reading frame from: {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _pick_bg_crop(
    frame: np.ndarray, roi: dict | None, crop_w: int = 180, crop_h: int = 120
) -> np.ndarray:
    """Pick a background corner patch farthest from the ROI center for visual comparison."""
    h, w = frame.shape[:2]
    cw, ch = min(crop_w, w // 2), min(crop_h, h // 2)
    if not roi:
        return frame[0:ch, 0:cw]
    rcx = roi["x"] + roi["width"] / 2.0
    rcy = roi["y"] + roi["height"] / 2.0
    x0 = w - cw if rcx < 0.5 else 0
    y0 = h - ch if rcy < 0.5 else 0
    return frame[y0 : y0 + ch, x0 : x0 + cw]


def _overlay_heatmap(frame_rgb: np.ndarray, heatmap: list[list[float]]) -> np.ndarray:
    """Blend Member 1's 2D saliency grid over an RGB frame using a thermal colormap."""
    if not heatmap or not heatmap[0]:
        return frame_rgb
    arr = np.clip(np.array(heatmap, dtype=np.float32), 0.0, 1.0)
    up = cv2.resize(arr, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    mask = np.clip((up - 0.2) * 1.25, 0.0, 0.65)[..., None]
    heat_bgr = cv2.applyColorMap((up * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = frame_rgb.astype(np.float32) * (1.0 - mask) + heat_rgb * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def extract_visual_inspection(
    conv_manifest_path: str | Path,
    ca_manifest_path: str | Path,
    conv_log_entry: dict,
    ca_log_entry: dict,
    chunk_index: int,
    draw_roi_box: bool = True,
    heatmap: list[list[float]] | None = None,
) -> dict:
    """Extract normalized side-by-side frames plus zoomed ROI and background crops."""
    conv_mp = Path(conv_manifest_path)
    ca_mp = Path(ca_manifest_path)

    ca_man = json.loads(ca_mp.read_text(encoding="utf-8"))
    chunk_meta = next((c for c in ca_man["chunks"] if c["index"] == chunk_index), {})
    roi = chunk_meta.get("roi_box")

    conv_file = conv_mp.parent / conv_log_entry["path"]
    ca_file = ca_mp.parent / ca_log_entry["path"]

    conv_rgb = _read_mid_frame(conv_file)
    ca_rgb = _read_mid_frame(ca_file)

    # Upscale lower-resolution frame to match the higher one (preserving genuine blur/artifacts)
    target_h = max(conv_rgb.shape[0], ca_rgb.shape[0], 360)
    target_w = int(round(target_h * conv_rgb.shape[1] / max(conv_rgb.shape[0], 1)))
    conv_scaled = cv2.resize(conv_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    ca_scaled = cv2.resize(ca_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if roi:
        x1 = max(0, int(roi["x"] * target_w))
        y1 = max(0, int(roi["y"] * target_h))
        x2 = min(target_w, int((roi["x"] + roi["width"]) * target_w))
        y2 = min(target_h, int((roi["y"] + roi["height"]) * target_h))
        conv_roi_crop = conv_scaled[y1:y2, x1:x2]
        ca_roi_crop = ca_scaled[y1:y2, x1:x2]
    else:
        conv_roi_crop = conv_scaled
        ca_roi_crop = ca_scaled

    conv_bg_crop = _pick_bg_crop(conv_scaled, roi)
    ca_bg_crop = _pick_bg_crop(ca_scaled, roi)

    ca_display = _overlay_heatmap(ca_scaled.copy(), heatmap) if heatmap else ca_scaled.copy()
    if draw_roi_box and roi:
        x1 = max(0, int(roi["x"] * target_w))
        y1 = max(0, int(roi["y"] * target_h))
        x2 = min(target_w - 1, int((roi["x"] + roi["width"]) * target_w))
        y2 = min(target_h - 1, int((roi["y"] + roi["height"]) * target_h))
        cv2.rectangle(ca_display, (x1, y1), (x2, y2), (229, 169, 60), 3)

    return {
        "roi_box": roi,
        "conv_frame": conv_scaled,
        "ca_frame": ca_display,
        "conv_roi_crop": conv_roi_crop,
        "ca_roi_crop": ca_roi_crop,
        "conv_bg_crop": conv_bg_crop,
        "ca_bg_crop": ca_bg_crop,
    }
