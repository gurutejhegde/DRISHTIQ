
"""Streaming simulation engine for DrishtiQ."""

import json
from pathlib import Path
from typing import Any

from drishtiq.streaming.selector import (
    select_budget_aware,
    select_conventional,
)


def load_manifest(manifest: str | Path | dict) -> dict:
    """Load a chunk manifest from JSON or an existing dictionary."""

    if isinstance(manifest, dict):
        return manifest

    path = Path(manifest)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def run_stream(
    mode: str,
    manifest: str | Path | dict,
    bandwidth: float | list[float],
    budget: int,
    initial_buffer_s: float = 0.0,
    viewer_focus: str = "balanced",
) -> dict[str, Any]:
    """Simulate adaptive streaming and return per-chunk performance logs.

    Args:
        mode: "conventional" or "budget_aware".
        manifest: Path to a Member 2 JSON manifest or its dictionary.
        bandwidth: Available network speed in kbps, or a per-chunk list.
        budget: Maximum data budget in bytes.
        initial_buffer_s: Playback buffer available before the first chunk.

    Returns:
        A dictionary containing chunk logs and streaming statistics.
    """

    if mode not in ("conventional", "budget_aware"):
        raise ValueError("Mode must be 'conventional' or 'budget_aware'.")

    if budget < 0:
        raise ValueError("Budget cannot be negative.")

    if initial_buffer_s < 0:
        raise ValueError("Initial buffer cannot be negative.")

    if isinstance(bandwidth, list):
        if not bandwidth or any(speed <= 0 for speed in bandwidth):
            raise ValueError("All bandwidth values must be greater than zero.")
    elif bandwidth <= 0:
        raise ValueError("Bandwidth must be greater than zero.")

    data = load_manifest(manifest)
    chunks = data.get("chunks", [])

    if not chunks:
        raise ValueError("Manifest contains no chunks.")

    remaining_budget = budget
    buffer_s = initial_buffer_s
    logs = []
    total_downloaded = 0
    total_stall_s = 0.0
    estimated_bandwidth = None

    for index, chunk in enumerate(chunks):
        actual_bandwidth = (
            bandwidth[index % len(bandwidth)]
            if isinstance(bandwidth, list)
            else bandwidth
        )

        if estimated_bandwidth is None:
            estimated_bandwidth = float(actual_bandwidth)

        renditions = chunk.get("renditions", [])

        if not renditions:
            raise ValueError(f"Chunk {index} has no renditions.")

        if mode == "conventional":
            selected = select_conventional(
                renditions,
                bandwidth_kbps=estimated_bandwidth,
            )
        else:
            roi_box = chunk.get("roi_box") or {}
            selected = select_budget_aware(
                renditions,
                bandwidth_kbps=estimated_bandwidth,
                remaining_budget_bytes=remaining_budget,
                remaining_chunks=len(chunks) - index,
                priority=chunk.get("priority", "medium"),
                buffer_s=buffer_s if index > 0 else None,
                viewer_focus=viewer_focus,
                roi_label=roi_box.get("label") if isinstance(roi_box, dict) else None,
            )

        size_bytes = int(selected["size_bytes"])

        # Do not download a chunk that exceeds the remaining data budget.
        if size_bytes > remaining_budget:
            logs.append({
                "chunk_index": chunk.get("index", index),
                "status": "budget_exhausted",
                "remaining_budget_bytes": remaining_budget,
                "selected_rendition": selected.get("name"),
            })
            break

        # Simulate download time using the actual network bandwidth.
        download_time_s = (size_bytes * 8) / (
            float(actual_bandwidth) * 1000
        )

        duration_s = float(chunk["duration"])

        # The first chunk fills the playback buffer before playback starts.
        # Later chunks may cause rebuffering if they download too slowly.
        if index == 0:
            stall_s = 0.0
            buffer_s += duration_s
        else:
            stall_s = max(0.0, download_time_s - buffer_s)
            buffer_s = max(0.0, buffer_s - download_time_s)
            buffer_s += duration_s

        total_stall_s += stall_s
        remaining_budget -= size_bytes
        total_downloaded += size_bytes

        # Update throughput estimate using an exponential moving average.
        measured_throughput = (
            size_bytes * 8 / (max(download_time_s, 1e-6) * 1000)
        )

        estimated_bandwidth = (
            0.7 * estimated_bandwidth
            + 0.3 * measured_throughput
        )

        logs.append({
            "chunk_index": chunk.get("index", index),
            "status": "downloaded",
            "rendition": selected.get("name"),
            "path": selected.get("path", ""),
            "bitrate_kbps": float(selected["bitrate_kbps"]),
            "size_bytes": size_bytes,
            "ssim": float(selected.get("ssim", 0.0)),
            "roi_ssim": float(selected.get("roi_ssim") or selected.get("ssim") or 0.0),
            "download_time_s": round(download_time_s, 3),
            "buffer_s": round(buffer_s, 3),
            "stall_s": round(stall_s, 3),
            "remaining_budget_bytes": remaining_budget,
            "estimated_bandwidth_kbps": round(estimated_bandwidth, 2),
        })

    downloaded_entries = [entry for entry in logs if entry["status"] == "downloaded"]
    downloaded_count = len(downloaded_entries)
    avg_buffer = sum(e["buffer_s"] for e in downloaded_entries) / max(downloaded_count, 1)
    avg_ssim = sum(e["ssim"] for e in downloaded_entries) / max(downloaded_count, 1)
    avg_roi_ssim = sum(e["roi_ssim"] for e in downloaded_entries) / max(downloaded_count, 1)

    return {
        "mode": mode,
        "viewer_focus": viewer_focus,
        "total_chunks": len(chunks),
        "downloaded_chunks": downloaded_count,
        "total_downloaded_bytes": total_downloaded,
        "remaining_budget_bytes": remaining_budget,
        "total_stall_s": round(total_stall_s, 3),
        "average_buffer_s": round(avg_buffer, 3),
        "average_ssim": round(avg_ssim, 5),
        "average_roi_ssim": round(avg_roi_ssim, 5),
        "logs": logs,
    }


def total_downloaded_chunks(logs: list[dict]) -> int:
    """Count successfully downloaded chunks."""

    return sum(1 for entry in logs if entry["status"] == "downloaded")
