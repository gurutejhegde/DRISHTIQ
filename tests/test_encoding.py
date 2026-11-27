"""Tests for Member 2: Content-Aware Encoding pipeline."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from drishtiq.encoding import encode_video, load_metadata, verify_alignment
from drishtiq.encoding.pipeline import crf_for


@pytest.fixture(scope="module")
def mini_video(tmp_path_factory) -> Path:
    """Create a 2-second 480x360 synthetic video for fast encoding tests."""
    tmp_dir = tmp_path_factory.mktemp("enc_video")
    path = tmp_dir / "mini.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (480, 360))
    assert out.isOpened()
    for i in range(48):
        frame = np.full((360, 480, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (60 + i * 6, 180), 30, (0, 200, 255), -1)
        out.write(frame)
    out.release()
    return path


def test_load_metadata_and_crf_allocation(tmp_path):
    report = {
        "chunks": [
            {
                "chunk_index": 0,
                "start_time_s": 0.0,
                "duration_s": 2.0,
                "combined_score": 0.1,
                "complexity": "low",
                "priority": "low",
            },
            {
                "chunk_index": 1,
                "start_time_s": 2.0,
                "duration_s": 2.0,
                "combined_score": 0.8,
                "complexity": "high",
                "priority": "high",
            },
        ]
    }
    meta_path = tmp_path / "analysis.json"
    meta_path.write_text(json.dumps(report), encoding="utf-8")

    chunks = load_metadata(meta_path, normalize=True)
    assert len(chunks) == 2
    assert chunks[0]["complexity"] == 0.0
    assert chunks[1]["complexity"] == 1.0
    # Higher complexity + priority must receive a lower CRF (more bits)
    assert crf_for(chunks[1], "360p") < crf_for(chunks[0], "360p")


def test_encode_video_and_alignment(mini_video, tmp_path):
    metadata = [
        {
            "chunk_index": 0,
            "start_time_s": 0.0,
            "duration_s": 2.0,
            "combined_score": 0.6,
            "priority": "high",
            "complexity": "high",
        }
    ]
    out_dir = tmp_path / "chunks"
    manifests = encode_video(
        str(mini_video),
        metadata,
        mode="both",
        out_dir=str(out_dir),
        preset="ultrafast",
        workers=2,
        compute_ssim=True,
    )

    for mode_name in ("conventional", "content_aware"):
        mpath = Path(manifests[mode_name])
        assert mpath.exists()
        assert (mpath.parent / "manifest.csv").exists()
        assert verify_alignment(str(mpath))

        data = json.loads(mpath.read_text(encoding="utf-8"))
        renditions = data["chunks"][0]["renditions"]
        assert [r["name"] for r in renditions] == ["240p", "360p"]
        assert all(r["size_bytes"] > 0 and r["ssim"] is not None for r in renditions)
