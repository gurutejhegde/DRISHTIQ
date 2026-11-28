"""Integration tests for Member 4: Demo helpers and end-to-end pipeline."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from drishtiq.demo.helpers import (
    discover_videos,
    ensure_encoded_assets,
    extract_visual_inspection,
    stitch_stream_video,
)
from drishtiq.demo.live_player import build_live_player_html
from drishtiq.streaming import run_stream


@pytest.fixture(scope="module")
def demo_video(tmp_path_factory) -> Path:
    raw_dir = tmp_path_factory.mktemp("raw_demo")
    vp = raw_dir / "sample_clip.mp4"
    out = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 24, (480, 360))
    assert out.isOpened()
    for i in range(48):
        frame = np.full((360, 480, 3), 35, dtype=np.uint8)
        cv2.circle(frame, (80 + i * 5, 180), 40, (0, 220, 255), -1)
        out.write(frame)
    out.release()
    return vp


def test_discover_videos(demo_video):
    found = discover_videos(demo_video.parent)
    assert len(found) == 1
    assert found[0].name == "sample_clip.mp4"


def test_end_to_end_demo_pipeline(demo_video, tmp_path):
    analysis_json, conv_man, ca_man = ensure_encoded_assets(
        demo_video,
        manifests_dir=tmp_path / "manifests",
        chunks_root=tmp_path / "chunks",
        chunk_duration_s=2.0,
        preset="ultrafast",
        workers=2,
    )
    assert analysis_json.exists() and conv_man.exists() and ca_man.exists()

    conv_res = run_stream("conventional", conv_man, [800], 5_000_000)
    ca_res = run_stream("budget_aware", ca_man, [800], 5_000_000, viewer_focus="people")

    conv_vid = stitch_stream_video(conv_res, conv_man, tmp_path / "conv_out.mp4")
    ca_vid = stitch_stream_video(ca_res, ca_man, tmp_path / "ca_out.mp4")
    assert conv_vid is not None and conv_vid.exists() and conv_vid.stat().st_size > 0
    assert ca_vid is not None and ca_vid.exists() and ca_vid.stat().st_size > 0

    vis = extract_visual_inspection(
        conv_man,
        ca_man,
        conv_res["logs"][0],
        ca_res["logs"][0],
        chunk_index=0,
        heatmap=[[0.1, 0.9], [0.2, 0.8]],
    )
    assert vis["conv_frame"].shape == vis["ca_frame"].shape
    assert vis["conv_roi_crop"].size > 0 and vis["ca_roi_crop"].size > 0

    html = build_live_player_html("sample_clip", conv_man, ca_man, analysis_json, 8502)
    assert "Drop to 300 kbps" in html and "Saliency Heatmap" in html
    assert "btnAudio" in html and 'pState.mode !== "prop"' in html
    assert "cVidA" in html and "pVidB" in html and "preloadStandby" in html
