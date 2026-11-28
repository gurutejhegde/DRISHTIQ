"""Tests for Member 3: Budget-Aware Adaptive Streaming."""

import pytest

from drishtiq.streaming import (
    run_stream,
    save_report,
    select_budget_aware,
    select_conventional,
)


@pytest.fixture
def sample_renditions():
    return [
        {
            "name": "240p",
            "path": "240p/chunk_000.mp4",
            "bitrate_kbps": 400,
            "size_bytes": 100_000,
            "ssim": 0.95,
        },
        {
            "name": "360p",
            "path": "360p/chunk_000.mp4",
            "bitrate_kbps": 800,
            "size_bytes": 180_000,
            "ssim": 0.97,
        },
        {
            "name": "480p",
            "path": "480p/chunk_000.mp4",
            "bitrate_kbps": 1200,
            "size_bytes": 260_000,
            "ssim": 0.98,
        },
        {
            "name": "720p",
            "path": "720p/chunk_000.mp4",
            "bitrate_kbps": 2500,
            "size_bytes": 550_000,
            "ssim": 0.99,
        },
    ]


@pytest.fixture
def sample_manifest(sample_renditions):
    chunks = []
    for i in range(4):
        rends = [{**r, "path": f"{r['name']}/chunk_{i:03d}.mp4"} for r in sample_renditions]
        chunks.append({
            "index": i,
            "start": i * 2.0,
            "duration": 2.0,
            "complexity": 0.2 if i % 2 == 0 else 0.85,
            "priority": "low" if i % 2 == 0 else "high",
            "renditions": rends,
        })
    return {"mode": "test", "chunks": chunks}


def test_conventional_selects_highest_safe_quality(sample_renditions):
    selected = select_conventional(sample_renditions, bandwidth_kbps=2000)
    assert selected["name"] == "480p"


def test_conventional_falls_back_to_lowest_quality(sample_renditions):
    selected = select_conventional(sample_renditions, bandwidth_kbps=200)
    assert selected["name"] == "240p"


def test_budget_aware_respects_data_budget(sample_renditions):
    selected = select_budget_aware(
        sample_renditions,
        bandwidth_kbps=4000,
        remaining_budget_bytes=200_000,
        remaining_chunks=1,
        priority="medium",
    )
    assert selected["size_bytes"] <= 200_000


def test_budget_aware_prefers_higher_quality_when_budget_allows(sample_renditions):
    selected = select_budget_aware(
        sample_renditions,
        bandwidth_kbps=4000,
        remaining_budget_bytes=600_000,
        remaining_chunks=1,
        priority="high",
    )
    assert selected["name"] == "720p"


def test_budget_aware_low_buffer_tightens_selection(sample_renditions):
    # At 1500 kbps: 1500 * 0.85 = 1275 (fits 480p), while low buffer 1500 * 0.65 = 975 (360p).
    selected = select_budget_aware(
        sample_renditions,
        bandwidth_kbps=1500,
        remaining_budget_bytes=1_000_000,
        remaining_chunks=1,
        priority="high",
        buffer_s=0.8,
    )
    assert selected["name"] == "360p"


def test_invalid_bandwidth_raises_error(sample_renditions):
    with pytest.raises(ValueError):
        select_conventional(sample_renditions, bandwidth_kbps=0)


def test_empty_renditions_raise_error():
    with pytest.raises(ValueError):
        select_conventional([], bandwidth_kbps=1000)


def test_run_stream_and_save_report(sample_manifest, tmp_path):
    budget = 700_000
    conv = run_stream("conventional", sample_manifest, [2000, 1200, 800, 1500], budget)
    prop = run_stream("budget_aware", sample_manifest, [2000, 1200, 800, 1500], budget)

    assert prop["total_downloaded_bytes"] <= budget
    assert prop["downloaded_chunks"] >= conv["downloaded_chunks"]
    assert "average_ssim" in prop
    assert "average_roi_ssim" in prop
    assert prop["logs"][0]["path"].endswith("chunk_000.mp4")

    json_file, csv_file = save_report(prop, output_dir=tmp_path)
    assert json_file.exists() and csv_file.exists()


def test_viewer_focus_refines_roi_quality(sample_renditions):
    focused = [{**r, "roi_ssim": min(0.999, r["ssim"] + 0.02)} for r in sample_renditions]
    pick = select_budget_aware(
        focused,
        bandwidth_kbps=1600,
        remaining_budget_bytes=300_000,
        remaining_chunks=1,
        priority="medium",
        viewer_focus="people",
        roi_label="face",
    )
    assert pick["name"] == "480p"

