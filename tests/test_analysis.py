"""Unit and integration tests for Member 1: Content Analysis."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from drishtiq.analysis.analyzer import ContentAnalyzer
from drishtiq.core.schemas import ComplexityLevel, VideoAnalysisReport


@pytest.fixture(scope="module")
def sample_synthetic_video(tmp_path_factory) -> Path:
    """Generate a 4-second synthetic test video with:
    - 2 seconds of dark, static frames (low complexity).
    - Hard scene cut at t = 2.0s.
    - 2 seconds of dynamic noise and high-contrast motion (high complexity).
    """
    tmp_dir = tmp_path_factory.mktemp("video_test")
    video_path = tmp_dir / "test_synth.mp4"

    fps = 30
    width, height = 640, 360
    total_frames = fps * 4  # 4 seconds
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert out.isOpened(), "Failed to open synthetic video writer"

    for i in range(total_frames):
        if i < total_frames // 2:
            # Segment 1: Low complexity, static dark frame with a tiny gray rectangle
            frame = np.full((height, width, 3), 30, dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (150, 150), (60, 60, 60), -1)
        else:
            # Segment 2: High complexity, high contrast moving texture & checkerboard noise
            frame = np.random.randint(40, 220, (height, width, 3), dtype=np.uint8)
            offset = (i * 15) % width
            cv2.circle(frame, (offset, height // 2), 60, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"High Detail Motion Frame {i}",
                (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        out.write(frame)

    out.release()
    return video_path


def test_metadata_extraction(sample_synthetic_video: Path):
    analyzer = ContentAnalyzer(chunk_duration_s=2.0)
    meta = analyzer.extract_metadata(sample_synthetic_video)

    assert meta.filename == "test_synth.mp4"
    assert meta.width == 640
    assert meta.height == 360
    assert meta.fps == 30.0
    assert meta.duration_s >= 3.9


def test_texture_and_motion_metrics():
    analyzer = ContentAnalyzer()

    # Low complexity test image
    flat_frame = np.full((180, 320), 40, dtype=np.uint8)
    flat_texture = analyzer.compute_texture_score(flat_frame)
    assert flat_texture < 0.1

    # High complexity textured image (noise/edges)
    noise_frame = np.random.randint(0, 256, (180, 320), dtype=np.uint8)
    high_texture = analyzer.compute_texture_score(noise_frame)
    assert high_texture > flat_texture

    # Motion calculation test
    no_motion = analyzer.compute_motion_score(flat_frame, flat_frame)
    assert no_motion == 0.0

    moved_frame = np.roll(noise_frame, shift=20, axis=1)
    detected_motion = analyzer.compute_motion_score(noise_frame, moved_frame)
    assert detected_motion > 0.05


def test_full_analysis_pipeline(sample_synthetic_video: Path, tmp_path: Path):
    analyzer = ContentAnalyzer(chunk_duration_s=2.0, sample_fps=5.0)
    report = analyzer.analyze(sample_synthetic_video)

    # 4 second video with 2.0s chunk duration should yield 2 chunks
    assert report.total_chunks == 2
    assert len(report.chunks) == 2

    chunk_0 = report.chunks[0]
    chunk_1 = report.chunks[1]

    assert chunk_0.chunk_index == 0
    assert chunk_0.start_time_s == 0.0
    assert chunk_0.end_time_s == 2.0

    assert chunk_1.chunk_index == 1
    assert chunk_1.start_time_s == 2.0
    assert chunk_1.end_time_s >= 3.9

    # Chunk 1 (high texture + movement) must have higher complexity than Chunk 0
    assert chunk_1.combined_score > chunk_0.combined_score
    assert chunk_1.complexity in [ComplexityLevel.MEDIUM, ComplexityLevel.HIGH]

    # Verify JSON export & import round-trip
    json_path = tmp_path / "report.json"
    report.to_json(json_path)
    assert json_path.exists()

    loaded = VideoAnalysisReport.from_json(json_path)
    assert loaded.total_chunks == report.total_chunks
    assert loaded.chunks[0].combined_score == report.chunks[0].combined_score

    # Verify CSV export
    csv_path = tmp_path / "report.csv"
    report.to_csv(csv_path)
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0
