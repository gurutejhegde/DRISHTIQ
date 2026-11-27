
"""Tests for Member 3: Budget-Aware Adaptive Streaming."""

import pytest

from drishtiq.streaming.selector import (
    select_budget_aware,
    select_conventional,
)


@pytest.fixture
def sample_renditions():
    return [
        {
            "name": "240p",
            "bitrate_kbps": 400,
            "size_bytes": 100_000,
            "ssim": 0.95,
        },
        {
            "name": "360p",
            "bitrate_kbps": 800,
            "size_bytes": 180_000,
            "ssim": 0.97,
        },
        {
            "name": "480p",
            "bitrate_kbps": 1200,
            "size_bytes": 260_000,
            "ssim": 0.98,
        },
        {
            "name": "720p",
            "bitrate_kbps": 2500,
            "size_bytes": 550_000,
            "ssim": 0.99,
        },
    ]


def test_conventional_selects_highest_safe_quality(sample_renditions):
    selected = select_conventional(
        sample_renditions,
        bandwidth_kbps=2000,
    )

    assert selected["name"] == "480p"


def test_conventional_falls_back_to_lowest_quality(sample_renditions):
    selected = select_conventional(
        sample_renditions,
        bandwidth_kbps=200,
    )

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


def test_budget_aware_prefers_higher_quality_when_budget_allows(
    sample_renditions,
):
    selected = select_budget_aware(
        sample_renditions,
        bandwidth_kbps=4000,
        remaining_budget_bytes=600_000,
        remaining_chunks=1,
        priority="high",
    )

    assert selected["name"] == "720p"


def test_invalid_bandwidth_raises_error(sample_renditions):
    with pytest.raises(ValueError):
        select_conventional(sample_renditions, bandwidth_kbps=0)


def test_empty_renditions_raise_error():
    with pytest.raises(ValueError):
        select_conventional([], bandwidth_kbps=1000)
