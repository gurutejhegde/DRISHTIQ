
"""Rendition selection algorithms for DrishtiQ adaptive streaming."""

from typing import Any


def _get_bitrate(rendition: dict[str, Any]) -> float:
    return float(rendition["bitrate_kbps"])


def _get_size(rendition: dict[str, Any]) -> int:
    return int(rendition["size_bytes"])


def select_conventional(
    renditions: list[dict[str, Any]],
    bandwidth_kbps: float,
    safety_factor: float = 0.85,
) -> dict[str, Any]:
    """Select the highest bitrate that safely fits available bandwidth."""

    if not renditions:
        raise ValueError("At least one rendition is required.")

    if bandwidth_kbps <= 0:
        raise ValueError("Bandwidth must be greater than zero.")

    if not 0 < safety_factor <= 1:
        raise ValueError("Safety factor must be between 0 and 1.")

    safe_bandwidth = bandwidth_kbps * safety_factor

    suitable = [
        r for r in renditions
        if _get_bitrate(r) <= safe_bandwidth
    ]

    if suitable:
        return max(suitable, key=_get_bitrate)

    # If none fits, select the lowest available quality.
    return min(renditions, key=_get_bitrate)



def select_budget_aware(
    renditions: list[dict[str, Any]],
    bandwidth_kbps: float,
    remaining_budget_bytes: int,
    remaining_chunks: int,
    priority: str | float = "medium",
    safety_factor: float = 0.85,
    buffer_s: float | None = None,
) -> dict[str, Any]:
    """Select quality using bandwidth, remaining budget, buffer, and chunk priority."""

    if not renditions:
        raise ValueError("At least one rendition is required.")

    if bandwidth_kbps <= 0:
        raise ValueError("Bandwidth must be greater than zero.")

    if remaining_budget_bytes < 0:
        raise ValueError("Remaining budget cannot be negative.")

    if remaining_chunks <= 0:
        raise ValueError("Remaining chunks must be greater than zero.")

    if not 0 < safety_factor <= 1:
        raise ValueError("Safety factor must be between 0 and 1.")

    priority_weights = {
        "low": 0.04,
        "medium": 0.02,
        "high": 0.005,
    }

    if isinstance(priority, (int, float)):
        if priority >= 0.75:
            priority = "high"
        elif priority >= 0.4:
            priority = "medium"
        else:
            priority = "low"
    else:
        priority = str(priority).lower()

    if priority not in priority_weights:
        raise ValueError("Priority must be low, medium or high.")

    # Tighten bandwidth safety margin when playback buffer is dangerously low.
    low_buffer = buffer_s is not None and buffer_s < 1.5
    effective_safety = min(safety_factor, 0.65) if low_buffer else safety_factor
    safe_bandwidth = bandwidth_kbps * effective_safety
    chunk_allowance = remaining_budget_bytes / remaining_chunks

    bandwidth_options = [
        r for r in renditions
        if _get_bitrate(r) <= safe_bandwidth
    ]

    if not bandwidth_options:
        bandwidth_options = [
            min(renditions, key=_get_bitrate)
        ]

    budget_options = [
        r for r in bandwidth_options
        if _get_size(r) <= chunk_allowance
    ]

    if not budget_options:
        return min(bandwidth_options, key=_get_size)

    cost_weight = priority_weights[priority]

    def utility(rendition: dict[str, Any]) -> float:
        ssim = float(rendition.get("ssim", 0.0))
        size_ratio = _get_size(rendition) / max(chunk_allowance, 1)
        return ssim - (cost_weight * size_ratio)

    return max(budget_options, key=utility)
