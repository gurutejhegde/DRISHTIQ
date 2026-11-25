"""Shared data schemas and contracts for NazarBit."""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel, Field


class ComplexityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BoundingBox(BaseModel):
    x: float = Field(description="Normalized x coordinate (0.0 to 1.0)")
    y: float = Field(description="Normalized y coordinate (0.0 to 1.0)")
    width: float = Field(description="Normalized width (0.0 to 1.0)")
    height: float = Field(description="Normalized height (0.0 to 1.0)")
    label: str = Field(default="roi", description="ROI type, e.g. face, object, motion")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ChunkAnalysis(BaseModel):
    chunk_index: int = Field(description="Zero-indexed segment number")
    start_time_s: float = Field(description="Start time of chunk in seconds")
    end_time_s: float = Field(description="End time of chunk in seconds")
    duration_s: float = Field(description="Duration in seconds")
    scene_cut_count: int = Field(default=0, description="Number of scene transitions in chunk")
    texture_score: float = Field(
        ge=0.0, le=1.0, description="Spatial edge & texture detail score (0=flat, 1=complex)"
    )
    motion_score: float = Field(
        ge=0.0, le=1.0, description="Temporal motion score (0=static, 1=fast motion)"
    )
    combined_score: float = Field(ge=0.0, le=1.0, description="Weighted composite complexity score")
    complexity: ComplexityLevel = Field(description="Overall complexity bucket")
    priority: PriorityLevel = Field(description="Encoding priority bucket")
    roi_boxes: List[BoundingBox] = Field(
        default_factory=list, description="Regions of interest in chunk"
    )


class VideoMetadata(BaseModel):
    filename: str
    file_path: str
    duration_s: float
    fps: float
    total_frames: int
    width: int
    height: int


class VideoAnalysisReport(BaseModel):
    metadata: VideoMetadata
    chunk_duration_s: float
    total_chunks: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    chunks: List[ChunkAnalysis]

    def to_json(self, path: str | Path) -> None:
        """Export report to a formatted JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "VideoAnalysisReport":
        """Load report from a JSON file."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_csv(self, path: str | Path) -> None:
        """Export tabular chunk analysis data to CSV."""
        rows = []
        for c in self.chunks:
            rows.append({
                "chunk_index": c.chunk_index,
                "start_time_s": c.start_time_s,
                "end_time_s": c.end_time_s,
                "duration_s": c.duration_s,
                "scene_cut_count": c.scene_cut_count,
                "texture_score": round(c.texture_score, 4),
                "motion_score": round(c.motion_score, 4),
                "combined_score": round(c.combined_score, 4),
                "complexity": c.complexity.value,
                "priority": c.priority.value,
                "roi_count": len(c.roi_boxes),
            })
        df = pd.DataFrame(rows)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)
