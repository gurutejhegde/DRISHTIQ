"""Member 1: Content Analysis Engine for DrishtiQ.

Performs:
1. Video ingestion & frame sampling.
2. Scene cut detection (via PySceneDetect with graceful fallback).
3. Spatial edge & texture detail measurement (Laplacian / Sobel).
4. Temporal motion complexity measurement (Dense Optical Flow / Frame Differencing).
5. Region of Interest (ROI) extraction (Saliency & Face detection).
6. Chunk complexity and priority classification.
7. Clean Python API and JSON/CSV export.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video

from drishtiq.core.schemas import (
    BoundingBox,
    ChunkAnalysis,
    ComplexityLevel,
    PriorityLevel,
    VideoAnalysisReport,
    VideoMetadata,
)


class ContentAnalyzer:
    """Video content analyzer that labels chunks by spatial/temporal complexity and priority."""

    def __init__(
        self,
        chunk_duration_s: float = 2.0,
        sample_fps: float = 4.0,
        scene_threshold: float = 27.0,
        texture_weight: float = 0.45,
        motion_weight: float = 0.55,
    ):
        """Initialize the analyzer with customizable weights and sampling rates."""
        self.chunk_duration_s = max(0.5, float(chunk_duration_s))
        self.sample_fps = max(1.0, float(sample_fps))
        self.scene_threshold = float(scene_threshold)
        self.texture_weight = texture_weight
        self.motion_weight = motion_weight

        # Precompute gentle spatial center-bias mask at saliency working size (160x90)
        yy, xx = np.mgrid[0:90, 0:160]
        cx, cy = 80.0, 45.0
        dist2 = ((xx - cx) / 70.0) ** 2 + ((yy - cy) / 40.0) ** 2
        self._center_prior = (0.78 + 0.22 * np.exp(-0.5 * dist2)).astype(np.float32)

        # Optional facial cascade (if supported by installed OpenCV build)
        self.face_cascade = None
        has_cascade = (
            hasattr(cv2, "CascadeClassifier")
            and hasattr(cv2, "data")
            and hasattr(cv2.data, "haarcascades")
        )
        if has_cascade:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if Path(cascade_path).exists():
                self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract basic video stream metadata."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_s = total_frames / fps if fps > 0 else 0.0

            return VideoMetadata(
                filename=video_path.name,
                file_path=str(video_path.resolve()),
                duration_s=round(duration_s, 3),
                fps=round(fps, 2),
                total_frames=total_frames,
                width=width,
                height=height,
            )
        finally:
            cap.release()

    def detect_scene_cuts(self, video_path: Path) -> List[float]:
        """Detect scene cut timestamps (in seconds) using PySceneDetect."""
        try:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold))
            scene_manager.detect_scenes(video)
            scenes = scene_manager.get_scene_list()

            cut_times: List[float] = []
            for sc in scenes:
                start_obj = sc[0]
                if hasattr(start_obj, "seconds"):
                    sec = float(start_obj.seconds)
                elif hasattr(start_obj, "get_seconds"):
                    sec = float(start_obj.get_seconds())
                else:
                    sec = float(start_obj)
                if sec > 0.05:
                    cut_times.append(sec)
            return cut_times
        except Exception:
            # Fallback if scenedetect encounters format issues
            return []

    def compute_texture_score(self, gray_frame: np.ndarray) -> float:
        """Measure spatial detail and texture complexity using Laplacian and Sobel gradients."""
        if gray_frame is None or gray_frame.size == 0:
            return 0.0

        # Laplacian variance measures edge sharpness and high-frequency details
        lap_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

        # Sobel gradients in X and Y
        sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))

        # Soft normalized scaling into [0.0, 1.0]
        norm_lap = np.clip(lap_var / 350.0, 0.0, 1.0)
        norm_grad = np.clip(grad_mag / 30.0, 0.0, 1.0)

        return float(0.6 * norm_lap + 0.4 * norm_grad)

    def compute_motion_score(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Measure temporal motion using downsampled optical flow magnitude."""
        if prev_gray is None or curr_gray is None:
            return 0.0

        # Downsample for computational efficiency during analysis
        small_prev = cv2.resize(prev_gray, (320, 180), interpolation=cv2.INTER_AREA)
        small_curr = cv2.resize(curr_gray, (320, 180), interpolation=cv2.INTER_AREA)

        # Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            small_prev,
            small_curr,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=11,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = np.mean(mag)

        # Normalize motion into [0.0, 1.0]
        norm_motion = np.clip(avg_mag / 4.0, 0.0, 1.0)
        return float(norm_motion)

    def compute_saliency_map(
        self, frame: np.ndarray, prev_gray: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute normalized 2D visual saliency map (90x160) from LAB contrast, motion, & edges."""
        small_bgr = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_blur = cv2.GaussianBlur(lab, (5, 5), 0)
        lab_mean = lab.mean(axis=(0, 1), keepdims=True)
        color_sal = np.linalg.norm(lab_blur - lab_mean, axis=2)
        color_sal = color_sal / (color_sal.max() + 1e-6)

        gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_sal = cv2.magnitude(gx, gy)
        edge_sal = edge_sal / (edge_sal.max() + 1e-6)

        if prev_gray is not None:
            small_prev = cv2.resize(prev_gray, (160, 90), interpolation=cv2.INTER_AREA)
            mot = cv2.absdiff(gray, small_prev).astype(np.float32)
            mot_sal = mot / (mot.max() + 1e-6)
            fused = 0.40 * color_sal + 0.35 * mot_sal + 0.25 * edge_sal
        else:
            fused = 0.58 * color_sal + 0.42 * edge_sal

        fused = cv2.GaussianBlur(fused * self._center_prior, (11, 11), 0)
        peak = float(fused.max())
        return (fused / peak) if peak > 1e-6 else np.zeros((90, 160), dtype=np.float32)

    def detect_rois(
        self,
        frame: np.ndarray,
        prev_gray: Optional[np.ndarray] = None,
        saliency_map: Optional[np.ndarray] = None,
    ) -> List[BoundingBox]:
        """Detect key visual regions of interest (Faces and multi-cue Saliency boxes)."""
        boxes: List[BoundingBox] = []
        if frame is None or frame.size == 0:
            return boxes

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 1. Face detection (if cascade available)
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30)
            )
            for x, y, fw, fh in faces:
                boxes.append(
                    BoundingBox(
                        x=round(x / w, 4),
                        y=round(y / h, 4),
                        width=round(fw / w, 4),
                        height=round(fh / h, 4),
                        label="face",
                        confidence=0.9,
                    )
                )

        # 2. Multi-cue saliency peak localization
        if not boxes:
            smap = (
                saliency_map
                if saliency_map is not None
                else self.compute_saliency_map(frame, prev_gray)
            )
            if float(smap.max()) < 0.05:
                return boxes

            thresh_val = max(0.42, float(np.percentile(smap, 84)))
            mask = (smap >= thresh_val).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if cw < 12 or ch < 12:
                    continue
                rw, rh = min(cw / 160.0, 0.50), min(ch / 90.0, 0.62)
                patch = smap[cy : cy + ch, cx : cx + cw]
                py, px = np.unravel_index(np.argmax(patch), patch.shape)
                peak_x, peak_y = (cx + px) / 160.0, (cy + py) / 90.0
                rx = float(np.clip(peak_x - rw / 2, 0.0, 1.0 - rw))
                ry = float(np.clip(peak_y - rh / 2, 0.0, 1.0 - rh))
                conf = round(float(np.clip(patch.mean(), 0.5, 0.95)), 2)
                boxes.append(
                    BoundingBox(
                        x=round(rx, 4),
                        y=round(ry, 4),
                        width=round(rw, 4),
                        height=round(rh, 4),
                        label="motion" if prev_gray is not None else "saliency",
                        confidence=conf,
                    )
                )

        return boxes

    def classify_chunk(
        self,
        texture_score: float,
        motion_score: float,
        scene_cut_count: int,
        roi_count: int,
    ) -> Tuple[float, ComplexityLevel, PriorityLevel]:
        """Compute combined score and categorize into Complexity and Priority levels."""
        combined = (self.texture_weight * texture_score) + (self.motion_weight * motion_score)
        combined = float(np.clip(combined, 0.0, 1.0))

        # Complexity categorization
        if combined < 0.20:
            complexity = ComplexityLevel.LOW
        elif combined < 0.55:
            complexity = ComplexityLevel.MEDIUM
        else:
            complexity = ComplexityLevel.HIGH

        # Priority calculation: scene cuts or high complexity or high ROI count boost priority
        if scene_cut_count > 0 or roi_count >= 2 or combined > 0.50:
            priority = PriorityLevel.HIGH
        elif combined >= 0.18 or roi_count == 1:
            priority = PriorityLevel.MEDIUM
        else:
            priority = PriorityLevel.LOW

        return combined, complexity, priority

    def analyze(self, video_path: str | Path) -> VideoAnalysisReport:
        """Run full content analysis pipeline on the given video."""
        video_p = Path(video_path)
        if not video_p.exists():
            raise FileNotFoundError(f"Video file not found at {video_p}")

        metadata = self.extract_metadata(video_p)
        scene_cuts = self.detect_scene_cuts(video_p)

        cap = cv2.VideoCapture(str(video_p))
        fps = metadata.fps if metadata.fps > 0 else 30.0
        frame_step = max(1, int(round(fps / self.sample_fps)))
        frames_per_chunk = int(round(fps * self.chunk_duration_s))

        chunk_analyses: List[ChunkAnalysis] = []
        chunk_idx = 0
        current_frame_idx = 0
        prev_gray_frame: Optional[np.ndarray] = None

        while current_frame_idx < metadata.total_frames:
            start_s = round(chunk_idx * self.chunk_duration_s, 3)
            end_s = round(min(metadata.duration_s, (chunk_idx + 1) * self.chunk_duration_s), 3)
            chunk_end_frame = min(metadata.total_frames, current_frame_idx + frames_per_chunk)

            chunk_textures: List[float] = []
            chunk_motions: List[float] = []
            chunk_rois: List[BoundingBox] = []
            chunk_smaps: List[np.ndarray] = []

            # Count scene cuts falling inside this chunk
            cuts_in_chunk = sum(1 for cut in scene_cuts if start_s <= cut < end_s)

            # Sample frames within chunk
            for f_idx in range(current_frame_idx, chunk_end_frame, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                texture_val = self.compute_texture_score(gray)
                chunk_textures.append(texture_val)

                if prev_gray_frame is not None:
                    motion_val = self.compute_motion_score(prev_gray_frame, gray)
                    chunk_motions.append(motion_val)

                smap = self.compute_saliency_map(frame, prev_gray=prev_gray_frame)
                chunk_smaps.append(cv2.resize(smap, (24, 14), interpolation=cv2.INTER_AREA))

                # Sample ROI using multi-cue saliency before updating prev_gray_frame
                if len(chunk_rois) < 3 and (len(chunk_textures) % 2 == 1):
                    detected = self.detect_rois(
                        frame, prev_gray=prev_gray_frame, saliency_map=smap
                    )
                    chunk_rois.extend(detected)

                prev_gray_frame = gray

            # Aggregate scores and average saliency heatmap for chunk
            avg_texture = float(np.mean(chunk_textures)) if chunk_textures else 0.0
            avg_motion = float(np.mean(chunk_motions)) if chunk_motions else 0.0
            if chunk_smaps:
                mean_smap = np.mean(chunk_smaps, axis=0)
                heatmap_grid = [[round(float(v), 2) for v in row] for row in mean_smap]
            else:
                heatmap_grid = []

            combined, complexity, priority = self.classify_chunk(
                texture_score=avg_texture,
                motion_score=avg_motion,
                scene_cut_count=cuts_in_chunk,
                roi_count=len(chunk_rois),
            )

            chunk_analyses.append(
                ChunkAnalysis(
                    chunk_index=chunk_idx,
                    start_time_s=start_s,
                    end_time_s=end_s,
                    duration_s=round(end_s - start_s, 3),
                    scene_cut_count=cuts_in_chunk,
                    texture_score=round(avg_texture, 4),
                    motion_score=round(avg_motion, 4),
                    combined_score=round(combined, 4),
                    complexity=complexity,
                    priority=priority,
                    roi_boxes=chunk_rois[:4],
                    saliency_heatmap=heatmap_grid,
                )
            )

            chunk_idx += 1
            current_frame_idx = chunk_end_frame

        cap.release()

        return VideoAnalysisReport(
            metadata=metadata,
            chunk_duration_s=self.chunk_duration_s,
            total_chunks=len(chunk_analyses),
            chunks=chunk_analyses,
        )

