from __future__ import annotations
import time
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import random

import cv2
import numpy as np

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = None
    score: Optional[float] = None
    
    def scale(self, factor: float) -> 'BoundingBox':
        """Return scaled version of the box"""
        return BoundingBox(
            self.x1 / factor, self.y1 / factor,
            self.x2 / factor, self.y2 / factor,
            self.track_id, self.score
        )
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside the box"""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Keypoint:
    """Represents a single keypoint"""
    x: float
    y: float
    
    def scale(self, factor: float) -> 'Keypoint':
        """Return scaled version of the keypoint"""
        return Keypoint(self.x / factor, self.y / factor)


@dataclass
class Skeleton:
    """Represents a person's skeleton with keypoints"""
    keypoints: List[Keypoint]
    track_id: Optional[int] = None
    
    def scale(self, factor: float) -> 'Skeleton':
        """Return scaled version of the skeleton"""
        return Skeleton(
            [kp.scale(factor) for kp in self.keypoints],
            self.track_id
        )
    
    def get_ankle_points(self) -> Tuple[Optional[Keypoint], Optional[Keypoint]]:
        """Get left and right ankle keypoints (indices 15, 16 in COCO)"""
        if len(self.keypoints) < 17:
            return None, None
        return self.keypoints[15], self.keypoints[16]

@dataclass
class AnnotationData:
    """Complete annotation data for a frame"""
    frame_id: int
    keypoints: List[List[float]] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    track_scores: List[float] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    scale: float = 1.0
    arrival_ts: float = field(default_factory=time.time)


# ============================================================================
# Color Management
# ============================================================================

class ColorManager:
    """Manages consistent color assignment for tracking IDs"""
    
    def __init__(self):
        self._color_cache: Dict[Optional[int], Tuple[int, int, int]] = {}
        self._default_color = (255, 0, 255)  # Magenta for None/unknown
    
    def get_color(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        """Get consistent BGR color for a tracking ID"""
        if track_id not in self._color_cache:
            if track_id is None:
                self._color_cache[track_id] = self._default_color
            else:
                # Deterministic color based on ID
                rng = random.Random(int(track_id))
                self._color_cache[track_id] = (
                    rng.randint(0, 255),
                    rng.randint(0, 255),
                    rng.randint(0, 255)
                )
        return self._color_cache[track_id]
    
    def clear_cache(self):
        """Clear the color cache (useful for memory management)"""
        self._color_cache.clear()


# ============================================================================
# Trajectory Manager
# ============================================================================

class TrajectoryManager:
    """Manages foot trajectories for tracked objects"""
    
    def __init__(self, max_length: int = 50, stale_threshold: int = 5, gap_threshold: float = 75.0):
        self.max_length = max_length
        self.stale_threshold = stale_threshold
        self.gap_threshold = gap_threshold
        
        # Store trajectories: track_id -> {side -> deque of points, last_frame}
        self._trajectories: Dict[int, Dict[str, Any]] = defaultdict(self._create_trajectory_dict)
    
    def _create_trajectory_dict(self) -> Dict[str, Any]:
        """Create a new trajectory dictionary for a track ID"""
        return {
            "left": deque(maxlen=self.max_length),
            "right": deque(maxlen=self.max_length),
            "last_frame": -1
        }
    
    def update(self, skeleton: Skeleton, frame_id: int):
        """Update trajectory for a skeleton"""
        if skeleton.track_id is None:
            return
        
        left_ankle, right_ankle = skeleton.get_ankle_points()
        if left_ankle is None or right_ankle is None:
            return
        
        traj = self._trajectories[skeleton.track_id]
        
        for side, ankle in [("left", left_ankle), ("right", right_ankle)]:
            points = traj[side]
            new_point = (ankle.x, ankle.y)
            
            # Check for gap - reset if movement is too large
            if points:
                last_point = points[-1]
                distance = np.sqrt((new_point[0] - last_point[0])**2 + 
                                 (new_point[1] - last_point[1])**2)
                if distance > self.gap_threshold:
                    points.clear()
            
            points.append(new_point)
        
        traj["last_frame"] = frame_id
    
    def purge_stale(self, current_frame: int):
        """Remove trajectories that haven't been updated recently"""
        to_remove = [
            tid for tid, traj in self._trajectories.items()
            if current_frame - traj["last_frame"] > self.stale_threshold
        ]
        for tid in to_remove:
            del self._trajectories[tid]
    
    def get_trajectories(self) -> Dict[int, Dict[str, deque]]:
        """Get all current trajectories"""
        return {
            tid: {"left": traj["left"], "right": traj["right"]}
            for tid, traj in self._trajectories.items()
        }
    
    def clear(self):
        """Clear all trajectories"""
        self._trajectories.clear()


# ============================================================================
# Drawing Components
# ============================================================================

class BoundingBoxDrawer:
    """Handles drawing of bounding boxes"""
    
    def __init__(self, color_manager: ColorManager):
        self.color_manager = color_manager
    
    def draw(self, frame: np.ndarray, bboxes: List[BoundingBox]) -> Dict[int, Tuple[int, int, int]]:
        """Draw bounding boxes on frame and return color mapping"""
        color_map = {}
        
        for bbox in bboxes:
            color = self.color_manager.get_color(bbox.track_id)
            if bbox.track_id is not None:
                color_map[bbox.track_id] = color
            
            # Draw rectangle
            cv2.rectangle(
                frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color,
                1
            )
            
            # Draw label
            label_parts = []
            if bbox.track_id is not None:
                label_parts.append(f"ID:{bbox.track_id}")
            if bbox.score is not None:
                label_parts.append(f"{bbox.score:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                cv2.putText(
                    frame,
                    label,
                    (int(bbox.x1), max(int(bbox.y1) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )
        
        return color_map


class SkeletonDrawer:
    """Handles drawing of skeletons/keypoints"""
    
    # COCO skeleton connections (zero-based indices)
    COCO_SKELETON = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6)
    ]
    
    def __init__(self, color_manager: ColorManager):
        self.color_manager = color_manager
    
    def draw(self, frame: np.ndarray, skeletons: List[Skeleton], 
             color_override: Optional[Dict[int, Tuple[int, int, int]]] = None,
             draw_bones: bool = True):
        """Draw skeletons on frame"""
        for skeleton in skeletons:
            if color_override and skeleton.track_id in color_override:
                color = color_override[skeleton.track_id]
            else:
                color = self.color_manager.get_color(skeleton.track_id)
            
            # Draw joints
            for kp in skeleton.keypoints:
                cv2.circle(frame, (int(kp.x), int(kp.y)), 2, color, -1)
            
            # Draw bones
            if draw_bones and len(skeleton.keypoints) >= 17:
                for i1, i2 in self.COCO_SKELETON:
                    if i1 < len(skeleton.keypoints) and i2 < len(skeleton.keypoints):
                        p1 = skeleton.keypoints[i1]
                        p2 = skeleton.keypoints[i2]
                        cv2.line(
                            frame,
                            (int(p1.x), int(p1.y)),
                            (int(p2.x), int(p2.y)),
                            color,
                            1
                        )


class TrajectoryDrawer:
    """Handles drawing of trajectories"""
    
    def __init__(self, color_manager: ColorManager):
        self.color_manager = color_manager
    
    def draw(self, frame: np.ndarray, trajectories: Dict[int, Dict[str, deque]]):
        """Draw trajectories on frame"""
        for track_id, feet in trajectories.items():
            color = self.color_manager.get_color(track_id)
            
            for side in ["left", "right"]:
                points = feet[side]
                if len(points) > 1:
                    pts_array = np.array(list(points), dtype=np.int32)
                    cv2.polylines(
                        frame,
                        [pts_array],
                        isClosed=False,
                        color=color,
                        thickness=1
                    )


# ============================================================================
# Track Association
# ============================================================================

class TrackAssociator:
    """Associates skeletons with tracking IDs based on bounding box overlap"""
    
    @staticmethod
    def associate(skeletons: List[Skeleton], bboxes: List[BoundingBox], 
                  threshold: float = 0.5) -> List[Skeleton]:
        """
        Associate skeletons with tracking IDs from bounding boxes.
        Returns updated skeletons with track_id set.
        """
        associated_skeletons = []
        
        for skeleton in skeletons:
            best_track_id = None
            best_overlap = 0
            
            for bbox in bboxes:
                if bbox.track_id is None:
                    continue
                
                # Count keypoints inside this box
                hits = sum(
                    1 for kp in skeleton.keypoints
                    if bbox.contains_point(kp.x, kp.y)
                )
                
                overlap_ratio = hits / len(skeleton.keypoints) if skeleton.keypoints else 0
                
                if overlap_ratio >= threshold and hits > best_overlap:
                    best_overlap = hits
                    best_track_id = bbox.track_id
            
            skeleton.track_id = best_track_id
            associated_skeletons.append(skeleton)
        
        return associated_skeletons


# ============================================================================
# Annotation Processor
# ============================================================================

class AnnotationProcessor:
    """Processes and draws all annotations on a frame"""
    
    def __init__(self):
        self.color_manager = ColorManager()
        self.trajectory_manager = TrajectoryManager()
        self.bbox_drawer = BoundingBoxDrawer(self.color_manager)
        self.skeleton_drawer = SkeletonDrawer(self.color_manager)
        self.trajectory_drawer = TrajectoryDrawer(self.color_manager)
        self.track_associator = TrackAssociator()
    
    def process_frame(self, frame: np.ndarray, annotation: AnnotationData) -> np.ndarray:
        annotated_frame = frame.copy()

        scaled_bboxes = []
        for i, bbox in enumerate(annotation.bboxes):
            track_id = annotation.track_ids[i] if i < len(annotation.track_ids) else None
            score = annotation.track_scores[i] if i < len(annotation.track_scores) else None
            scaled_bboxes.append(
                BoundingBox(
                    x1=bbox[0] / annotation.scale,
                    y1=bbox[1] / annotation.scale,
                    x2=bbox[2] / annotation.scale,
                    y2=bbox[3] / annotation.scale,
                    track_id=track_id,
                    score=score
                )
            )
        
        # Convert keypoints to Skeleton objects with scaling
        # Based on original code: keypoints is List[List[Tuple[float, float]]]
        # Each person is a list of (x, y) tuples
        scaled_skeletons = []
        for person_kps in annotation.keypoints:
            keypoints = []
            for x, y in person_kps:  # person_kps is a list of (x, y) tuples
                keypoints.append(
                    Keypoint(
                        x=x / annotation.scale,
                        y=y / annotation.scale
                    )
                )
            scaled_skeletons.append(Skeleton(keypoints=keypoints))
        
        # Associate skeletons with track IDs
        if scaled_bboxes and scaled_skeletons:
            scaled_skeletons = self.track_associator.associate(
                scaled_skeletons, scaled_bboxes
            )
        
        # Update trajectories
        for skeleton in scaled_skeletons:
            self.trajectory_manager.update(skeleton, annotation.frame_id)
        
        # Purge stale trajectories
        self.trajectory_manager.purge_stale(annotation.frame_id)
        
        # Draw in order: trajectories -> boxes -> skeletons
        # (so newest elements appear on top)
        
        # 1. Draw trajectories (bottom layer)
        trajectories = self.trajectory_manager.get_trajectories()
        self.trajectory_drawer.draw(annotated_frame, trajectories)
        
        # 2. Draw bounding boxes
        color_map = self.bbox_drawer.draw(annotated_frame, scaled_bboxes)
        
        # 3. Draw skeletons (top layer)
        self.skeleton_drawer.draw(
            annotated_frame,
            scaled_skeletons,
            color_override=color_map
        )
        
        return annotated_frame
    
    def reset(self):
        """Reset all stateful components"""
        self.color_manager.clear_cache()
        self.trajectory_manager.clear()

    def __call__(self, **input):

        frame = input.pop('frame')
        annotation = AnnotationData(**input)
        return self.process_frame(frame, annotation)