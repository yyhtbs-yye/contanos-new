#!/usr/bin/env python3
"""
RTMPose Pose Estimation Service using the simplified base framework with multi-GPU support.
Reads RTSP frames + MQTT DET Bbox, runs RTMPose detection, publishes keypoints to MQTT.
"""

from typing import Any, Dict

from contanos.base_worker import BaseWorker
from rtmlib.tools.pose_estimation import RTMPose
class RTMPoseWorker(BaseWorker):
    """RTMPose detection processor with multi-GPU parallel processing."""
    
    def __init__(self, worker_id: int, device: str, 
                 model_config: Dict,
                 input_interface, 
                 output_interface):
        super().__init__(worker_id, device, model_config,
                         input_interface, output_interface)
    
    def _model_init(self):
        self.model = RTMPose(**self.model_config,
                           device=self.device)  # Use the specific device for this model
        
    def _predict(self, input: Any, metadata: Any) -> Any:

        keypoints, keypoint_scores = self.model(input[0], input[1]['results']['bboxes'])
        return {'scale': 1, 'keypoints': keypoints, 'keypoint_scores': keypoint_scores}
