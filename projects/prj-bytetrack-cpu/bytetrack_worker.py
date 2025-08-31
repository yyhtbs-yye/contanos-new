#!/usr/bin/env python3
"""
YOLOX Detection Service using the simplified base framework with multi-GPU support.
Reads RTSP frames, runs YOLOX detection, publishes bounding boxes to MQTT.
"""

from typing import Any, Dict
import numpy as np

from contanos.base_worker import BaseWorker

from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.bytetrack.bytetrack import STrack


class ByteTrackWorker(BaseWorker):
    """ByteTrack tracking processor with single CPU serial processing."""

    def __init__(self, worker_id: int, device: str,
                 model_config: Dict,
                 input_interface,
                 output_interface):
        super().__init__(worker_id, device, model_config,
                         input_interface, output_interface)

    def _model_init(self):
        self.model = ByteTrack(**self.model_config)  # Use the specific device for this model

    def _predict(self, input: Any, metadata: Any) -> Any:

        if int(metadata.get('frame_id_str').split('FRAME:')[-1]) <= self.model_config.get('starting_frame_id', 1):
            print(f"[RESET] First Frame received - clearing buffers & restarting tracker")

            self.model = ByteTrack(**self.model_config)
            STrack.clear_count()  # reset track ID counter

        dets = []
        for i in range(len(input['results']['det_scores'])):
            dets.append(
                [*input['results']['bboxes'][i], input['results']['det_scores'][i], input['results']['classes'][i]])

        dets = np.array(dets)
        tracklets = self.model.update(dets)

        track_ids = [tracklet[4] for tracklet in tracklets]
        bboxes = [[tracklet[0], tracklet[1], tracklet[2], tracklet[3]] for tracklet in tracklets]
        track_scores = [tracklet[5] for tracklet in tracklets]
        return {'scale': 1, 'bboxes': bboxes, 'track_scores': track_scores, 'track_ids': track_ids}