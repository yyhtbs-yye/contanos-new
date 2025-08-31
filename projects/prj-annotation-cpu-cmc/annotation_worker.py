from typing import Any, Dict

# from annotators.box_drawer import BoxDrawer
# from annotators.skeleton_drawer import SkeletonDrawer
# from annotators.trajectory_drawer import TrajectoryDrawer

from contanos.base_worker import BaseWorker
from annotation_management import AnnotationProcessor

class AnnotationWorker(BaseWorker):
    """ByteTrack tracking processor with single CPU serial processing."""
    
    def __init__(self, worker_id: int, device: str, 
                 model_config: Dict,
                 input_interface, 
                 output_interface):
        super().__init__(worker_id, device, model_config,
                         input_interface, output_interface)
    
    def _model_init(self):
        self.model = AnnotationProcessor()
        
    def _predict(self, input: Any, metadata: Any) -> Any:

        frame = input[0]

        frame_id = input[1]['frame_id']
        bboxes = input[1]['results']['bboxes']
        track_ids = [int(it) for it in input[1]['results']['track_ids']]
        track_scores = input[1]['results']['track_scores']
        scale = input[1]['results']['scale']
        keypoints = input[2]['results']['keypoints']

        if len(input) > 3:
            proj_matrix = input[3]['results']['proj_matrix'] if 'proj_matrix' in input[3]['results'] else None
        else:
            proj_matrix = None


        annotated_frame = self.model(frame=frame, frame_id=frame_id, bboxes=bboxes, track_ids=track_ids, track_scores=track_scores, scale=scale, keypoints=keypoints, 
                                     proj_matrix=proj_matrix)

        return {'img': annotated_frame}

