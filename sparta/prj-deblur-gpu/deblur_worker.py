#!/usr/bin/env python3
"""
Deblur Service using the simplified base framework with single-GPU support.
Reads RTSP frames, runs Deblur, publishes RTSP frames.
"""
from typing import Any, Dict

from contanos.base_worker import BaseWorker
from pelpers.lvr_helper import LVRHelper
class DeblurWorker(BaseWorker):
    
    def __init__(self, worker_id: int, device: str, 
                 model_config: Dict,
                 input_interface, 
                 output_interface):

        super().__init__(worker_id, device, model_config,
                         input_interface, output_interface)
    
    def _model_init(self):
        self.model = LVRHelper(**self.model_config,
                           device=self.device)  # Use the specific device for this model
        
    def _predict(self, input: Any, metadata: Any=None) -> Any:

        model_output = self.model(input)
        
        return {'img': model_output}
