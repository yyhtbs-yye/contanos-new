#!/usr/bin/env python3

from typing import Any, Dict
from contanos.base_worker import BaseWorker
from pelpers.jomn_helper import JOMNHelper

class JerseyOCRWorker(BaseWorker):
    
    def __init__(self, worker_id: int, device: str, 
                 model_config: Dict,
                 input_interface, 
                 output_interface):
        super().__init__(worker_id, device, model_config,
                         input_interface, output_interface)
    
    def _model_init(self):
        self.model = JOMNHelper(**self.model_config,
                           device=self.device)  
        
    def _predict(self, input: Any, metadata: Any) -> Any:

        numbers, potential_numbers, confidences = self.model(input[0], input[1]['results']['bboxes'])

        return {
            'track_ids': [int(it) for it in input[1]['results']['track_ids']],
            'numbers': numbers,
            'potential_numbers': potential_numbers,
            'confidences': confidences
        }
