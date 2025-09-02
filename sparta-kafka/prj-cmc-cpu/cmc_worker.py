#!/usr/bin/env python3
from typing import Any, Dict

from contanos.base_worker import BaseWorker
from pelpers.ecc import ECC

class CMCWorker(BaseWorker):
    
    def __init__(self, worker_id: int, device: str, 
                 model_config: Dict,
                 input_interface, 
                 output_interface):
        super().__init__(worker_id, device, model_config,
                         input_interface, output_interface)
    
    def _model_init(self):
        self.model = ECC(**self.model_config)  # Use the specific device for this model
        
    def _predict(self, inputs: Any, metadata: Any=None) -> Any:
        proj_matrix = self.model.apply(inputs)

        return {'proj_matrix': proj_matrix}
