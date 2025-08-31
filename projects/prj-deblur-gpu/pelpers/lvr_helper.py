import torch
import numpy as np
import os
from pelpers.lvr_arch import LiteVideoRestorer

from time import time, sleep

class LVRHelper:
    def __init__(self,
                 weights_path: str,
                 device: str):
        
        self.device = device
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        self.model = self._load_model(weights_path)
    
    def _load_model(self, ckpt_path: str):

        # Assumes model class is imported/available in scope
        model = LiteVideoRestorer()  # Replace with your actual model class
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = None
        if isinstance(ckpt, dict):
            for k in ["params_ema", "params", "state_dict", "model", "net", "net_g", "model_state_dict"]:
                if k in ckpt:
                    state = ckpt[k]
                    break
            if state is None:
                state = ckpt  # maybe directly a state_dict
        if not isinstance(state, dict):
            raise RuntimeError(f"Unrecognized checkpoint structure at {ckpt_path}")
        
        # Remove 'module.' prefix if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        
        return model
    
    @torch.no_grad()
    def __call__(self, image: np.ndarray):

        now = time()

        # Convert to tensor format
        image_tensor = torch.from_numpy(image.astype(np.float32)).to(self.device, non_blocking=True)
        
        assert image_tensor.ndim == 3, "Input image must be 3D numpy (HWC) array"
        assert image_tensor.shape[2] == 3, "Input image must have 3 channels (RGB)"

        image_tensor = image_tensor.permute(2, 0, 1)
        
        image_tensor = image_tensor / 255.0  # C,H,W
        
        frame = image_tensor.unsqueeze(0)  # 1,C,H,W
        restored = self.model.stream_process(frame).squeeze().detach()

        output = restored.clamp_(0.0, 1.0).mul(255.0).round().to(torch.uint8).permute(1, 2, 0) # C,H,W -> H,W,C
        
        output = output.cpu().numpy()  # T,H,W,C

        # # delay 50ms for receiver fps debug
        # sleep(0.05)
        
        print(f"LVRHelper processing time: {(time() - now)*1000:.3f} miliseconds")
        return output