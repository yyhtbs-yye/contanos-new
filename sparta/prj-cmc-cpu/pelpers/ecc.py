# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import cv2
import numpy as np
import logging

from pelpers.base_cmc import BaseCMC

class ECC(BaseCMC):
    def __init__(
        self,
        warp_mode: str = 'MOTION_TRANSLATION',
        eps: float = 1e-5,
        max_iter: int = 100,
        scale: float = 0.15,
        align: bool = False,
        grayscale: bool = True,
    ) -> None:
        self.align = align
        self.grayscale = grayscale
        self.scale = scale
        self.warp_mode = getattr(cv2, warp_mode)
        self.termination_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iter,
            eps,
        )
        self.prev_img = None

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply sparse optical flow to compute the warp matrix.

        Parameters:
            img (ndarray): The input image.

        Returns:
            ndarray: The warp matrix from the source to the destination.
                If the motion model is homography, the warp matrix will be 3x3; otherwise, it will be 2x3.
        """

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        if self.prev_img is None:
            self.prev_img = self.preprocess(img)
            return warp_matrix

        img = self.preprocess(img)

        try:
            (ret_val, warp_matrix) = cv2.findTransformECC(
                self.prev_img,  # already processed
                img,
                warp_matrix,
                self.warp_mode,
                self.termination_criteria,
                None,
                1,
            )
        except cv2.error as e:
            # error 7 is StsNoConv, according to https://docs.opencv.org/3.4/d1/d0d/namespacecv_1_1Error.html
            if e.code == cv2.Error.StsNoConv:
                logging.warning(
                    f"Affine matrix could not be generated: {e}. Returning identity"
                )
                return warp_matrix
            else:  # other error codes
                raise

        # upscale warp matrix to original images size
        if self.scale < 1:
            warp_matrix[0, 2] /= self.scale
            warp_matrix[1, 2] /= self.scale

        self.prev_img = img

        return warp_matrix 
