#!/usr/bin/env python3
"""
Homography overlay demo:
- Uses pelpers.ecc.ECC (MOTION_HOMOGRAPHY) to estimate transform between frames.
- Warps the previous frame into the current frame's coordinates.
- Overlays warped previous on top of current at 50% opacity.
- Writes each result to PNG.

Usage:
  python homography_overlay.py --video input.mp4 --out out_frames --shade 0.5 --scale 0.15
"""

import os
import argparse
import cv2
import numpy as np
from abc import ABC, abstractmethod

class BaseCMC(ABC):

    @abstractmethod
    def apply(self, im):
        pass

    def generate_mask(self, img, dets, scale):
        h, w = img.shape
        mask = np.zeros_like(img)

        mask[int(0.02 * h) : int(0.98 * h), int(0.02 * w) : int(0.98 * w)] = 255
        if dets is not None:
            for det in dets:
                tlbr = np.multiply(det, scale).astype(int)
                mask[tlbr[1] : tlbr[3], tlbr[0] : tlbr[2]] = 0

        return mask

    def preprocess(self, img):

        # bgr2gray
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize
        if self.scale is not None:
            img = cv2.resize(
                img,
                (0, 0),
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_LINEAR,
            )

        return img

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
                print(
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



def ensure_3x3(H: np.ndarray) -> np.ndarray:
    """Ensure the warp is 3x3 (homography) so we can invert/warp with perspective."""
    H = np.asarray(H, dtype=np.float32)
    if H.shape == (3, 3):
        return H
    if H.shape == (2, 3):  # upgrade affine to homography
        H3 = np.vstack([H, np.array([0, 0, 1], dtype=np.float32)])
        return H3
    raise ValueError(f"Unexpected warp matrix shape: {H.shape}")


def safe_invert(H: np.ndarray) -> np.ndarray:
    """Invert a 3x3 homography; fall back to identity if singular."""
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float32)


def process_video(
    video_path: str,
    out_dir: str,
    shade: float = 0.5,
    scale: float = 0.15,
    grayscale: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read first frame
    ok, prev_color = cap.read()
    if not ok or prev_color is None:
        raise RuntimeError("Failed to read first frame.")

    h, w = prev_color.shape[:2]

    # Initialize ECC with homography mode
    ecc = ECC(
        warp_mode="MOTION_HOMOGRAPHY",
        scale=scale,
        grayscale=grayscale,
        align=False,
        max_iter=100,
        eps=1e-5,
    )

    # Prime ECC: first call stores internal prev image and returns identity
    _ = ecc.apply(prev_color)

    # Save first frame as-is (no previous to warp)
    idx = 0
    out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
    cv2.imwrite(out_path, prev_color)
    idx += 1

    # Process the rest
    while True:
        ok, curr_color = cap.read()
        if not ok or curr_color is None:
            break

        # Get warp that aligns CURRENT → PREVIOUS (that's how findTransformECC defines it
        # when called as findTransformECC(prev, curr, ...))
        H_c2p = ecc.apply(curr_color)

        # Convert to 3x3 and invert to obtain PREVIOUS → CURRENT
        H_c2p = ensure_3x3(H_c2p)
        H_p2c = safe_invert(H_c2p)

        # Warp previous color frame into current frame coordinates
        warped_prev = cv2.warpPerspective(
            prev_color,
            H_p2c,
            (curr_color.shape[1], curr_color.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # Overlay: 50% shade for warped previous on top of current
        # (current kept at full strength, warped prev at `shade`)
        overlay = cv2.addWeighted(curr_color, 1.0, warped_prev, float(shade), 0.0)

        out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        cv2.imwrite(out_path, overlay)

        # Advance
        prev_color = curr_color
        idx += 1

    cap.release()


def parse_args():
    p = argparse.ArgumentParser(description="Warp previous frame onto current using ECC homography and save overlays.")
    p.add_argument("--video", default='projects/media/rte_far_seg_1_x2.mp4', help="Path to input MP4 (or any OpenCV-readable video).")
    p.add_argument("--out", default='output_align.mp4', help="Output folder for PNG frames.")
    p.add_argument("--shade", type=float, default=0.5, help="Opacity for warped previous overlay (0..1). Default 0.5.")
    p.add_argument(
        "--scale",
        type=float,
        default=0.15,
        help="Downscale factor used by ECC for speed (same as ECC.scale). Default 0.15.",
    )
    p.add_argument(
        "--grayscale",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Use grayscale in ECC preprocessing (True/False). Default True.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args.video, args.out, args.shade, args.scale, args.grayscale)
