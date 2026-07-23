"""
phase2_alignment/aligner.py
============================
Part alignment module — aligns any inspection image to a fixed reference frame
so that all downstream ROI coordinates are stable.

Strategy
--------
Primary   : ORB keypoint detection + BFMatcher (Lowe ratio test) + RANSAC
            homography → warpPerspective to a square output canvas.
Fallback  : Contour-based alignment — find the largest connected component
            (the metal part) and transform its bounding box to the reference.

The output image is always ``cfg['alignment']['output_size']`` pixels square,
with the part centred.  This gives fixed, stable ROI coordinates for every
subsequent phase.

Usage
-----
    from phase2_alignment.aligner import Aligner
    aligner = Aligner(config)
    aligned, quality = aligner.align(bgr_image)

``quality`` is a dict with keys:
    method      : 'orb' | 'contour' | 'failed'
    n_matches   : int (ORB only)
    reproj_err  : float (pixels, ORB only)
    success     : bool
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import yaml


# ---------------------------------------------------------------------------
# Config helper (standalone usage)
# ---------------------------------------------------------------------------

def _load_cfg_from_disk() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Aligner class
# ---------------------------------------------------------------------------

class Aligner:
    """Aligns an inspection image to the reference template."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = _load_cfg_from_disk()
        self.cfg = config
        al = config["alignment"]

        self.n_features   = int(al.get("orb_features", 5000))
        self.ratio        = float(al.get("match_ratio", 0.75))
        self.min_matches  = int(al.get("min_good_matches", 10))
        out_w, out_h      = al.get("output_size", [1080, 1080])
        self.out_w        = int(out_w)
        self.out_h        = int(out_h)

        # Build the SIFT detector + BF matcher once (SIFT is much more robust for rotation/defects)
        self.sift    = cv2.SIFT_create(nfeatures=self.n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Load reference image
        ref_path = config["paths"]["reference_image"]
        ref_bgr  = cv2.imread(ref_path)
        if ref_bgr is None:
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        self.ref_bgr  = self._downscale(ref_bgr)
        self.ref_kp, self.ref_desc = self._detect(self.ref_bgr)

        # Pre-compute reference corners for reprojection error
        h_ref, w_ref = ref_bgr.shape[:2]
        self._ref_corners = np.array([
            [0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]
        ], dtype=np.float32).reshape(-1, 1, 2)

        print(f"[Aligner] reference image  : {ref_path}")
        print(f"[Aligner] output size      : {self.out_w}×{self.out_h}")
        print(f"[Aligner] SIFT features    : {self.n_features}")
        print(f"[Aligner] ref keypoints    : {len(self.ref_kp)}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def _downscale(self, img: np.ndarray, max_dim: int = 2400) -> np.ndarray:
        """Downscale images to max_dim to ensure fast feature matching."""
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def align(self, bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Align *bgr* to the reference frame.

        Returns
        -------
        aligned : np.ndarray
            The warped image, shape (out_h, out_w, 3).
        quality : dict
            Alignment quality metrics.
        """
        # Downscale input to match reference scale and speed up SIFT
        bgr = self._downscale(bgr)

        # Check if user explicitly wants only contour alignment (much faster)
        if self.cfg.get("alignment", {}).get("method") == "contour":
            return self._align_contour(bgr)

        # Try SIFT + homography first
        aligned, quality = self._align_sift(bgr)
        if quality["success"]:
            return aligned, quality

        # Fallback to contour alignment
        # print("[Aligner] SIFT failed — falling back to contour alignment")
        aligned, quality = self._align_contour(bgr)
        return aligned, quality

    def align_file(self, image_path: str) -> tuple[np.ndarray, dict]:
        """Convenience wrapper: load file then align."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)
        return self.align(bgr)

    # ------------------------------------------------------------------ #
    # SIFT alignment
    # ------------------------------------------------------------------ #

    def _detect(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # CLAHE to normalise local contrast before detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        kp, desc = self.sift.detectAndCompute(gray, None)
        return kp, desc

    def _align_sift(self, bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        quality = {"method": "sift", "n_matches": 0, "reproj_err": 9999.0, "success": False}

        kp, desc = self._detect(bgr)
        if desc is None or len(desc) < self.min_matches:
            return bgr, quality

        # Lowe's ratio test
        raw_matches = self.matcher.knnMatch(self.ref_desc, desc, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio * n.distance:
                    good.append(m)

        quality["n_matches"] = len(good)
        if len(good) < self.min_matches:
            return bgr, quality

        src_pts = np.array([self.ref_kp[m.queryIdx].pt for m in good],
                           dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([kp[m.trainIdx].pt for m in good],
                           dtype=np.float32).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return bgr, quality

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_matches:
            return bgr, quality

        # Reprojection error on inlier points
        reproj_err = self._reprojection_error(src_pts, dst_pts, H, mask)
        quality["reproj_err"] = reproj_err

        # Warp + resize to output canvas
        h_ref, w_ref = self.ref_bgr.shape[:2]
        warped  = cv2.warpPerspective(bgr, H, (w_ref, h_ref),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        aligned = cv2.resize(warped, (self.out_w, self.out_h),
                             interpolation=cv2.INTER_AREA)

        quality["success"]  = True
        quality["inliers"]  = inliers
        return aligned, quality

    @staticmethod
    def _reprojection_error(src, dst, H, mask) -> float:
        """Mean reprojection error in pixels for inlier matches."""
        src_in = src[mask.ravel() == 1]
        dst_in = dst[mask.ravel() == 1]
        if len(src_in) == 0:
            return 9999.0
        projected = cv2.perspectiveTransform(dst_in, H)
        err = np.sqrt(((src_in - projected) ** 2).sum(axis=2))
        return float(err.mean())

    # ------------------------------------------------------------------ #
    # Contour-based fallback alignment
    # ------------------------------------------------------------------ #

    def _align_contour(self, bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Find the largest connected component (the metal part) in both the
        reference and the input image. Use their minAreaRect to
        compute a similarity transform (translation + scale + rotation).
        """
        quality = {"method": "contour", "success": False}

        def part_contour(img_bgr):
            gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(smoothed, 30, 100)
            
            h, w = edges.shape
            border = 50
            edges[0:border, :] = 0
            edges[h-border:h, :] = 0
            edges[:, 0:border] = 0
            edges[:, w-border:w] = 0
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated = cv2.dilate(edges, kernel, iterations=4)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=4)
            
            closed[0:border, :] = 0
            closed[h-border:h, :] = 0
            closed[:, 0:border] = 0
            closed[:, w-border:w] = 0
            
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            valid_cnts = [c for c in cnts if cv2.contourArea(c) > 50000]
            if not valid_cnts:
                return None
            return max(valid_cnts, key=cv2.contourArea)

        ref_cnt = part_contour(self.ref_bgr)
        src_cnt = part_contour(bgr)
        
        if ref_cnt is None or src_cnt is None:
            return bgr, quality

        def get_rect_props(cnt):
            (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
            # Normalize angle so width is always > height
            if w < h:
                angle += 90
                w, h = h, w
            return cx, cy, w, h, angle

        rx, ry, rw, rh, rangle = get_rect_props(ref_cnt)
        sx, sy, sw, sh, sangle = get_rect_props(src_cnt)

        scale_x = rw / max(sw, 1)
        scale_y = rh / max(sh, 1)
        scale   = 0.5 * (scale_x + scale_y)
        
        delta_angle = rangle - sangle

        # 1. Compute rotation and scale matrix around source center
        M_rot_scale = cv2.getRotationMatrix2D((sx, sy), delta_angle, scale)
        
        # 2. Add translation to move source center to reference center
        M_rot_scale[0, 2] += (rx - sx)
        M_rot_scale[1, 2] += (ry - sy)

        h_ref, w_ref = self.ref_bgr.shape[:2]
        warped  = cv2.warpAffine(bgr, M_rot_scale, (w_ref, h_ref),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        aligned = cv2.resize(warped, (self.out_w, self.out_h),
                             interpolation=cv2.INTER_AREA)
        quality["success"] = True
        return aligned, quality
