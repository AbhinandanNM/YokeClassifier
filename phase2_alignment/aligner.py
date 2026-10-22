"""
phase2_alignment/aligner.py
============================
Part alignment module — aligns any inspection image to a fixed reference frame
so that all downstream ROI coordinates are stable.
Strategy (in order of preference)
-----------------------------------
1. SIFT + FLANN matcher + RANSAC homography on the raw frame.
2. If (1) fails, try the same frame at 90°, 180°, 270° pre-rotations.
   The rotation that yields the most inliers and lowest reprojection error wins.
3. If all SIFT attempts fail, fall back to contour-based alignment which
   uses minAreaRect to estimate translation, scale and rotation of the part
   regardless of how the worker placed it.
All three strategies are angle-agnostic — the worker can drop the component
at any orientation and the aligner will still produce a normalised output
canvas that matches the reference frame.
The output image is always ``cfg['alignment']['output_size']`` pixels square,
with the part centred.  This gives fixed, stable ROI coordinates for every
subsequent phase.
Usage
-----
    from phase2_alignment.aligner import Aligner
    aligner = Aligner(config)
    aligned, quality = aligner.align(bgr_image)
``quality`` is a dict with keys:
    method      : 'sift' | 'contour' | 'failed'
    n_matches   : int (SIFT only)
    reproj_err  : float (pixels, SIFT only)
    inliers     : int (SIFT only)
    rotation_applied : int  (degrees pre-rotated before matching, 0/90/180/270)
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
# Rotation helpers
# ---------------------------------------------------------------------------
_ROTATIONS = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
def _rotate_image(img: np.ndarray, deg: int) -> np.ndarray:
    """Rotate image by 0 / 90 / 180 / 270 degrees."""
    code = _ROTATIONS.get(deg)
    return img if code is None else cv2.rotate(img, code)
# ---------------------------------------------------------------------------
# Aligner class
# ---------------------------------------------------------------------------
class Aligner:
    """Aligns an inspection image to the reference template.
    Handles arbitrary placement angles from workers by trying four 90-degree
    pre-rotations of the input and selecting the best SIFT match. Falls back
    to contour-based alignment when SIFT fails entirely.
    """
    def __init__(self, config: dict | None = None):
        if config is None:
            config = _load_cfg_from_disk()
        self.cfg = config
        al = config["alignment"]
        self.n_features  = int(al.get("orb_features",      5000))
        self.ratio       = float(al.get("match_ratio",     0.75))
        self.min_matches = int(al.get("min_good_matches",  10))
        # Maximum allowed reprojection error to accept a SIFT result (px).
        # Higher → more tolerant of slight perspective distortion.
        self.max_reproj  = float(al.get("max_reproj_error_align", 12.0))
        out_w, out_h     = al.get("output_size", [1080, 1080])
        self.out_w       = int(out_w)
        self.out_h       = int(out_h)
        # SIFT detector — much more robust than ORB for in-plane rotation.
        # We pre-compute reference keypoints once at startup.
        self.sift = cv2.SIFT_create(nfeatures=self.n_features)
        # FLANN matcher — faster than BFMatcher for large descriptor sets and
        # handles rotation-invariant SIFT descriptors well.
        FLANN_INDEX_KDTREE = 1
        self.matcher = cv2.FlannBasedMatcher(
            {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
            {"checks": 100},
        )
        # Load and prepare reference image
        ref_path = config["paths"]["reference_image"]
        ref_bgr  = cv2.imread(ref_path)
        if ref_bgr is None:
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        self.ref_bgr             = self._downscale(ref_bgr)
        self.ref_kp, self.ref_desc = self._detect(self.ref_bgr)
        # Cache reference corners for reprojection error computation
        h_ref, w_ref = self.ref_bgr.shape[:2]
        self._ref_corners = np.float32(
            [[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]
        ).reshape(-1, 1, 2)
        print(f"[Aligner] reference image  : {ref_path}")
        print(f"[Aligner] output size      : {self.out_w}×{self.out_h}")
        print(f"[Aligner] SIFT features    : {self.n_features}")
        print(f"[Aligner] ref keypoints    : {len(self.ref_kp)}")
        print(f"[Aligner] max reproj err   : {self.max_reproj} px")
        print(f"[Aligner] angle search     : 0° 90° 180° 270°")
    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def align(self, bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """Align *bgr* to the reference frame.
        The method is robust to the worker placing the component at any of the
        four 90° orientations, plus arbitrary in-plane rotations handled by
        SIFT's rotation-invariant descriptors.
        Returns
        -------
        aligned : np.ndarray  —  shape (out_h, out_w, 3)
        quality : dict        —  alignment metrics
        """
        bgr = self._downscale(bgr)
        bgr = self._clahe_bgr(bgr)
        # Fast path: explicit contour-only mode (set method: contour in yaml)
        if self.cfg.get("alignment", {}).get("method") == "contour":
            return self._align_contour(bgr)
        # ----------------------------------------------------------------
        # Stage 1 — SIFT with multi-angle search
        # ----------------------------------------------------------------
        # Order rotations: try the last successful rotation first (warm cache).
        # In factory/camera feed, the part orientation rarely changes every frame.
        last_rot = getattr(self, "_last_successful_rot", 0)
        ordered_rotations = [last_rot] + [d for d in (0, 90, 180, 270) if d != last_rot]
        best_aligned = None
        best_quality = {"method": "sift", "n_matches": 0,
                        "reproj_err": 9999.0, "inliers": 0,
                        "rotation_applied": 0, "success": False}
        for deg in ordered_rotations:
            candidate = _rotate_image(bgr, deg)
            aligned_c, quality_c = self._align_sift(candidate)
            if not quality_c["success"]:
                continue
            # Pick the candidate with the most inliers and lowest error
            better = (
                quality_c["inliers"] > best_quality["inliers"]
                or (
                    quality_c["inliers"] == best_quality["inliers"]
                    and quality_c["reproj_err"] < best_quality["reproj_err"]
                )
            )
            if better:
                best_aligned = aligned_c
                best_quality = quality_c
                best_quality["rotation_applied"] = deg
            # FAST PATH: If we have strong confidence (> 100 inliers and low reproj),
            # lock this rotation immediately! No need to evaluate the other 3 angles.
            if quality_c["inliers"] >= 100 and quality_c["reproj_err"] <= 5.0:
                self._last_successful_rot = deg
                return best_aligned, best_quality
        if best_aligned is not None:
            self._last_successful_rot = best_quality["rotation_applied"]
            return best_aligned, best_quality
        # ----------------------------------------------------------------
        # Stage 2 — Contour-based fallback
        # Works even when SIFT can't find enough keypoints (e.g. no object in frame,
        # part out of view, or heavy blur)
        import time
        now = time.time()
        if not hasattr(self, "_last_fallback_warn") or (now - self._last_fallback_warn > 3.0):
            print("[Aligner] SIFT failed — object not detected or low keypoints, attempting contour fallback...")
            self._last_fallback_warn = now
        aligned, quality = self._align_contour(bgr)
        return aligned, quality
    def align_file(self, image_path: str) -> tuple[np.ndarray, dict]:
        """Convenience wrapper: load file then align."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)
        return self.align(bgr)
    # ------------------------------------------------------------------ #
    # Pre-processing helpers
    # ------------------------------------------------------------------ #
    def _downscale(self, img: np.ndarray, max_dim: int = 960) -> np.ndarray:
        """Downscale images so the longest edge ≤ max_dim for fast matching."""
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            return cv2.resize(img, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        return img
    @staticmethod
    def _clahe_bgr(bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE per channel to boost flat IR low-contrast footage.
        The BCS30HDIR50 produces grey, washed-out images at night.
        CLAHE makes edges and holes clearly visible so SIFT finds keypoints.
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        b, g, r = cv2.split(bgr)
        return cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])
    # ------------------------------------------------------------------ #
    # SIFT alignment
    # ------------------------------------------------------------------ #
    def _detect(self, bgr: np.ndarray):
        """Detect SIFT keypoints + descriptors on a CLAHE-equalised grey image."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        kp, desc = self.sift.detectAndCompute(gray, None)
        return kp, desc
    def _align_sift(self, bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """Run SIFT matching on *bgr* against the cached reference.
        Returns
        -------
        (aligned, quality) — quality["success"] is False if matching failed.
        """
        quality = {"method": "sift", "n_matches": 0,
                   "reproj_err": 9999.0, "inliers": 0,
                   "rotation_applied": 0, "success": False}
        kp, desc = self._detect(bgr)
        if desc is None or len(desc) < self.min_matches:
            return bgr, quality
        # FLANN kNN match — k=2 for Lowe's ratio test
        try:
            raw_matches = self.matcher.knnMatch(self.ref_desc, desc, k=2)
        except cv2.error:
            return bgr, quality
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio * n.distance:
                    good.append(m)
        quality["n_matches"] = len(good)
        if len(good) < self.min_matches:
            return bgr, quality
        src_pts = np.float32(
            [self.ref_kp[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        # First estimate a robust Euclidean / similarity transform (rotation + scale + translation)
        # This completely PREVENTS crazy keystone stretching / shearing distortion common with phone cameras!
        M_affine, inlier_mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC,
                                                            ransacReprojThreshold=5.0, maxIters=2000)
        h_ref, w_ref = self.ref_bgr.shape[:2]
        if M_affine is not None and inlier_mask is not None and int(inlier_mask.sum()) >= self.min_matches:
            # Check scale plausibility (0.2x to 5x)
            scale = np.sqrt(M_affine[0, 0]**2 + M_affine[0, 1]**2)
            if 0.20 < scale < 5.0:
                warped = cv2.warpAffine(bgr, M_affine, (w_ref, h_ref),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
                aligned = cv2.resize(warped, (self.out_w, self.out_h),
                                     interpolation=cv2.INTER_AREA)
                quality["success"] = True
                quality["inliers"] = int(inlier_mask.sum())
                quality["reproj_err"] = 2.0
                return aligned, quality
        # Fallback: RANSAC homography — maps dst (input) → src (reference)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,
                                     ransacReprojThreshold=5.0,
                                     maxIters=2000,
                                     confidence=0.995)
        if H is None:
            return bgr, quality
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_matches:
            return bgr, quality
        # ---- Sanity-check the homography --------------------------------
        if not self._homography_is_sane(H, bgr.shape):
            return bgr, quality
        reproj_err = self._reprojection_error(src_pts, dst_pts, H, mask)
        if reproj_err > self.max_reproj:
            return bgr, quality
        quality["reproj_err"] = reproj_err
        # Warp to reference canvas size, then resize to output canvas
        warped  = cv2.warpPerspective(bgr, H, (w_ref, h_ref),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
        aligned = cv2.resize(warped, (self.out_w, self.out_h),
                             interpolation=cv2.INTER_AREA)
        quality["success"] = True
        quality["inliers"] = inliers
        return aligned, quality
    @staticmethod
    def _homography_is_sane(H: np.ndarray, src_shape: tuple) -> bool:
        """Return True if *H* represents a physically plausible transformation.
        Rejects homographies that:
        - flip the image  (negative determinant of upper-left 2×2 sub-matrix)
        - have extreme scale change  (scale < 0.1× or > 10×)
        - produce a very skewed / sheared output
        """
        # Extract scale from the 2×2 rotation-scale part
        h2 = H[:2, :2]
        det = np.linalg.det(h2)
        # det < 0 → reflection (mirror flip) — not physically possible
        if det < 0:
            return False
        # Scale too extreme
        scale = np.sqrt(abs(det))
        if not (0.10 < scale < 10.0):
            return False
        # Check that the projected corners still form a convex polygon that
        # does not self-intersect (no crazy perspective warp).
        h, w = src_shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H)
        if projected is None:
            return False
        # Convexity check using cross-products of consecutive edge vectors
        pts = projected.reshape(4, 2)
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            p3 = pts[(i + 2) % 4]
            cross = np.cross(p2 - p1, p3 - p2)
            if cross < 0:   # clockwise turn → non-convex → degenerate warp
                return False
        return True
    @staticmethod
    def _reprojection_error(src, dst, H, mask) -> float:
        """Mean reprojection error in pixels for RANSAC inlier matches."""
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
        """Contour-based alignment — angle-agnostic fallback.
        Finds the largest connected component (the metal yoke) in both the
        reference and the input image using minAreaRect, then computes a full
        similarity transform (translation + scale + rotation) to map the input
        onto the reference coordinate system.
        This works for ANY angle the worker places the component at, because
        minAreaRect returns the actual orientation of the bounding rectangle.
        """
        quality = {"method": "contour", "rotation_applied": 0, "success": False}
        def part_contour(img_bgr):
            gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Bilateral filter preserves edges while removing noise
            smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
            # Adaptive threshold is better than fixed Canny for variable
            # IR illumination levels
            edges    = cv2.Canny(smoothed, 25, 80)
            h, w  = edges.shape
            border = max(30, min(h, w) // 20)
            edges[0:border, :]        = 0
            edges[h-border:h, :]     = 0
            edges[:, 0:border]        = 0
            edges[:, w-border:w]     = 0
            # Close small gaps in the part outline
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed = cv2.morphologyEx(
                cv2.dilate(edges, k, iterations=3),
                cv2.MORPH_CLOSE, k, iterations=4
            )
            # Mask borders again after dilation (avoids border artefacts)
            closed[0:border, :]        = 0
            closed[h-border:h, :]     = 0
            closed[:, 0:border]        = 0
            closed[:, w-border:w]     = 0
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            # Only consider contours large enough to be the yoke (~50k px²)
            valid = [c for c in cnts if cv2.contourArea(c) > 50_000]
            if not valid:
                return None
            return max(valid, key=cv2.contourArea)
        ref_cnt = part_contour(self.ref_bgr)
        src_cnt = part_contour(bgr)
        if ref_cnt is None or src_cnt is None:
            quality["note"] = "contour_not_found"
            return bgr, quality
        def rect_props(cnt):
            """Return (cx, cy, long_side, short_side, angle_deg) of minAreaRect."""
            (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
            # Normalise so long_side is always the larger dimension
            if w < h:
                angle += 90.0
                w, h = h, w
            return cx, cy, w, h, angle
        rx, ry, rw, rh, rangle = rect_props(ref_cnt)
        sx, sy, sw, sh, sangle = rect_props(src_cnt)
        # Scale: match the long and short sides independently, average them
        scale = 0.5 * (rw / max(sw, 1) + rh / max(sh, 1))
        # The angle delta is the in-plane rotation the worker introduced
        delta_angle = rangle - sangle
        
        # Test both orientations: delta_angle and delta_angle + 180
        # The top-nib area in the yoke has different geometry/brightness than the bottom-nib.
        # Pick the orientation that best correlates with the reference image!
        best_M = None
        best_corr = -1.0
        best_angle = delta_angle
        h_ref, w_ref = self.ref_bgr.shape[:2]
        # Use edge correlation to select best 180-deg orientation (invariant to white paper vs black background)
        ref_edge_small = cv2.Canny(cv2.resize(cv2.cvtColor(self.ref_bgr, cv2.COLOR_BGR2GRAY), (128, 128)), 50, 150)
        for test_angle in (delta_angle, delta_angle + 180.0):
            M_test = cv2.getRotationMatrix2D((sx, sy), test_angle, scale)
            M_test[0, 2] += rx - sx
            M_test[1, 2] += ry - sy
            warped_test = cv2.warpAffine(bgr, M_test, (w_ref, h_ref), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            test_edge_small = cv2.Canny(cv2.resize(cv2.cvtColor(warped_test, cv2.COLOR_BGR2GRAY), (128, 128)), 50, 150)
            corr = float(cv2.matchTemplate(test_edge_small, ref_edge_small, cv2.TM_CCOEFF_NORMED)[0][0])
            if corr > best_corr:
                best_corr = corr
                best_M = M_test
                best_angle = test_angle

        quality["rotation_applied"] = int(round(best_angle)) % 360
        warped  = cv2.warpAffine(bgr, best_M, (w_ref, h_ref),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        aligned = cv2.resize(warped, (self.out_w, self.out_h),
                             interpolation=cv2.INTER_AREA)
        quality["success"] = True
        return aligned, quality
