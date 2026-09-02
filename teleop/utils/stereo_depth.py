"""ZED Mini stereo depth for teleop recording (no ZED SDK / no CUDA required).

The robot's head ZED Mini streams a side-by-side stereo pair (left|right) as a
plain color image through teleimager.  The robot PC (H1-2 PC4) has no NVIDIA
GPU, so the ZED SDK depth pipeline cannot run there.  Instead this module
computes metric depth on the teleop PC from the received stereo pair:

    1. rectify left/right using the Stereolabs FACTORY calibration
       (downloaded per serial number from calib.stereolabs.com, no SDK needed)
    2. StereoSGBM disparity
    3. depth_mm = fx_rect * baseline / disparity  -> uint16 millimeters

Depth is computed in a background worker thread so the 30Hz teleop control
loop is never blocked; the recorder simply picks up the latest finished map.

Calibration file format note: newer files from calib.stereolabs.com list the
distortion as k1..k4 with no p1/p2 (OpenCV fisheye/equidistant model); older
files list k1,k2,p1,p2,k3 (plumb bob).  Both are handled automatically.
"""

import configparser
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import logging_mp
    logger_mp = logging_mp.get_logger(__name__)
except ImportError:  # standalone use outside the teleop env
    import logging
    logger_mp = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


# calibration section suffix by image height of ONE eye
_RES_KEY_BY_HEIGHT = {376: "VGA", 720: "HD", 1080: "FHD", 1242: "2K"}


def _read_cam_section(conf, name):
    sec = conf[name]
    fx, fy = float(sec["fx"]), float(sec["fy"])
    cx, cy = float(sec["cx"]), float(sec["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    keys = {k.lower() for k in sec.keys()}
    if "p1" in keys:  # plumb bob: k1,k2,p1,p2[,k3]
        D = np.array([float(sec.get("k1", 0)), float(sec.get("k2", 0)),
                      float(sec.get("p1", 0)), float(sec.get("p2", 0)),
                      float(sec.get("k3", 0))], dtype=np.float64)
        model = "plumb_bob"
    else:  # fisheye/equidistant: k1..k4
        D = np.array([float(sec.get("k1", 0)), float(sec.get("k2", 0)),
                      float(sec.get("k3", 0)), float(sec.get("k4", 0))],
                     dtype=np.float64).reshape(4, 1)
        model = "fisheye"
    return K, D, model


class ZEDStereoRectifier:
    """Builds rectification maps from a Stereolabs factory .conf file."""

    def __init__(self, calib_path, eye_size):
        """eye_size: (width, height) of ONE eye image, e.g. (1280, 720)."""
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for stereo depth.")
        w, h = int(eye_size[0]), int(eye_size[1])
        res_key = _RES_KEY_BY_HEIGHT.get(h)
        if res_key is None:
            raise ValueError(f"Unsupported eye resolution {eye_size}; "
                             f"expected height in {sorted(_RES_KEY_BY_HEIGHT)}")
        conf = configparser.ConfigParser()
        if not conf.read(calib_path):
            raise FileNotFoundError(f"Cannot read ZED calibration: {calib_path}")

        K_l, D_l, model = _read_cam_section(conf, f"LEFT_CAM_{res_key}")
        K_r, D_r, _ = _read_cam_section(conf, f"RIGHT_CAM_{res_key}")
        st = conf["STEREO"]
        baseline_mm = float(st["Baseline"])
        ty = float(st.get("TY", 0.0))
        tz = float(st.get("TZ", 0.0))
        rx = float(st.get(f"RX_{res_key}", st.get("RX_HD", 0.0)))
        cv_ = float(st.get(f"CV_{res_key}", st.get("CV_HD", 0.0)))
        rz = float(st.get(f"RZ_{res_key}", st.get("RZ_HD", 0.0)))

        R, _ = cv2.Rodrigues(np.array([rx, cv_, rz], dtype=np.float64))
        # OpenCV convention: T maps left-cam coords into right-cam coords.
        T = np.array([-baseline_mm, ty, tz], dtype=np.float64) / 1000.0  # meters
        size = (w, h)

        if model == "fisheye":
            R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
                K_l, D_l, K_r, D_r, size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0)
            self.map_lx, self.map_ly = cv2.fisheye.initUndistortRectifyMap(
                K_l, D_l, R1, P1, size, cv2.CV_16SC2)
            self.map_rx, self.map_ry = cv2.fisheye.initUndistortRectifyMap(
                K_r, D_r, R2, P2, size, cv2.CV_16SC2)
        else:
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                K_l, D_l, K_r, D_r, size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
            self.map_lx, self.map_ly = cv2.initUndistortRectifyMap(
                K_l, D_l, R1, P1, size, cv2.CV_16SC2)
            self.map_rx, self.map_ry = cv2.initUndistortRectifyMap(
                K_r, D_r, R2, P2, size, cv2.CV_16SC2)

        self.model = model
        self.size = size
        self.fx = float(P1[0, 0])
        self.cx = float(P1[0, 2])
        self.cy = float(P1[1, 2])
        self.baseline_m = abs(float(P2[0, 3]) / float(P2[0, 0]))
        logger_mp.info(
            "[stereo depth] calib=%s res=%s model=%s fx_rect=%.2f baseline=%.4fm",
            calib_path, res_key, model, self.fx, self.baseline_m)

    def rectify(self, left_bgr, right_bgr):
        l = cv2.remap(left_bgr, self.map_lx, self.map_ly, cv2.INTER_LINEAR)
        r = cv2.remap(right_bgr, self.map_rx, self.map_ry, cv2.INTER_LINEAR)
        return l, r

    def intrinsics(self, scale=1.0):
        return {
            "model": self.model,
            "width": int(round(self.size[0] * scale)),
            "height": int(round(self.size[1] * scale)),
            "fx": self.fx * scale,
            "fy": self.fx * scale,
            "cx": self.cx * scale,
            "cy": self.cy * scale,
            "baseline_m": self.baseline_m,
            "depth_unit": "mm(uint16), 0=invalid",
        }


class StereoDepthEstimator:
    """Rectify + SGBM + disparity->uint16 mm depth (left-eye aligned)."""

    def __init__(self, calib_path, eye_size, scale=0.5,
                 num_disparities=128, block_size=5, max_depth_mm=8000):
        self.rectifier = ZEDStereoRectifier(calib_path, eye_size)
        self.scale = float(scale)
        self.max_depth_mm = int(max_depth_mm)
        num_disparities = max(16, (int(num_disparities) // 16) * 16)
        bs = max(3, int(block_size) | 1)
        cn = 1  # grayscale matching
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=bs,
            P1=8 * cn * bs * bs,
            P2=32 * cn * bs * bs,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        # fx scaled to the disparity-computation resolution
        self.fx_scaled = self.rectifier.fx * self.scale
        self.baseline_m = self.rectifier.baseline_m

    def intrinsics(self):
        info = self.rectifier.intrinsics(self.scale)
        info["max_depth_mm"] = self.max_depth_mm
        return info

    def compute_from_sbs(self, sbs_bgr):
        """side-by-side (left|right) frame -> uint16 depth in mm (left aligned)."""
        half = sbs_bgr.shape[1] // 2
        return self.compute(sbs_bgr[:, :half], sbs_bgr[:, half:])

    def compute(self, left_bgr, right_bgr):
        l, r = self.rectifier.rectify(left_bgr, right_bgr)
        if self.scale != 1.0:
            l = cv2.resize(l, None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_AREA)
            r = cv2.resize(r, None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_AREA)
        lg = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        disp = self.sgbm.compute(lg, rg).astype(np.float32) / 16.0
        valid = disp > 0.5
        depth_mm = np.zeros(disp.shape, dtype=np.uint16)
        # depth[m] = fx[px] * B[m] / disp[px]; store millimeters
        d = (self.fx_scaled * self.baseline_m * 1000.0) / np.maximum(disp, 1e-6)
        d[~valid] = 0
        d[d > self.max_depth_mm] = 0
        depth_mm[:] = d.astype(np.uint16)
        return depth_mm


class AsyncStereoDepthWorker:
    """Background depth computation; the control loop never blocks.

    submit() hands over the newest side-by-side frame (older pending frame is
    dropped).  get_latest() returns the most recent finished (depth_mm, ts).
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self._pending = None       # (sbs_bgr, ts)
        self._latest = None        # (depth_mm, ts)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = False
        self._compute_count = 0
        self._compute_time_sum = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, sbs_bgr, ts=None):
        with self._lock:
            self._pending = (sbs_bgr, time.time() if ts is None else ts)
        self._event.set()

    def get_latest(self):
        with self._lock:
            return self._latest

    def stats(self):
        with self._lock:
            n, s = self._compute_count, self._compute_time_sum
        return {"count": n, "avg_ms": (s / n * 1000.0) if n else 0.0}

    def stop(self):
        self._stop = True
        self._event.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop:
            self._event.wait(timeout=0.5)
            self._event.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            sbs, ts = job
            try:
                t0 = time.time()
                depth = self.estimator.compute_from_sbs(sbs)
                dt = time.time() - t0
                with self._lock:
                    self._latest = (depth, ts)
                    self._compute_count += 1
                    self._compute_time_sum += dt
            except Exception as exc:
                logger_mp.warning("[stereo depth] compute failed: %s", exc)


def _main():
    """Offline smoke test: python -m teleop.utils.stereo_depth <calib> [sbs_image]"""
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("calib")
    p.add_argument("image", nargs="?", help="side-by-side stereo image (e.g. 2560x720)")
    p.add_argument("--scale", type=float, default=0.5)
    a = p.parse_args()
    if a.image:
        sbs = cv2.imread(a.image)
        assert sbs is not None, f"cannot read {a.image}"
        eye = (sbs.shape[1] // 2, sbs.shape[0])
    else:  # synthetic shifted pattern
        rng = np.random.default_rng(0)
        left = (rng.random((720, 1280, 3)) * 255).astype(np.uint8)
        left = cv2.GaussianBlur(left, (0, 0), 2)
        shift = 40
        right = np.roll(left, -shift, axis=1)
        sbs = np.concatenate([left, right], axis=1)
        eye = (1280, 720)
        print(f"synthetic pattern with {shift}px uniform shift")
    est = StereoDepthEstimator(a.calib, eye, scale=a.scale)
    t0 = time.time()
    depth = est.compute_from_sbs(sbs)
    dt = time.time() - t0
    v = depth[depth > 0]
    print(f"depth shape={depth.shape} dtype={depth.dtype} compute={dt*1000:.1f}ms")
    print(f"valid={v.size}/{depth.size} ({100.0*v.size/depth.size:.1f}%) "
          f"min={v.min() if v.size else 0}mm median={int(np.median(v)) if v.size else 0}mm "
          f"max={v.max() if v.size else 0}mm")
    print("intrinsics:", est.intrinsics())
    if a.image:
        vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=255.0 / max(1, est.max_depth_mm)),
            cv2.COLORMAP_JET)
        out = a.image + ".depth.png"
        cv2.imwrite(out, depth)
        cv2.imwrite(a.image + ".depth_vis.jpg", vis)
        print("saved:", out)


if __name__ == "__main__":
    _main()
