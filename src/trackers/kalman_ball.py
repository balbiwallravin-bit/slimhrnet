import numpy as np


class KalmanBallTracker:
    def __init__(self, cfg):
        tracker_cfg = cfg["tracker"]
        self._dt = float(tracker_cfg.get("dt", 1.0 / 60.0))
        self._max_miss = int(tracker_cfg.get("max_miss", 8))
        self._gate_radius = float(tracker_cfg.get("gate_radius", 80.0))
        self._score_high = float(tracker_cfg.get("score_high", 0.5))
        self._score_low = float(tracker_cfg.get("score_low", 0.2))
        self.refresh()

    def refresh(self):
        self._state = None
        self._P = None
        self._miss_count = 0
        self._last_score = 0.0

        self._F = np.array(
            [
                [1.0, 0.0, self._dt, 0.0],
                [0.0, 1.0, 0.0, self._dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        self._Q = np.diag([1.0, 1.0, 50.0, 50.0]).astype(np.float32)
        self._R = np.diag([4.0, 4.0]).astype(np.float32)

    def _init_kf(self, x, y):
        self._state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self._P = np.eye(4, dtype=np.float32) * 100.0

    def _predict(self):
        if self._state is None:
            return None
        self._state = self._F @ self._state
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._state[:2].copy()

    def _update_kf(self, measurement):
        z = np.asarray(measurement, dtype=np.float32)
        innovation = z - (self._H @ self._state)
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._state = self._state + K @ innovation
        self._P = (np.eye(4, dtype=np.float32) - K @ self._H) @ self._P
        return self._state[:2].copy()

    def _select_measurement(self, frame_dets):
        if not frame_dets:
            return None, 0.0, "no_det"
        best = max(frame_dets, key=lambda det: float(det.get("score", 0.0)))
        best_xy = np.asarray(best["xy"], dtype=np.float32)
        best_score = float(best.get("score", 0.0))
        if best_score >= self._score_high:
            return best_xy, best_score, "high"
        if best_score >= self._score_low:
            return best_xy, best_score, "mid"
        return None, best_score, "low"

    def update(self, frame_dets):
        measurement, det_score, det_mode = self._select_measurement(frame_dets)

        if self._state is None and measurement is None:
            return {"x": None, "y": None, "visi": False, "score": det_score, "status": "no_state"}

        pred = self._predict()

        if measurement is None:
            self._miss_count += 1
            if self._miss_count > self._max_miss:
                self._state = None
                self._P = None
                return {"x": None, "y": None, "visi": False, "score": det_score, "status": "lost"}
            if pred is None:
                return {"x": None, "y": None, "visi": False, "score": det_score, "status": "no_state"}
            return {
                "x": float(pred[0]),
                "y": float(pred[1]),
                "visi": True,
                "score": det_score,
                "status": "predicted",
            }

        if self._state is None:
            self._init_kf(float(measurement[0]), float(measurement[1]))
            self._miss_count = 0
            return {
                "x": float(measurement[0]),
                "y": float(measurement[1]),
                "visi": True,
                "score": det_score,
                "status": "init",
            }

        if det_mode == "mid" and pred is not None:
            dist = float(np.linalg.norm(measurement - pred))
            if dist > self._gate_radius:
                self._miss_count += 1
                return {
                    "x": float(pred[0]),
                    "y": float(pred[1]),
                    "visi": True,
                    "score": det_score,
                    "status": "gated",
                }

        updated = self._update_kf(measurement)
        self._miss_count = 0
        return {
            "x": float(updated[0]),
            "y": float(updated[1]),
            "visi": True,
            "score": det_score,
            "status": "updated",
        }
