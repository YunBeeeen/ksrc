"""Map a Nav2 path and command into the route fields of the trained RL observation."""

import numpy as np

import path as pth


class PolicyPath:
    def __init__(self, points):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("path needs at least two 2-D points")
        if not np.isfinite(pts).all():
            raise ValueError("path contains non-finite coordinates")
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("path has zero length")
        self.ref = pth.RefPath(pts)

    def features(self, xy, yaw):
        """Return body-frame lookahead points, signed cross-track and heading errors."""
        ref = self.ref
        xy = np.asarray(xy, dtype=float)
        delta = xy - ref.P[:-1]
        along = np.clip(np.einsum("ij,ij->i", delta, ref.U), 0.0, ref.L)
        closest = ref.P[:-1] + along[:, None] * ref.U
        offset = xy - closest
        j = int(np.argmin(np.einsum("ij,ij->i", offset, offset)))
        s = float(ref.S[j] + along[j])
        cross = float(np.cross(ref.U[j], offset[j]))
        e_y = float(np.linalg.norm(offset[j])) * (-1.0 if cross < 0.0 else 1.0)
        e_psi = pth._wrap(float(yaw) - pth.heading_at(ref, s))
        wp_b = pth.lookahead_body(ref, s, xy, float(yaw))
        return wp_b, e_y, e_psi


def set_route_fields(history, cmd_nom, v_max, features, all_frames=False):
    """Patch the route-related fields in env._frame() without resampling sensors.

    Layout in sim/rl/env.py: cmd 0:3, lookahead 23:27, e_y 27,
    cos/sin(e_psi) 28:30.  A new Nav2 plan replaces route history too.
    """
    wp_b, e_y, e_psi = features
    route = np.array([*np.clip(np.asarray(wp_b).ravel(), -2.0, 2.0),
                      np.clip(e_y / 0.3, -3.0, 3.0),
                      np.cos(e_psi), np.sin(e_psi)], dtype=np.float32)
    cmd = np.asarray(cmd_nom, dtype=np.float32) / np.array(
        [v_max, v_max, 3.0], dtype=np.float32)
    frames = history if all_frames else history[-1:]
    for frame in frames:
        if frame.shape != (33,):
            raise ValueError(f"observation frame layout changed: {frame.shape}")
        frame[:3] = cmd
        frame[23:30] = route
