#!/usr/bin/env python3
"""
ROS 2 Node: Vision & Maze Solver with Live Preview
Wraps test_detect_wo_aruco.py functionality with ROS 2 interface

Publishes:
    /maze/status (std_msgs/String): Current status
    /maze/waypoints_pixel (std_msgs/Float32MultiArray): Pixel waypoints
    /maze/info (std_msgs/String): Grid size and path info
    /maze/image (sensor_msgs/Image): Camera snapshot (optional)

Subscribes:
    /maze/command (std_msgs/String): Commands from Claude
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from collections import deque
import csv
import threading

# ========================= USER CONSTANTS =========================
CAM_INDEX = 2
CAP_WIDTH = 640
CAP_HEIGHT = 480
OUT_SIZE = 600
MARGIN_PX = 2
DRAW_GUIDES = True

# Main folder - 3 latest images (overwritten each time)
MAIN_OUTPUT_DIR = Path("/home/nivas/maze_solver_ws/install/lib/python3.10/site-packages/maze_solver")
# Archive folder - all timestamped images
ARCHIVE_DIR = Path("/home/nivas/maze_solver_ws/maze_archive")
# CSV output for converter node
CSV_OUTPUT_DIR = Path("maze_captures")
CSV_OUTPUT_FILENAME = "waypoints_pixel.csv"

TARGET_PREVIEW_FPS = 6
FRAME_DELAY_MS = max(1, int(1000 / max(0.5, TARGET_PREVIEW_FPS)))

MARKER_HOLD_SEC = 3.0
QUAD_SCORE_THRESHOLD = 0.45
PIXEL_STABILITY_TOL = 8.0
# =================================================================

def _project_points(H, pts):
    pts = np.asarray(pts, dtype=np.float32)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    P = np.hstack([pts, ones]) @ H.T
    P = P[:, :2] / P[:, 2:3]
    return P

def _order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _quad_score_rectangularity(quad: np.ndarray) -> float:
    def angle_cos(p0, p1, p2):
        v1 = p0 - p1; v2 = p2 - p1
        denom = (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
        return abs(np.dot(v1, v2)/denom)
    cos_sum = 0.0
    for i in range(4):
        cos_sum += angle_cos(quad[(i-1)%4], quad[i], quad[(i+1)%4])
    return 1.0/(1.0 + cos_sum)

def detect_maze_quad(gray: np.ndarray):
    """Return (src_quad_tl_tr_br_bl, score) or (None, 0.0)."""
    H, W = gray.shape[:2]
    g = cv2.GaussianBlur(gray, (5,5), 0)
    
    thr = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 7
    )
    thr = cv2.medianBlur(thr, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.Canny(thr, 60, 180)
    
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    
    best = (None, 0.0)
    img_area = float(H * W)
    for c in cnts:
        if cv2.contourArea(c) < 0.02 * img_area:
            continue
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(-1, 2).astype(np.float32)
        rect_score = _quad_score_rectangularity(quad)
        area_score = min(1.0, cv2.contourArea(approx) / img_area / 0.6)
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect = max(w, h) / max(1, min(w, h))
        asp_score = 1.0 if 0.7 <= aspect <= 1.4 else max(0.0, 1.4 / aspect - 0.5)
        score = 0.55 * rect_score + 0.35 * area_score + 0.10 * asp_score
        if score > best[1]:
            best = (quad, score)
    
    if best[0] is None or best[1] < 0.35:
        return None, 0.0
    return _order_quad_tl_tr_br_bl(best[0]), float(best[1])

def warp_square(img, src_pts, out_size=OUT_SIZE, inner_margin=MARGIN_PX):
    dst = np.array([[0, 0],
                    [out_size - 1, 0],
                    [out_size - 1, out_size - 1],
                    [0, out_size - 1]], dtype=np.float32)
    H0 = cv2.getPerspectiveTransform(src_pts, dst)
    
    warped = cv2.warpPerspective(img, H0, (out_size, out_size), flags=cv2.INTER_LINEAR)
    
    if inner_margin > 0:
        m = inner_margin
        warped = warped[m:out_size - m, m:out_size - m]
        warped = cv2.resize(warped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        s = out_size / float(out_size - 2 * m)
        A = np.array([[s, 0, -s * m],
                      [0, s, -s * m],
                      [0, 0, 1     ]], dtype=np.float32)
        H_total = A @ H0
    else:
        H_total = H0
    
    return warped, H_total

class StabilityLock:
    """Tracks quad stability over time"""
    def __init__(self, hold_sec=3.0, px_tol=8.0):
        self.hold_sec = hold_sec
        self.px_tol = px_tol
        self._last = None
        self._since = None
    
    def update(self, quad):
        now = time.time()
        if quad is None:
            self._last = None
            self._since = None
            return 0.0
        if self._last is None:
            self._last = quad
            self._since = now
            return 0.0
        d = np.linalg.norm((quad - self._last), axis=1).mean()
        if d < self.px_tol:
            return now - (self._since or now)
        else:
            self._last = quad
            self._since = now
            return 0.0
    
    def reset(self):
        self._last = None
        self._since = None

def _snap_extend(lines, length, snap_tol, extend_tol):
    if not lines:
        return []
    lines = sorted(int(x) for x in lines)
    if lines[0] < snap_tol:
        lines[0] = 0
    if (length - 1) - lines[-1] < snap_tol:
        lines[-1] = length - 1
    if len(lines) >= 2:
        _ = np.median(np.diff(lines))
    else:
        _ = max(1, length // 5)
    if lines[0] > extend_tol:
        lines = [0] + lines
    if (length - 1) - lines[-1] > extend_tol:
        lines = lines + [length - 1]
    dedup = []
    for v in lines:
        if not dedup or abs(v - dedup[-1]) > 1:
            dedup.append(v)
    return dedup

class LiveMazeAnalyzer:
    def __init__(self):
        self.last_analysis = None

    def _cluster_lines(self, positions, min_distance=10):
        if not positions:
            return []
        positions = sorted(positions)
        clusters, cur = [], [positions[0]]
        for p in positions[1:]:
            if p - cur[-1] < min_distance:
                cur.append(p)
            else:
                clusters.append(int(np.mean(cur)))
                cur = [p]
        clusters.append(int(np.mean(cur)))
        return clusters

    def detect_grid_size(self, binary_image):
        h, w = binary_image.shape
        den = cv2.medianBlur(binary_image, 3)
        pad = 4
        den_pad = cv2.copyMakeBorder(den, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        kx = max(5, (w // 22) | 1)
        ky = max(5, (h // 22) | 1)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
        vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, ky))

        horiz_bands_pad = cv2.morphologyEx(den_pad, cv2.MORPH_OPEN, horiz_kernel)
        vert_bands_pad  = cv2.morphologyEx(den_pad, cv2.MORPH_OPEN, vert_kernel)

        horiz_bands = horiz_bands_pad[pad:-pad, pad:-pad]
        vert_bands  = vert_bands_pad[pad:-pad, pad:-pad]

        hp = np.sum(horiz_bands > 0, axis=1).astype(np.float32)
        vp = np.sum(vert_bands  > 0, axis=0).astype(np.float32)

        hp = cv2.blur(hp.reshape(-1, 1), (21, 1)).ravel()
        vp = cv2.blur(vp.reshape(1, -1), (1, 21)).ravel()

        h_thr = max(np.percentile(hp, 70) * 0.9, 0.2 * (hp.max() if hp.max() > 0 else 1.0))
        v_thr = max(np.percentile(vp, 70) * 0.9, 0.2 * (vp.max() if vp.max() > 0 else 1.0))

        h_idx = np.where(hp >= h_thr)[0].tolist()
        v_idx = np.where(vp >= v_thr)[0].tolist()

        h_lines = self._cluster_lines(h_idx, min_distance=max(2, h // 18))
        v_lines = self._cluster_lines(v_idx, min_distance=max(2, w // 18))

        snap_tol_h = max(2, h // 60)
        snap_tol_v = max(2, w // 60)
        est_h_gap = np.median(np.diff(sorted(h_lines))) if len(h_lines) >= 2 else max(1, h // 5)
        est_v_gap = np.median(np.diff(sorted(v_lines))) if len(v_lines) >= 2 else max(1, w // 5)
        extend_tol_h = int(0.60 * est_h_gap)
        extend_tol_v = int(0.60 * est_v_gap)

        def _try_fill(lines, length):
            if len(lines) >= 2:
                ls = sorted(lines)
                gaps = np.diff(ls)
                med = np.median(gaps) if len(gaps) else max(1, length // 5)
                ins = []
                for i, g in enumerate(gaps):
                    if g > 1.75 * med:
                        ins.append(int(ls[i] + round(g / 2)))
                if ins:
                    lines = sorted(lines + ins)
            return lines

        h_lines = _snap_extend(h_lines, h, snap_tol_h, extend_tol_h)
        v_lines = _snap_extend(v_lines, w, snap_tol_v, extend_tol_v)
        if len(h_lines) - 1 <= 4:
            h_lines = _try_fill(h_lines, h)
        if len(v_lines) - 1 <= 4:
            v_lines = _try_fill(v_lines, w)

        rows = len(h_lines) - 1 if len(h_lines) > 1 else 0
        cols = len(v_lines) - 1 if len(v_lines) > 1 else 0
        rows = max(3, min(10, rows))
        cols = max(3, min(10, cols))

        grid_debug = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        for yy in h_lines:
            cv2.line(grid_debug, (0, yy), (w - 1, yy), (0, 255, 0), 2)
        for xx in v_lines:
            cv2.line(grid_debug, (xx, 0), (xx, h - 1), (255, 0, 0), 2)

        edges_full = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        return (rows, cols, grid_debug, edges_full)

    def extract_grid_lines(self, binary_image):
        h, w = binary_image.shape
        den = cv2.medianBlur(binary_image, 3)
        pad = 4
        den_pad = cv2.copyMakeBorder(den, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        kx = max(5, (w // 22) | 1)
        ky = max(5, (h // 22) | 1)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
        vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, ky))

        hb = cv2.morphologyEx(den_pad, cv2.MORPH_OPEN, horiz_kernel)[pad:-pad, pad:-pad]
        vb = cv2.morphologyEx(den_pad, cv2.MORPH_OPEN, vert_kernel )[pad:-pad, pad:-pad]

        hp = cv2.blur(np.sum(hb > 0, axis=1).astype(np.float32).reshape(-1,1), (21,1)).ravel()
        vp = cv2.blur(np.sum(vb > 0, axis=0).astype(np.float32).reshape(1,-1), (1,21)).ravel()

        h_thr = max(np.percentile(hp, 70) * 0.9, 0.2 * (hp.max() if hp.max() > 0 else 1.0))
        v_thr = max(np.percentile(vp, 70) * 0.9, 0.2 * (vp.max() if vp.max() > 0 else 1.0))

        h_idx = np.where(hp >= h_thr)[0].tolist()
        v_idx = np.where(vp >= v_thr)[0].tolist()

        h_lines = self._cluster_lines(h_idx, min_distance=max(2, h // 18))
        v_lines = self._cluster_lines(v_idx, min_distance=max(2, w // 18))

        snap_tol_h = max(2, h // 60)
        snap_tol_v = max(2, w // 60)
        est_h_gap = np.median(np.diff(sorted(h_lines))) if len(h_lines) >= 2 else max(1, h // 5)
        est_v_gap = np.median(np.diff(sorted(v_lines))) if len(v_lines) >= 2 else max(1, w // 5)
        extend_tol_h = int(0.60 * est_h_gap)
        extend_tol_v = int(0.60 * est_v_gap)

        def _try_fill(lines, length):
            if len(lines) >= 2:
                ls = sorted(lines)
                gaps = np.diff(ls)
                med = np.median(gaps) if len(gaps) else max(1, length // 5)
                ins = []
                for i, g in enumerate(gaps):
                    if g > 1.75 * med:
                        ins.append(int(ls[i] + round(g / 2)))
                if ins:
                    lines = sorted(lines + ins)
            return lines

        h_lines = _snap_extend(h_lines, h, snap_tol_h, extend_tol_h)
        v_lines = _snap_extend(v_lines, w, snap_tol_v, extend_tol_v)
        if len(h_lines) - 1 <= 4:
            h_lines = _try_fill(h_lines, h)
        if len(v_lines) - 1 <= 4:
            v_lines = _try_fill(v_lines, w)

        return h_lines, v_lines

    def detect_entrance_exit(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)

        def _center(mask):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)
                if M['m00'] != 0:
                    return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            return None

        return _center(green_mask), _center(red_mask)

    def _carve_dots_from_binary(self, warped_bgr, binary):
        hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
        g  = cv2.inRange(hsv, np.array([35, 50, 50]),  np.array([85, 255, 255]))
        r1 = cv2.inRange(hsv, np.array([0, 50, 50]),  np.array([10, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        dots = cv2.bitwise_or(g, cv2.bitwise_or(r1, r2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dots = cv2.dilate(dots, kernel, iterations=1)
        out = binary.copy()
        out[dots > 0] = 0
        return out

    def _clean_binary_for_graph(self, binary):
        b = cv2.medianBlur(binary, 3)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=1)
        return b

    def _cell_of_point(self, pt, h_lines, v_lines):
        if pt is None:
            return None
        x, y = pt
        def idx(val, lines):
            val = max(lines[0] + 1, min(val, lines[-1] - 2))
            j = int(np.searchsorted(lines, val, side="right")) - 1
            return max(0, min(j, len(lines) - 2))
        r = idx(y, h_lines)
        c = idx(x, v_lines)
        return (r, c)

    def _cell_center(self, r, c, h_lines, v_lines):
        cx = (v_lines[c] + v_lines[c+1]) // 2
        cy = (h_lines[r] + h_lines[r+1]) // 2
        return (cx, cy)
    
    def _add_corner_waypoints(self, path_cells, h_lines, v_lines):
        """
        Insert intermediate waypoints at corners to force robot to follow walls
        instead of cutting diagonally through corners.
        
        For each turn in the path, adds a waypoint at the corner intersection.
        """
        if len(path_cells) < 3:
            # No corners to add for paths with less than 3 cells
            return [self._cell_center(r, c, h_lines, v_lines) for (r, c) in path_cells]
        
        waypoints = []
        
        for i in range(len(path_cells)):
            r, c = path_cells[i]
            cx, cy = self._cell_center(r, c, h_lines, v_lines)
            
            if i == 0:
                # First waypoint - just add cell center
                waypoints.append((cx, cy))
            elif i == len(path_cells) - 1:
                # Last waypoint - just add cell center
                waypoints.append((cx, cy))
            else:
                # Middle waypoint - check if there's a direction change (corner)
                prev_r, prev_c = path_cells[i-1]
                next_r, next_c = path_cells[i+1]
                
                # Calculate movement directions
                dr_in = r - prev_r   # -1=up, 0=same, 1=down
                dc_in = c - prev_c   # -1=left, 0=same, 1=right
                dr_out = next_r - r
                dc_out = next_c - c
                
                # Check if direction changes (corner detected)
                if (dr_in, dc_in) != (dr_out, dc_out):
                    # This is a corner! Add waypoint at the edge, not center
                    
                    # Determine corner position based on turn direction
                    if dr_in != 0 and dc_out != 0:
                        # Vertical then horizontal turn
                        # Use the edge where the turn happens
                        corner_x = (v_lines[c] + v_lines[c+1]) // 2
                        if dr_in < 0:  # Coming from below
                            corner_y = h_lines[r]
                        else:  # Coming from above
                            corner_y = h_lines[r+1]
                        waypoints.append((corner_x, corner_y))
                        
                        # Add second point for the turn
                        if dc_out < 0:  # Going left
                            corner_x2 = v_lines[c]
                        else:  # Going right
                            corner_x2 = v_lines[c+1]
                        corner_y2 = corner_y
                        waypoints.append((corner_x2, corner_y2))
                        
                    elif dc_in != 0 and dr_out != 0:
                        # Horizontal then vertical turn
                        if dc_in < 0:  # Coming from right
                            corner_x = v_lines[c]
                        else:  # Coming from left
                            corner_x = v_lines[c+1]
                        corner_y = (h_lines[r] + h_lines[r+1]) // 2
                        waypoints.append((corner_x, corner_y))
                        
                        # Add second point for the turn
                        corner_x2 = corner_x
                        if dr_out < 0:  # Going up
                            corner_y2 = h_lines[r]
                        else:  # Going down
                            corner_y2 = h_lines[r+1]
                        waypoints.append((corner_x2, corner_y2))
                    else:
                        # Straight line - just use center
                        waypoints.append((cx, cy))
                else:
                    # No corner - straight line, use center
                    waypoints.append((cx, cy))
        
        return waypoints

    def _neighbors_open(self, binary, r, c, h_lines, v_lines):
        H, W = binary.shape
        cx = (v_lines[c] + v_lines[c+1]) // 2
        cy = (h_lines[r] + h_lines[r+1]) // 2

        stripe_w = max(1, (v_lines[c+1] - v_lines[c]) // 20)
        stripe_h = max(1, (h_lines[r+1] - h_lines[r]) // 20)
        band_t = 3

        def is_free(mean_val):
            return mean_val < 40.0

        opens = {}
        if r > 0:
            yb = h_lines[r]
            xs = slice(max(0, cx - stripe_w), min(W, cx + stripe_w + 1))
            ys_a = slice(max(0, yb - band_t - 1), max(0, yb - 1))
            ys_m = slice(max(0, yb - band_t//2), min(H, yb + band_t//2 + 1))
            ys_b = slice(min(H, yb + 1), min(H, yb + band_t + 1))
            opens['U'] = all([
                is_free(np.mean(binary[ys_a, xs])),
                is_free(np.mean(binary[ys_m, xs])),
                is_free(np.mean(binary[ys_b, xs]))
            ])

        if r < len(h_lines) - 2:
            yb = h_lines[r+1]
            xs = slice(max(0, cx - stripe_w), min(W, cx + stripe_w + 1))
            ys_a = slice(max(0, yb - band_t - 1), max(0, yb - 1))
            ys_m = slice(max(0, yb - band_t//2), min(H, yb + band_t//2 + 1))
            ys_b = slice(min(H, yb + 1), min(H, yb + band_t + 1))
            opens['D'] = all([
                is_free(np.mean(binary[ys_a, xs])),
                is_free(np.mean(binary[ys_m, xs])),
                is_free(np.mean(binary[ys_b, xs]))
            ])

        if c > 0:
            xb = v_lines[c]
            ys = slice(max(0, cy - stripe_h), min(H, cy + stripe_h + 1))
            xs_a = slice(max(0, xb - band_t - 1), max(0, xb - 1))
            xs_m = slice(max(0, xb - band_t//2), min(W, xb + band_t//2 + 1))
            xs_b = slice(min(W, xb + 1), min(W, xb + band_t + 1))
            opens['L'] = all([
                is_free(np.mean(binary[ys, xs_a])),
                is_free(np.mean(binary[ys, xs_m])),
                is_free(np.mean(binary[ys, xs_b]))
            ])

        if c < len(v_lines) - 2:
            xb = v_lines[c+1]
            ys = slice(max(0, cy - stripe_h), min(H, cy + stripe_h + 1))
            xs_a = slice(max(0, xb - band_t - 1), max(0, xb - 1))
            xs_m = slice(max(0, xb - band_t//2), min(W, xb + band_t//2 + 1))
            xs_b = slice(min(W, xb + 1), min(W, xb + band_t + 1))
            opens['R'] = all([
                is_free(np.mean(binary[ys, xs_a])),
                is_free(np.mean(binary[ys, xs_m])),
                is_free(np.mean(binary[ys, xs_b]))
            ])

        return opens

    def _build_graph(self, binary, h_lines, v_lines):
        rows = len(h_lines) - 1
        cols = len(v_lines) - 1
        graph = { (r,c): [] for r in range(rows) for c in range(cols) }
        for r in range(rows):
            for c in range(cols):
                op = self._neighbors_open(binary, r, c, h_lines, v_lines)
                if op.get('U'): graph[(r,c)].append((r-1, c))
                if op.get('D'): graph[(r,c)].append((r+1, c))
                if op.get('L'): graph[(r,c)].append((r, c-1))
                if op.get('R'): graph[(r,c)].append((r, c+1))
        return graph

    def _bfs_shortest_path(self, graph, start, goal):
        if start is None or goal is None:
            return None
        q = deque([start])
        parent = {start: None}
        while q:
            u = q.popleft()
            if u == goal:
                path = []
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for v in graph.get(u, []):
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        return None

    def _verify_solution_warped(self, binary_inv_255, path_pts_warped):
        if not path_pts_warped or len(path_pts_warped) < 2:
            return {"ok": False, "reason": "no_path"}
        collisions = 0
        samples = 0
        for (x0,y0), (x1,y1) in zip(path_pts_warped[:-1], path_pts_warped[1:]):
            N = max(8, int(np.hypot(x1-x0, y1-y0)//3))
            xs = np.linspace(x0, x1, N).astype(int)
            ys = np.linspace(y0, y1, N).astype(int)
            ys = np.clip(ys, 0, binary_inv_255.shape[0]-1)
            xs = np.clip(xs, 0, binary_inv_255.shape[1]-1)
            vals = binary_inv_255[ys, xs]
            samples += N
            collisions += int((vals > 120).sum())
        coll_ratio = collisions/max(1, samples)
        return {"ok": coll_ratio < 0.05, "coll_ratio": float(coll_ratio), "samples": int(samples)}

    def analyze_snapshot(self, warped_frame, H_total, original_frame, start_color: str):
        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
        )

        binary = self._carve_dots_from_binary(warped_frame, binary)
        binary = self._clean_binary_for_graph(binary)

        rows, cols, grid_debug, edges_full = self.detect_grid_size(binary)
        grid_size = (rows, cols)

        green_center, red_center = self.detect_entrance_exit(warped_frame)
        sc = (start_color or 'green').strip().lower()
        if sc.startswith('r'):
            entrance, exit_point = red_center, green_center
            chosen = 'red'
        else:
            entrance, exit_point = green_center, red_center
            chosen = 'green'

        result_image = warped_frame.copy()
        if entrance:
            cv2.circle(result_image, entrance, 15, (0, 255, 0), 3)
            cv2.putText(result_image, "START", (entrance[0]-30, entrance[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if exit_point:
            cv2.circle(result_image, exit_point, 15, (0, 0, 255), 3)
            cv2.putText(result_image, "END", (exit_point[0]-20, exit_point[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result_image, f"Grid: {grid_size[0]}x{grid_size[1]}  Start: {chosen}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        h_lines, v_lines = self.extract_grid_lines(binary)
        graph = self._build_graph(binary, h_lines, v_lines)
        start_cell = self._cell_of_point(entrance, h_lines, v_lines)
        goal_cell  = self._cell_of_point(exit_point, h_lines, v_lines)
        path_cells = self._bfs_shortest_path(graph, start_cell, goal_cell)

        overlay = warped_frame.copy()
        path_pts_warped = []
        if path_cells:
            # SIMPLE: Just use cell centers - BFS path is already correct!
            path_pts_warped = [self._cell_center(r, c, h_lines, v_lines) for (r, c) in path_cells]
            for i in range(len(path_pts_warped)-1):
                cv2.arrowedLine(overlay, path_pts_warped[i], path_pts_warped[i+1], (255, 0, 255), 2, cv2.LINE_AA)
            if path_pts_warped:
                cv2.circle(overlay, path_pts_warped[0], 8, (0, 255, 0), -1)
                cv2.circle(overlay, path_pts_warped[-1], 8, (0, 0, 255), -1)

        H_inv = np.linalg.inv(H_total)
        waypoints_original = []
        if path_pts_warped:
            waypoints_original = _project_points(H_inv, path_pts_warped).astype(int).tolist()

        entry_original = _project_points(H_inv, [entrance]).astype(int).tolist()[0] if entrance else None
        exit_original  = _project_points(H_inv, [exit_point]).astype(int).tolist()[0] if exit_point else None

        solved_on_camera = original_frame.copy()
        if waypoints_original:
            for i in range(len(waypoints_original)-1):
                cv2.arrowedLine(solved_on_camera, tuple(waypoints_original[i]),
                         tuple(waypoints_original[i+1]), (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(solved_on_camera, tuple(waypoints_original[0]), 8, (0, 255, 0), -1)
            cv2.circle(solved_on_camera, tuple(waypoints_original[-1]), 8, (0, 0, 255), -1)

        verify = self._verify_solution_warped(binary, path_pts_warped)

        self.last_analysis = {
            'entrance': entrance,
            'exit': exit_point,
            'grid_size': grid_size,
            'result_image': result_image,
            'binary_image': binary,
            'original_warped': warped_frame,
            'grid_debug': grid_debug,
            'edges': edges_full,
            'solved_overlay': overlay,
            'entry_original': tuple(entry_original) if entry_original else None,
            'exit_original':  tuple(exit_original)  if exit_original  else None,
            'waypoints_original': [tuple(p) for p in waypoints_original],
            'solved_on_camera': solved_on_camera,
            'camera_frame': original_frame.copy(),
            'path_cells': path_cells,
            'verify': verify
        }
        return self.last_analysis


class VisionSolverNode(Node):
    """ROS 2 Node for vision detection and maze solving with live preview"""
    
    def __init__(self):
        super().__init__('vision_solver_node')
        
        self.status_pub = self.create_publisher(String, '/maze/status', 10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/maze/waypoints_pixel', 10)
        self.info_pub = self.create_publisher(String, '/maze/info', 10)
        self.image_pub = self.create_publisher(Image, '/maze/image', 10)
        
        self.command_sub = self.create_subscription(
            String, '/maze/command', self.command_callback, 10
        )
        
        self.running = False
        self.auto_capture_enabled = False  # New flag to control auto-capture
        self.start_color = 'green'
        self.camera = None
        self.analyzer = None
        self.lock = None
        self.bridge = CvBridge()
        self.detection_thread = None  # Thread for camera loop
        
        # Create output directories
        MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.get_logger().info('Vision Solver Node initialized')
        self.publish_status('idle')
    
    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Status: {status}')
    
    def command_callback(self, msg):
        command = msg.data.lower()
        self.get_logger().info(f'Received command: {command}')
        
        if 'green' in command:
            self.start_color = 'green'
        elif 'red' in command:
            self.start_color = 'red'
        
        # Solve command - start camera AND enable auto-capture immediately
        if any(word in command for word in ['start', 'solve', 'begin', 'run']):
            if not self.running:
                # Start camera with auto-capture enabled
                self.start_detection()
            else:
                # Camera already running, just enable capture with NEW color
                self.auto_capture_enabled = True
                self.lock.reset()  # Reset lock for fresh detection
                self.get_logger().info(f'🎯 Auto-capture ENABLED - will capture stable maze from {self.start_color.upper()}')
        
        elif 'stop' in command or 'abort' in command:
            self.stop_detection()
        
        elif 'reset' in command:
            self.reset()
    
    def start_detection(self):
        """Start camera with auto-capture enabled"""
        if self.running:
            self.get_logger().info('Camera already running')
            return
            
        self.get_logger().info(f'Starting detection with start_color={self.start_color}')
        
        try:
            # Close any previous result windows
            cv2.destroyWindow("1. Waypoints (Numbered)")
            cv2.destroyWindow("2. Solution Path")
            cv2.destroyWindow("3. Maze Analysis")
        except:
            pass
        
        self.running = True
        self.auto_capture_enabled = True  # Enable capture immediately
        
        try:
            self.camera = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            
            if not self.camera.isOpened():
                self.publish_status('error')
                self.get_logger().error('Could not open camera')
                return
        except Exception as e:
            self.publish_status('error')
            self.get_logger().error(f'Camera error: {e}')
            return
        
        self.analyzer = LiveMazeAnalyzer()
        self.lock = StabilityLock(hold_sec=MARKER_HOLD_SEC, px_tol=PIXEL_STABILITY_TOL)
        
        self.publish_status('detecting')
        
        # Start detection loop in separate thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        self.get_logger().info('Detection thread started')
    
    def detection_loop(self):
        """Main detection loop with LIVE PREVIEW and countdown"""
        while self.running and rclpy.ok():
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            src, score = detect_maze_quad(gray)
            
            # Create display frame
            view = frame.copy()
            
            if src is not None:
                # Draw quad on preview
                if DRAW_GUIDES:
                    for i in range(4):
                        p1 = tuple(src[i].astype(int))
                        p2 = tuple(src[(i+1)%4].astype(int))
                        cv2.line(view, p1, p2, (0, 255, 0), 3)
                
                # ONLY process if auto-capture is enabled
                if self.auto_capture_enabled and score >= QUAD_SCORE_THRESHOLD:
                    held = self.lock.update(src)
                    remaining = max(0.0, MARKER_HOLD_SEC - held)
                    
                    if remaining <= 0.0:
                        # AUTO SNAPSHOT!
                        self.get_logger().info(f'Maze stable! Score: {score:.2f}')
                        self.publish_status('detected')
                        
                        try:
                            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                            self.image_pub.publish(img_msg)
                        except Exception as e:
                            self.get_logger().warn(f'Could not publish image: {e}')
                        
                        # Process maze
                        self.process_maze(frame, src)
                        
                        # DISABLE auto-capture after solving - camera stays on but no capture
                        self.auto_capture_enabled = False
                        self.lock.reset()
                        self.get_logger().info('✓ Maze solved! Camera still running - waiting for next command')
                        self.publish_status('idle')
                        continue
                    else:
                        # Show countdown
                        cv2.putText(
                            view,
                            f"Quad score {score:.2f} - Auto-snapshot in {remaining:.1f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                        )
                        cv2.putText(
                            view,
                            f"Start color: {self.start_color.upper()}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                        )
                else:
                    # Either auto-capture disabled OR score too low
                    self.lock.reset()
                    if not self.auto_capture_enabled:
                        cv2.putText(
                            view,
                            f"Quad detected (score: {score:.2f}) - Waiting for solve command",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA
                        )
                    else:
                        cv2.putText(
                            view,
                            f"Quad score {score:.2f} (< {QUAD_SCORE_THRESHOLD}) - hold steady...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2, cv2.LINE_AA
                        )
            else:
                self.lock.reset()
                cv2.putText(view, "Show the maze board; seeking a rectangular quad.",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
            
            # Show status at bottom
            if self.auto_capture_enabled:
                cv2.putText(view, "ROS Maze Solver (auto-capture enabled)", 
                           (10, view.shape[0]-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(view, f"Command: solve from {self.start_color.upper()}", 
                           (10, view.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(view, "Waiting for next solve command from Claude...", 
                           (10, view.shape[0]-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.putText(view, "Camera in preview mode", 
                           (10, view.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Display live preview
            cv2.imshow("ROS Maze Solver - Live Camera", view)
            cv2.waitKey(FRAME_DELAY_MS)
            
            # Small delay for ROS
            time.sleep(0.01)
    
    def process_maze(self, frame, src_quad):
        self.publish_status('solving')
        self.get_logger().info('Solving maze...')
        
        warped, H_total = warp_square(frame, src_quad)
        result = self.analyzer.analyze_snapshot(warped, H_total, frame, start_color=self.start_color)
        
        waypoints_original = result.get('waypoints_original', [])
        
        if not waypoints_original:
            self.get_logger().error('No waypoints generated')
            self.publish_status('error')
            return
        
        paths = self.save_all_debug_outputs(result, preview_note=f"start_color={self.start_color}")
        self.publish_waypoints(waypoints_original)
        
        grid_size = result.get('grid_size', (0, 0))
        path_cells = result.get('path_cells', [])
        info_msg = f"start_color:{self.start_color},waypoints:{len(waypoints_original)},path_length:{len(path_cells)}"
        
        info = String()
        info.data = info_msg
        self.info_pub.publish(info)
        
        # Display results
        self.display_results(result)
        
        self.publish_status('solved')
        self.get_logger().info(f'Solved! {len(waypoints_original)} waypoints')
    
    def publish_waypoints(self, waypoints):
        msg = Float32MultiArray()
        flattened = []
        for x, y in waypoints:
            flattened.append(float(x))
            flattened.append(float(y))
        msg.data = flattened
        self.waypoints_pub.publish(msg)
        self.get_logger().info(f'Published {len(waypoints)} pixel waypoints')
    
    def save_all_debug_outputs(self, result, preview_note=""):
        """Save 3 main images (overwrite) + all 9 images to archive (timestamped)"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # === MAIN FOLDER - 3 images (no timestamps, overwritten each time) ===
        main_paths = {
            "waypoints": MAIN_OUTPUT_DIR / "waypoints.png",
            "solution": MAIN_OUTPUT_DIR / "solution.png",
            "analysis": MAIN_OUTPUT_DIR / "analysis.png",
        }
        
        # === ARCHIVE FOLDER - 9 images (with timestamps) ===
        archive_paths = {
            "analyzed": ARCHIVE_DIR / f"maze_analyzed_{ts}.png",
            "binary":   ARCHIVE_DIR / f"maze_binary_{ts}.png",
            "warped":   ARCHIVE_DIR / f"maze_warped_{ts}.png",
            "grid":     ARCHIVE_DIR / f"debug_detected_grid_{ts}.png",
            "edges":    ARCHIVE_DIR / f"debug_edges_{ts}.png",
            "solved":   ARCHIVE_DIR / f"maze_solved_{ts}.png",
            "solved_cam": ARCHIVE_DIR / f"maze_solved_on_camera_{ts}.png",
            "waypoints_csv": ARCHIVE_DIR / f"waypoints_original_{ts}.csv",
            "waypoints_img": ARCHIVE_DIR / f"waypoints_on_camera_{ts}.png",
        }
        
        # Prepare the 3 display images
        camera_frame = result.get('camera_frame', None)
        waypoints_original = result.get('waypoints_original', [])
        
        # Image 1: Waypoints (numbered)
        waypoints_img = camera_frame.copy() if camera_frame is not None else np.zeros((600, 600, 3), dtype=np.uint8)
        if waypoints_original:
            for i, (x, y) in enumerate(waypoints_original):
                cv2.circle(waypoints_img, (int(x), int(y)), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.putText(waypoints_img, str(i), (int(x) + 8, int(y) - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            start_pt = waypoints_original[0]
            end_pt = waypoints_original[-1]
            cv2.circle(waypoints_img, (int(start_pt[0]), int(start_pt[1])), 12, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(waypoints_img, (int(end_pt[0]), int(end_pt[1])), 12, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(waypoints_img, f"Waypoints: {len(waypoints_original)} points", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Image 2: Solution path
        solution_img = result.get('solved_on_camera', waypoints_img.copy())
        cv2.putText(solution_img, f"Started from {self.start_color.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Image 3: Analysis (warped with START/END, no grid text)
        analysis_img = result.get('original_warped', np.zeros((600, 600, 3), dtype=np.uint8)).copy()
        entrance = result.get('entrance', None)
        exit_point = result.get('exit', None)
        if entrance:
            cv2.circle(analysis_img, entrance, 15, (0, 255, 0), 3)
            cv2.putText(analysis_img, "START", (entrance[0]-30, entrance[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if exit_point:
            cv2.circle(analysis_img, exit_point, 15, (0, 0, 255), 3)
            cv2.putText(analysis_img, "END", (exit_point[0]-20, exit_point[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # === SAVE MAIN IMAGES (overwrite) ===
        main_ok = []
        main_ok.append(("waypoints.png", cv2.imwrite(str(main_paths["waypoints"]), waypoints_img)))
        main_ok.append(("solution.png", cv2.imwrite(str(main_paths["solution"]), solution_img)))
        main_ok.append(("analysis.png", cv2.imwrite(str(main_paths["analysis"]), analysis_img)))
        
        # === SAVE ARCHIVE IMAGES (timestamped) ===
        archive_ok = []
        archive_ok.append(("analyzed", cv2.imwrite(str(archive_paths["analyzed"]), result.get('result_image', np.zeros((100,100,3), dtype=np.uint8)))))
        archive_ok.append(("binary",   cv2.imwrite(str(archive_paths["binary"]),   result.get('binary_image', np.zeros((100,100), dtype=np.uint8)))))
        archive_ok.append(("warped",   cv2.imwrite(str(archive_paths["warped"]),   result.get('original_warped', np.zeros((100,100,3), dtype=np.uint8)))))
        archive_ok.append(("grid",     cv2.imwrite(str(archive_paths["grid"]),     result.get('grid_debug', np.zeros((100,100,3), dtype=np.uint8)))))
        archive_ok.append(("edges",    cv2.imwrite(str(archive_paths["edges"]),    result.get('edges', np.zeros((100,100), dtype=np.uint8)))))
        archive_ok.append(("solved",   cv2.imwrite(str(archive_paths["solved"]),   result.get('solved_overlay', np.zeros((100,100,3), dtype=np.uint8)))))
        archive_ok.append(("solved_cam", cv2.imwrite(str(archive_paths["solved_cam"]), result.get('solved_on_camera', np.zeros((100,100,3), dtype=np.uint8)))))
        archive_ok.append(("waypoints.png", cv2.imwrite(str(archive_paths["waypoints_img"]), waypoints_img)))
        
        # Save CSV to archive
        csv_ok = False
        try:
            with open(archive_paths["waypoints_csv"], "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                for (x, y) in waypoints_original:
                    writer.writerow([int(x), int(y)])
            csv_ok = True
        except Exception:
            csv_ok = False
        
        # Save CSV to main output folder for converter_node
        main_csv_ok = False
        try:
            CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            main_csv_path = CSV_OUTPUT_DIR / CSV_OUTPUT_FILENAME
            with open(main_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                for (x, y) in waypoints_original:
                    writer.writerow([int(x), int(y)])
            main_csv_ok = True
            self.get_logger().info(f"✓ Saved main CSV for converter: {main_csv_path}")
        except Exception as e:
            main_csv_ok = False
            self.get_logger().error(f"Failed to save main CSV: {e}")
        
        # Log results
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info(f"✓ SNAPSHOT ANALYZED & SAVED @ {ts}")
        self.get_logger().info(f"  Started from: {self.start_color.upper()} dot")
        grid_size = result.get('grid_size', (0, 0))
        self.get_logger().info(f"  Grid: {grid_size[0]}x{grid_size[1]}")
        entry = result.get('entry_original', None)
        exit_pt = result.get('exit_original', None)
        if entry:
            self.get_logger().info(f"  Entrance (orig px): {entry}")
        if exit_pt:
            self.get_logger().info(f"  Exit (orig px): {exit_pt}")
        if result.get('path_cells'):
            self.get_logger().info(f"  Path length (cells): {len(result['path_cells'])}")
        if result.get('verify'):
            v = result['verify']
            self.get_logger().info(f"  Verify: ok={v.get('ok')}  collisions≈{v.get('coll_ratio', 0):.3f}")
        if preview_note:
            self.get_logger().info(f"  Note: {preview_note}")
        
        self.get_logger().info("-"*60)
        self.get_logger().info("MAIN FOLDER (3 latest images):")
        self.get_logger().info(f"  {MAIN_OUTPUT_DIR.resolve()}")
        for key, success in main_ok:
            self.get_logger().info(f"  • {key:15s}: {'OK' if success else 'FAILED'}")
        self.get_logger().info(f"  • {CSV_OUTPUT_FILENAME:15s}: {'OK' if main_csv_ok else 'FAILED'} -> {CSV_OUTPUT_DIR.resolve()}")
        
        self.get_logger().info("-"*60)
        self.get_logger().info("ARCHIVE FOLDER (all timestamped):")
        self.get_logger().info(f"  {ARCHIVE_DIR.resolve()}")
        for key, success in archive_ok:
            self.get_logger().info(f"  • {key:11s}_{ts}.png: {'OK' if success else 'FAILED'}")
        self.get_logger().info(f"  • waypoints_{ts}.csv: {'OK' if csv_ok else 'FAILED'}")
        self.get_logger().info("="*60 + "\n")
        
        return {"main": main_paths, "archive": archive_paths}
    
    def display_results(self, result):
        """Display 3 result windows that stay open until next command"""
        waypoints_original = result.get('waypoints_original', [])
        camera_frame = result.get('camera_frame', None)
        
        if camera_frame is None or not waypoints_original:
            self.get_logger().warn("Could not display results - missing data")
            return
        
        # WINDOW 1: Waypoints on camera (numbered)
        waypoints_img = camera_frame.copy()
        for i, (x, y) in enumerate(waypoints_original):
            cv2.circle(waypoints_img, (int(x), int(y)), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(waypoints_img, str(i), (int(x) + 8, int(y) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Add start/end markers
        if waypoints_original:
            start_pt = waypoints_original[0]
            end_pt = waypoints_original[-1]
            cv2.circle(waypoints_img, (int(start_pt[0]), int(start_pt[1])), 12, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(waypoints_img, (int(end_pt[0]), int(end_pt[1])), 12, (0, 0, 255), 3, cv2.LINE_AA)
        
        cv2.putText(waypoints_img, f"Waypoints: {len(waypoints_original)} points", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # WINDOW 2: Maze solved on camera (path overlay)
        solved_img = result.get('solved_on_camera', camera_frame.copy())
        cv2.putText(solved_img, f"Started from {self.start_color.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # WINDOW 3: Maze analyzed (warped with annotations - NO GRID TEXT)
        analyzed_img_raw = result.get('result_image', None)
        if analyzed_img_raw is None:
            analyzed_img_raw = result.get('original_warped', np.zeros((600, 600, 3), dtype=np.uint8))
        
        # Create clean copy without grid text
        analyzed_img = analyzed_img_raw.copy()
        # The grid text is already burned into result_image, so we use original_warped instead
        analyzed_img = result.get('original_warped', np.zeros((600, 600, 3), dtype=np.uint8)).copy()
        
        # Re-draw start/end markers without grid text
        entrance = result.get('entrance', None)
        exit_point = result.get('exit', None)
        if entrance:
            cv2.circle(analyzed_img, entrance, 15, (0, 255, 0), 3)
            cv2.putText(analyzed_img, "START", (entrance[0]-30, entrance[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if exit_point:
            cv2.circle(analyzed_img, exit_point, 15, (0, 0, 255), 3)
            cv2.putText(analyzed_img, "END", (exit_point[0]-20, exit_point[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display all 3 windows
        cv2.imshow("1. Waypoints (Numbered)", waypoints_img)
        cv2.imshow("2. Solution Path", solved_img)
        cv2.imshow("3. Maze Analysis", analyzed_img)
        
        # Keep windows open with a long waitKey
        cv2.waitKey(1)
        
        self.get_logger().info("✓ 3 result windows displayed. They will stay open until next command.")
        self.get_logger().info("  Press any key in a window or give new command to continue...")
    
    def stop_detection(self):
        """Stop camera and close windows"""
        self.running = False
        self.auto_capture_enabled = False
        if self.camera is not None:
            self.camera.release()
        # Close camera window
        cv2.destroyWindow("ROS Maze Solver - Live Camera")
        self.publish_status('idle')
        self.get_logger().info('Camera stopped')
    
    def reset(self):
        self.stop_detection()
        if self.lock:
            self.lock.reset()
        self.get_logger().info('Node reset')


def main(args=None):
    rclpy.init(args=args)
    node = VisionSolverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_detection()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()