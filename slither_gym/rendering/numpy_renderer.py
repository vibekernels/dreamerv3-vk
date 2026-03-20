from __future__ import annotations

import numpy as np

from ..engine.config import GameConfig
from ..engine.game import GameState


# Slither.io-style dark blue color palette
BG_COLOR = np.array([20, 28, 45], dtype=np.uint8)
HEX_LINE_COLOR = np.array([35, 50, 75], dtype=np.uint8)
HEX_FILL_COLOR = np.array([22, 32, 50], dtype=np.uint8)
PLAYER_HEAD_COLOR = np.array([80, 255, 80], dtype=np.uint8)
PLAYER_BODY_COLOR = np.array([50, 220, 50], dtype=np.uint8)
PLAYER_GLOW_COLOR = np.array([30, 100, 30], dtype=np.uint8)
NPC_COLORS = [
    (np.array([255, 100, 100], dtype=np.uint8), np.array([200, 60, 60], dtype=np.uint8)),
    (np.array([100, 200, 255], dtype=np.uint8), np.array([60, 140, 200], dtype=np.uint8)),
    (np.array([255, 255, 100], dtype=np.uint8), np.array([200, 200, 60], dtype=np.uint8)),
    (np.array([255, 100, 255], dtype=np.uint8), np.array([200, 60, 200], dtype=np.uint8)),
    (np.array([100, 255, 255], dtype=np.uint8), np.array([60, 200, 200], dtype=np.uint8)),
    (np.array([255, 180, 100], dtype=np.uint8), np.array([200, 130, 60], dtype=np.uint8)),
    (np.array([180, 100, 255], dtype=np.uint8), np.array([130, 60, 200], dtype=np.uint8)),
    (np.array([255, 150, 150], dtype=np.uint8), np.array([200, 100, 100], dtype=np.uint8)),
]
BOUNDARY_COLOR = np.array([180, 50, 50], dtype=np.uint8)


class NumpyRenderer:
    """Renders ego-centric observations as NumPy RGB arrays in slither.io style.

    All objects exist in world space with world-unit sizes. The renderer
    converts positions and radii to pixel space through a single viewport
    transform, so zooming in/out uniformly scales everything.
    Anti-aliased circles are used for smooth edges.
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.size = config.obs_size
        self.half = config.obs_size / 2.0
        self.base_vr = config.viewport_radius

        # Pre-compute normalised pixel coordinate grid [-1, 1]
        y_coords, x_coords = np.mgrid[0:self.size, 0:self.size]
        self.px_norm = (x_coords.astype(np.float32) - self.half + 0.5) / self.half
        self.py_norm = (y_coords.astype(np.float32) - self.half + 0.5) / self.half

    # ── Coordinate helpers ──────────────────────────────────────────────

    def _viewport_radius(self, snake) -> float:
        """Viewport scales with log of snake length for camera zoom-out."""
        ratio = snake.length / self.config.initial_length
        scale = 1.0 + 0.4 * np.log(max(1.0, ratio))
        return self.base_vr * scale

    def _world_to_px_radius(self, world_radius: float, vr: float) -> float:
        """Convert a world-space radius to a floating-point pixel radius."""
        return world_radius / vr * self.half

    # ── Anti-aliased circle drawing ─────────────────────────────────────

    def _draw_circle(self, img: np.ndarray, x: float, y: float,
                     r_px: float, color: np.ndarray):
        """Draw an anti-aliased filled circle via alpha blending on edges."""
        ri = int(np.ceil(r_px)) + 1
        x0, x1 = max(0, int(x) - ri), min(self.size, int(x) + ri + 1)
        y0, y1 = max(0, int(y) - ri), min(self.size, int(y) + ri + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        # Alpha: 1 inside, smooth falloff at edge (1px transition)
        alpha = np.clip(r_px - dist + 0.5, 0.0, 1.0).astype(np.float32)
        mask = alpha > 0
        if not np.any(mask):
            return
        region = img[y0:y1, x0:x1]
        alpha_3 = alpha[:, :, np.newaxis]
        blended = (region.astype(np.float32) * (1 - alpha_3)
                   + color.astype(np.float32) * alpha_3)
        region[mask] = np.clip(blended, 0, 255).astype(np.uint8)[mask]

    def _draw_circle_additive(self, img: np.ndarray, x: float, y: float,
                              r_px: float, color: np.ndarray):
        """Draw an anti-aliased circle with additive blending (for glows)."""
        ri = int(np.ceil(r_px)) + 1
        x0, x1 = max(0, int(x) - ri), min(self.size, int(x) + ri + 1)
        y0, y1 = max(0, int(y) - ri), min(self.size, int(y) + ri + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        alpha = np.clip(r_px - dist + 0.5, 0.0, 1.0).astype(np.float32)
        mask = alpha > 0
        if not np.any(mask):
            return
        region = img[y0:y1, x0:x1]
        add = color.astype(np.float32) * alpha[:, :, np.newaxis]
        blended = np.clip(region.astype(np.float32) + add, 0, 255).astype(np.uint8)
        region[mask] = blended[mask]

    # ── Main render ─────────────────────────────────────────────────────

    def render(self, state: GameState) -> np.ndarray:
        """Render ego-centric view around the player's head."""
        img = np.full((self.size, self.size, 3), BG_COLOR, dtype=np.uint8)

        player = state.player
        if not player.alive:
            return img

        cx, cy = player.head_pos
        vr = self._viewport_radius(player)

        # World-space pixel grids for this frame
        px = self.px_norm * vr
        py = self.py_norm * vr

        self._draw_hex_grid(img, cx, cy, px, py)
        self._draw_boundary(img, cx, cy, px, py)
        self._draw_food(img, state, cx, cy, vr)

        for i, snake in enumerate(state.snakes[1:]):
            if not snake.alive:
                continue
            head_color, body_color = NPC_COLORS[i % len(NPC_COLORS)]
            self._draw_snake(img, snake, cx, cy, body_color, head_color,
                             snake.boosting, vr)

        self._draw_snake(img, player, cx, cy, PLAYER_BODY_COLOR,
                         PLAYER_HEAD_COLOR, player.boosting, vr)

        return img

    # ── Background ──────────────────────────────────────────────────────

    def _draw_hex_grid(self, img: np.ndarray, cx: float, cy: float,
                       px: np.ndarray, py: np.ndarray):
        """Draw a hexagonal grid pattern like slither.io."""
        hex_size = 30.0

        wx = px + cx
        wy = py + cy

        sqrt3 = np.float32(1.7320508)
        q = (sqrt3 / 3.0 * wx - 1.0 / 3.0 * wy) / hex_size
        r = (2.0 / 3.0 * wy) / hex_size

        s = -q - r
        qi = np.round(q).astype(np.int32)
        ri = np.round(r).astype(np.int32)
        si = np.round(s).astype(np.int32)

        q_diff = np.abs(qi.astype(np.float32) - q)
        r_diff = np.abs(ri.astype(np.float32) - r)
        s_diff = np.abs(si.astype(np.float32) - s)

        fix_q = (q_diff > r_diff) & (q_diff > s_diff)
        fix_r = (~fix_q) & (r_diff > s_diff)
        qi[fix_q] = (-ri - si)[fix_q]
        ri[fix_r] = (-qi - si)[fix_r]

        hex_cx = hex_size * (sqrt3 * qi.astype(np.float32) + sqrt3 / 2.0 * ri.astype(np.float32))
        hex_cy = hex_size * (3.0 / 2.0 * ri.astype(np.float32))

        dx = wx - hex_cx
        dy = wy - hex_cy
        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)
        dist_to_edge = np.maximum(
            abs_dy * (2.0 / 3.0),
            abs_dy * (1.0 / 3.0) + abs_dx * (sqrt3 / 3.0),
        )
        edge_ratio = dist_to_edge / hex_size
        edge_mask = edge_ratio > 0.85

        img[~edge_mask] = HEX_FILL_COLOR
        img[edge_mask] = HEX_LINE_COLOR

    def _draw_boundary(self, img: np.ndarray, cx: float, cy: float,
                       px: np.ndarray, py: np.ndarray):
        """Draw arena boundary circle (constant world-width line)."""
        wx = px + cx
        wy = py + cy
        dist = np.sqrt(wx * wx + wy * wy)
        r = self.config.arena_radius
        boundary_mask = np.abs(dist - r) < 5.0
        img[boundary_mask] = BOUNDARY_COLOR
        outside_mask = dist > r
        img[outside_mask] = img[outside_mask] // 3

    # ── Food ────────────────────────────────────────────────────────────

    def _draw_food(self, img: np.ndarray, state: GameState,
                   cx: float, cy: float, vr: float):
        """Draw glowing food pellets. All sizes in world units."""
        food = state.food
        active = food.active
        if not np.any(active):
            return

        positions = food.positions[active]
        colors = food.colors[active]

        rel_x = positions[:, 0] - cx
        rel_y = positions[:, 1] - cy

        margin = vr + self.config.food_radius
        in_view = (np.abs(rel_x) < margin) & (np.abs(rel_y) < margin)
        if not np.any(in_view):
            return

        rel_x = rel_x[in_view]
        rel_y = rel_y[in_view]
        colors = colors[in_view]

        # Pixel positions (float for AA)
        fpx = rel_x / vr * self.half + self.half
        fpy = rel_y / vr * self.half + self.half

        # World-space radii → pixel radii
        fr = self._world_to_px_radius(self.config.food_radius, vr)
        gr = self._world_to_px_radius(self.config.food_radius * 1.8, vr)

        for k in range(len(fpx)):
            x, y = fpx[k], fpy[k]
            # Glow
            glow_color = colors[k].astype(np.int16) // 3
            self._draw_circle_additive(img, x, y, gr, glow_color)
            # Core
            self._draw_circle(img, x, y, fr, colors[k])

    # ── Snakes ──────────────────────────────────────────────────────────

    def _draw_snake(
        self,
        img: np.ndarray,
        snake,
        cx: float,
        cy: float,
        body_color: np.ndarray,
        head_color: np.ndarray,
        boosting: bool,
        vr: float,
    ):
        """Draw a snake. All sizes derived from the snake's world-space radius."""
        segments = snake.active_segments()

        rel_x = segments[:, 0] - cx
        rel_y = segments[:, 1] - cy

        # World-space radii (scale with this snake's own length)
        world_br = snake.get_radius(self.config.body_radius, self.config.initial_length)
        world_hr = snake.get_radius(self.config.head_radius, self.config.initial_length)

        # Convert to float pixel radii
        br = self._world_to_px_radius(world_br, vr)
        hr = self._world_to_px_radius(world_hr, vr)

        margin = vr + world_br * 2
        in_view = (np.abs(rel_x) < margin) & (np.abs(rel_y) < margin)

        # Float pixel positions for AA
        fpx = rel_x / vr * self.half + self.half
        fpy = rel_y / vr * self.half + self.half

        # Boost glow: brighten colours
        if boosting:
            body_color = np.clip(body_color.astype(np.int16) + 50, 0, 255).astype(np.uint8)
            head_color = np.clip(head_color.astype(np.int16) + 50, 0, 255).astype(np.uint8)

        dark_body = np.clip(body_color.astype(np.int16) * 2 // 3, 0, 255).astype(np.uint8)

        # Boost aura
        if boosting:
            glow_r = self._world_to_px_radius(world_br * 1.6, vr)
            glow_color = body_color.astype(np.int16) // 4
            for k in range(len(segments) - 1, -1, -1):
                if not in_view[k]:
                    continue
                self._draw_circle_additive(img, fpx[k], fpy[k], glow_r, glow_color)

        # Body & head (tail-first so head draws on top)
        for k in range(len(segments) - 1, -1, -1):
            if not in_view[k]:
                continue
            is_head = k == 0
            r = hr if is_head else br

            if is_head:
                self._draw_circle(img, fpx[k], fpy[k], r, head_color)
                # Highlight
                highlight_r = max(0.5, r / 3.0)
                bright = np.clip(head_color.astype(np.int16) + 60, 0, 255).astype(np.uint8)
                self._draw_circle(img, fpx[k], fpy[k], highlight_r, bright)
            else:
                color = body_color if k % 2 == 0 else dark_body
                self._draw_circle(img, fpx[k], fpy[k], r, color)
