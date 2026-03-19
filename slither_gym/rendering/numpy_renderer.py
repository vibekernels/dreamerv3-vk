from __future__ import annotations

import numpy as np

from ..engine.config import GameConfig
from ..engine.game import GameState


# Pre-defined colors (BGR-ish, but we use RGB)
BG_COLOR = np.array([15, 20, 15], dtype=np.uint8)
GRID_COLOR = np.array([25, 35, 25], dtype=np.uint8)
PLAYER_HEAD_COLOR = np.array([80, 200, 255], dtype=np.uint8)
PLAYER_BODY_COLOR = np.array([50, 150, 220], dtype=np.uint8)
NPC_COLORS = [
    np.array([255, 100, 100], dtype=np.uint8),
    np.array([100, 255, 100], dtype=np.uint8),
    np.array([255, 255, 100], dtype=np.uint8),
    np.array([255, 100, 255], dtype=np.uint8),
    np.array([100, 255, 255], dtype=np.uint8),
    np.array([255, 180, 100], dtype=np.uint8),
    np.array([180, 100, 255], dtype=np.uint8),
    np.array([100, 180, 255], dtype=np.uint8),
]
BOUNDARY_COLOR = np.array([180, 50, 50], dtype=np.uint8)


class NumpyRenderer:
    """Renders ego-centric observations as NumPy RGB arrays."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.size = config.obs_size
        self.vr = config.viewport_radius

        # Pre-compute pixel coordinate grid (centered on 0,0)
        half = self.size / 2
        y_coords, x_coords = np.mgrid[0:self.size, 0:self.size]
        # Map pixel coords to world-relative coords
        self.px = (x_coords.astype(np.float32) - half + 0.5) / half * self.vr
        self.py = (y_coords.astype(np.float32) - half + 0.5) / half * self.vr

    def render(self, state: GameState) -> np.ndarray:
        """Render ego-centric view around the player's head."""
        img = np.full((self.size, self.size, 3), BG_COLOR, dtype=np.uint8)

        player = state.player
        if not player.alive:
            return img

        cx, cy = player.head_pos

        # Draw grid lines (subtle)
        self._draw_grid(img, cx, cy)

        # Draw arena boundary
        self._draw_boundary(img, cx, cy)

        # Draw food
        self._draw_food(img, state, cx, cy)

        # Draw NPC snakes
        for i, snake in enumerate(state.snakes[1:]):
            if not snake.alive:
                continue
            color = NPC_COLORS[i % len(NPC_COLORS)]
            self._draw_snake(img, snake, cx, cy, color, color)

        # Draw player snake (on top)
        self._draw_snake(img, player, cx, cy, PLAYER_BODY_COLOR, PLAYER_HEAD_COLOR)

        return img

    def _draw_grid(self, img: np.ndarray, cx: float, cy: float):
        """Draw subtle grid lines."""
        grid_spacing = 40.0
        # World-space coordinates of each pixel
        wx = self.px + cx
        wy = self.py + cy
        # Grid lines where world coord is near a multiple of spacing
        gx = np.abs(wx % grid_spacing)
        gy = np.abs(wy % grid_spacing)
        grid_mask = (np.minimum(gx, grid_spacing - gx) < 1.0) | \
                    (np.minimum(gy, grid_spacing - gy) < 1.0)
        img[grid_mask] = GRID_COLOR

    def _draw_boundary(self, img: np.ndarray, cx: float, cy: float):
        """Draw arena boundary circle."""
        wx = self.px + cx
        wy = self.py + cy
        dist = np.sqrt(wx * wx + wy * wy)
        r = self.config.arena_radius
        boundary_mask = np.abs(dist - r) < 3.0
        img[boundary_mask] = BOUNDARY_COLOR
        # Shade outside arena
        outside_mask = dist > r
        img[outside_mask] = img[outside_mask] // 3

    def _draw_food(self, img: np.ndarray, state: GameState, cx: float, cy: float):
        """Draw food pellets."""
        food = state.food
        active = food.active
        if not np.any(active):
            return

        positions = food.positions[active]
        colors = food.colors[active]

        # Convert food positions to pixel space
        rel_x = positions[:, 0] - cx
        rel_y = positions[:, 1] - cy

        # Cull food outside viewport (with margin)
        margin = self.vr + self.config.food_radius
        in_view = (np.abs(rel_x) < margin) & (np.abs(rel_y) < margin)
        if not np.any(in_view):
            return

        rel_x = rel_x[in_view]
        rel_y = rel_y[in_view]
        colors = colors[in_view]

        # Convert to pixel coordinates
        half = self.size / 2
        px = (rel_x / self.vr * half + half).astype(np.int32)
        py = (rel_y / self.vr * half + half).astype(np.int32)

        # Food radius in pixels
        fr = max(1, int(self.config.food_radius / self.vr * half))

        for k in range(len(px)):
            x, y = px[k], py[k]
            x0, x1 = max(0, x - fr), min(self.size, x + fr + 1)
            y0, y1 = max(0, y - fr), min(self.size, y + fr + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            img[y0:y1, x0:x1] = colors[k]

    def _draw_snake(
        self,
        img: np.ndarray,
        snake,
        cx: float,
        cy: float,
        body_color: np.ndarray,
        head_color: np.ndarray,
    ):
        """Draw a snake's segments."""
        segments = snake.active_segments()
        half = self.size / 2

        # Convert to pixel space
        rel_x = segments[:, 0] - cx
        rel_y = segments[:, 1] - cy

        # Cull segments outside viewport
        margin = self.vr + self.config.body_radius * 2
        in_view = (np.abs(rel_x) < margin) & (np.abs(rel_y) < margin)

        px_all = (rel_x / self.vr * half + half).astype(np.int32)
        py_all = (rel_y / self.vr * half + half).astype(np.int32)

        # Body radius in pixels
        br = max(1, int(self.config.body_radius / self.vr * half))
        hr = max(2, int(self.config.head_radius / self.vr * half))

        # Draw body segments (tail-first so head draws on top)
        for k in range(len(segments) - 1, -1, -1):
            if not in_view[k]:
                continue
            x, y = px_all[k], py_all[k]
            r = hr if k == 0 else br
            color = head_color if k == 0 else body_color
            x0, x1 = max(0, x - r), min(self.size, x + r + 1)
            y0, y1 = max(0, y - r), min(self.size, y + r + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            # Circle mask within the bounding box
            yy, xx = np.ogrid[y0:y1, x0:x1]
            circle = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
            img[y0:y1, x0:x1][circle] = color
