"""
Custom widgets for VersperOmni TUI.

Braille-based line charts, metric panels, status/controls panels.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

from rich.style import Style
from rich.text import Text
from textual.strip import Strip
from textual.widgets import Static
from textual.widget import Widget
from textual.color import Color as TColor


# ── Colour palette (burn-inspired) ─────────────────────────────────────

COLOR_TRAIN = Style(color="#e2b714")   # burn yellow
COLOR_VALID = Style(color="#4fc3f7")   # burn blue
COLOR_TEST  = Style(color="#66bb6a")   # burn green
COLOR_DIM   = Style(color="#6b7280")   # grey
COLOR_TEXT  = Style(color="#a0b0c0")   # light grey
COLOR_ACCENT = Style(color="#e2b714", bold=True)


# ── Braille line chart ─────────────────────────────────────────────────

class BrailleLineChart(Widget):
    """A line chart that renders with Unicode braille characters.

    Each braille character (U+2800..U+28FF) encodes a 2×4 dot grid,
    giving effective pixel resolution of ``width*2 × height*4``.
    """

    BRAILLE_BASE = 0x2800

    # Dot → bit mapping (standard Unicode braille order):
    #   bit 0  = dot 1  (top-left)
    #   bit 3  = dot 2  (top-right)
    #   bit 1  = dot 3  (middle-left)
    #   bit 4  = dot 4  (middle-right)
    #   bit 2  = dot 5  (bottom-left)
    #   bit 5  = dot 6  (bottom-right)
    #   bit 6  = dot 7  (bottom row left)
    #   bit 7  = dot 8  (bottom row right)
    DOT_MAP = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    def __init__(
        self,
        *,
        title: str = "",
        max_points: int = 500,
        height_chars: int = 5,
        show_axes: bool = True,
        color: str = "#e2b714",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._title = title
        self._max_points = max_points
        self._height_chars = height_chars
        self._show_axes = show_axes
        self._color = TColor.parse(color)
        self._series: dict[str, deque[tuple[float, float]]] = {}
        self._series_colors: dict[str, str] = {}

    def add_series(self, name: str, color: str = "#e2b714") -> None:
        """Register a named series."""
        if name not in self._series:
            self._series[name] = deque(maxlen=self._max_points)
            self._series_colors[name] = color

    def add_point(self, series: str, x: float, y: float) -> None:
        """Append a point to *series* and schedule a redraw."""
        if series not in self._series:
            self.add_series(series)
        self._series[series].append((x, y))
        self.refresh()

    def clear(self):
        """Clear all series data."""
        for s in self._series:
            self._series[s].clear()
        self.refresh()

    def render_line(self, style: Style) -> list[Strip]:
        """Render the chart into strips (one per row)."""
        width = self.size.width
        height = self._height_chars

        # Collect all points across series
        all_y: list[float] = []
        for s in self._series.values():
            for _, y in s:
                all_y.append(y)

        if not all_y:
            title_line = Text(f" {self._title} [dim](no data)[/]", style=COLOR_TEXT)
            return [Strip(Text("").render(self.app), []) for _ in range(height)]

        y_min = min(all_y)
        y_max = max(all_y)
        y_range = y_max - y_min
        if y_range < 1e-12:
            y_range = 1.0

        braille_cols = width
        braille_rows = height
        pixels_x = braille_cols * 2
        pixels_y = braille_rows * 4

        # Build a 2D pixel grid (False = empty, (r,g,b) tuple = filled)
        grid: list[list[Optional[tuple[int, int, int]]]] = [
            [None] * pixels_x for _ in range(pixels_y)
        ]

        # Plot each series
        for name, points in self._series.items():
            if not points:
                continue
            color_hex = self._series_colors.get(name, "#e2b714")
            rgb = _hex_to_rgb(color_hex)
            pts = list(points)

            # Normalise points to pixel coords
            pixel_coords: list[tuple[int, int]] = []
            for i, (x, y) in enumerate(pts):
                px = int((i / max(len(pts) - 1, 1)) * (pixels_x - 1))
                py = pixels_y - 1 - int(
                    ((y - y_min) / y_range) * (pixels_y - 1)
                )
                py = max(0, min(pixels_y - 1, py))
                pixel_coords.append((px, py))

            # Rasterise lines between consecutive points
            for i in range(1, len(pixel_coords)):
                _draw_line(grid, pixel_coords[i - 1], pixel_coords[i], rgb)

            # Always plot the points themselves
            for px, py in pixel_coords:
                if 0 <= py < pixels_y and 0 <= px < pixels_x:
                    grid[py][px] = rgb

        # Convert pixel grid → braille strips
        strips: list[Strip] = []
        for row in range(braille_rows):
            cells: list[tuple[str, Style]] = []
            for col in range(braille_cols):
                mask = 0
                for dy in range(4):
                    for dx in range(2):
                        py = row * 4 + dy
                        px = col * 2 + dx
                        if py < pixels_y and px < pixels_x and grid[py][px] is not None:
                            mask |= self.DOT_MAP[dy][dx]
                ch = chr(self.BRAILLE_BASE + mask)
                # Determine colour from the pixel at this cell (prefer top-left)
                pixel_color = grid[row * 4][col * 2]
                if pixel_color is None:
                    pixel_color = grid[row * 4][col * 2 + 1]
                if pixel_color is None and row * 4 + 1 < pixels_y:
                    pixel_color = grid[row * 4 + 1][col * 2]
                if pixel_color is None and row * 4 + 1 < pixels_y:
                    pixel_color = grid[row * 4 + 1][col * 2 + 1]
                fg = f"#{pixel_color[0]:02x}{pixel_color[1]:02x}{pixel_color[2]:02x}" if pixel_color else "#a0b0c0"
                cells.append((ch, Style(color=fg)))
            # Build a Rich Text segment for this row
            text = Text.assemble(*[(ch, style) for ch, style in cells])
            strips.append(Strip(text.render(self.app), []))

        # Prepend title row
        if self._show_axes:
            y_label = f" {y_min:.2f} "
            title_rich = Text(f" {self._title}", style=COLOR_ACCENT)
            title_rich.append(f"  [{y_min:.2f}..{y_max:.2f}]", style=COLOR_DIM)

            for s_name, s_color in self._series_colors.items():
                title_rich.append(f"  ●", style=Style(color=s_color))
                title_rich.append(f"{s_name}", style=COLOR_TEXT)

            title_strip = Strip(title_rich.render(self.app), [])
            # Pad title strip to full width
            strips = [title_strip] + strips

        return strips

    def get_content_width(self, container: int, viewport: int) -> int:
        return container

    def get_content_height(self, container: int, viewport: int, width: int) -> int:
        # title row + braille rows + axis label row
        extra = 2 if self._show_axes else 0
        return self._height_chars + extra


# ── Sparkline (single-row compact chart) ───────────────────────────────

BLOCKS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def sparkline(values: list[float], width: int = 40) -> Text:
    """Render a compact single-row sparkline."""
    if not values:
        return Text("─" * width, style=COLOR_DIM)

    n = len(values)
    vmin = min(values)
    vrange = max(values) - vmin
    if vrange < 1e-12:
        vrange = 1.0

    result = Text()
    for i in range(width):
        idx = int((i / max(width - 1, 1)) * (n - 1))
        v = values[idx]
        level = int(((v - vmin) / vrange) * 7)
        level = max(0, min(7, level))
        result.append(BLOCKS[level], style=COLOR_TRAIN)
    return result


# ── Metric display helpers ─────────────────────────────────────────────

def metric_row(name: str, value: str, color: str = "#a0b0c0") -> Text:
    """A single metric row: ``name: value``."""
    t = Text()
    t.append(f" {name}: ", style=COLOR_ACCENT)
    t.append(value, style=Style(color=color))
    return t


def status_row(label: str, value: str) -> Text:
    """A label: value row."""
    t = Text()
    t.append(f" {label}: ", style=COLOR_ACCENT)
    t.append(value, style=COLOR_TEXT)
    return t


# ── Helper functions ───────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb`` → ``(r, g, b)``."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _draw_line(
    grid: list[list[Optional[tuple[int, int, int]]]],
    a: tuple[int, int],
    b: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Bresenham-style line rasterisation onto *grid*."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    pixels_y = len(grid)
    pixels_x = len(grid[0])

    while True:
        if 0 <= y0 < pixels_y and 0 <= x0 < pixels_x:
            grid[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


# ── ETA formatting ─────────────────────────────────────────────────────

def format_eta(seconds: float) -> str:
    """Human-readable ETA string — burn-inspired format."""
    secs = int(seconds)
    days = secs // 86400
    hours = (secs % 86400) // 3600
    minutes = (secs % 3600) // 60
    secs_rem = secs % 60

    if days > 1:
        return f"{days} days"
    elif days == 1:
        return "1 day"
    elif hours > 1:
        return f"{hours} hours"
    elif hours == 1:
        return "1 hour"
    elif minutes > 1:
        return f"{minutes} mins"
    elif minutes == 1:
        return "1 min"
    elif secs_rem > 1:
        return f"{secs_rem} secs"
    else:
        return "1 sec"


class ETAEstimator:
    """Simple per-step rate estimator with warmup.

    Computes ETA as ``rate⁻¹ × remaining_steps``.
    """

    def __init__(self, warmup_steps: int = 10):
        self._warmup_steps = warmup_steps
        self._start_time: Optional[float] = None
        self._step_at_warmup: int = 0
        self._steps_per_sec: Optional[float] = None
        self._warmed_up: bool = False

    def start(self) -> None:
        import time
        self._start_time = time.time()
        self._step_at_warmup = 0
        self._steps_per_sec = None
        self._warmed_up = False

    def update(self, cumulative_step: int) -> None:
        """Update with cumulative step count (across all epochs)."""
        import time
        now = time.time()
        elapsed = now - self._start_time

        if not self._warmed_up:
            if cumulative_step >= self._warmup_steps:
                self._step_at_warmup = cumulative_step
                self._warmed_up = True
            return

        # Rolling rate: total steps since start
        steps_done = cumulative_step - self._step_at_warmup
        warm_elapsed = now - (self._start_time + (self._step_at_warmup / max(cumulative_step, 1)) * elapsed)
        # Simpler: just use total elapsed since warmup
        if elapsed > 0 and cumulative_step > 0:
            self._steps_per_sec = cumulative_step / elapsed

    def eta(self, remaining_steps: int) -> Optional[float]:
        """Return estimated seconds for *remaining_steps*."""
        if self._steps_per_sec is None or self._steps_per_sec <= 0:
            return None
        return remaining_steps / self._steps_per_sec

    def eta_str_for(self, remaining_steps: int) -> str:
        secs = self.eta(remaining_steps)
        if secs is None:
            return "---"
        return format_eta(secs)
