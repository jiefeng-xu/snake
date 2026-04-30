"""Interactive pendulum snake (wave pendulum) visualizer.

This script creates a simple "pendulum snake" animation using Matplotlib.
Several pendulums with slightly different lengths swing side by side. Because
each pendulum has its own natural period, the collection of pendulums forms
beautiful wave patterns over time – often called a pendulum snake or pendulum
wave machine.

How to use the viewer
---------------------

Run the script with ``python examples/pendulum_snake.py``. A Matplotlib window
opens with animated pendulums and five sliders:

``Pendulums``
    Sets how many individual pendulums form the wave. More pendulums mean a
    longer wave train and richer interference patterns.

``Base length``
    Sets the starting string length (in metres) of the first pendulum. Longer
    strings swing more slowly because the period of a simple pendulum grows
    with the square root of its length.

``Length step``
    Adds extra length to each subsequent pendulum. A larger step makes the
    difference between adjacent pendulum periods more pronounced, changing the
    wave interference pattern.

``Release angle``
    Controls the starting angle (in degrees) away from vertical. Higher values
    give the bobs more energy and wider swings.

``Gravity``
    Adjusts the gravitational acceleration in metres per second squared. This
    allows experimentation with how the wave evolves under different gravity
    strengths (for example on the Moon vs. Earth).

You can pause the animation by pressing the ``space`` key. Drag any slider to
see immediately how the motion changes. Try small length steps for slow,
mesmerising waves and larger steps for quicker, more chaotic-looking motion.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.widgets import Slider


@dataclass
class PendulumParameters:
    """Physical parameters for a group of simple pendulums.

    Attributes
    ----------
    num_pendulums:
        Number of pendulums to display.
    base_length:
        Length in metres of the first pendulum.
    length_step:
        Additional length added to each subsequent pendulum.
    release_angle_deg:
        Initial release angle in degrees. (Small-angle approximation assumed.)
    gravity:
        Gravitational acceleration in metres per second squared.
    """

    num_pendulums: int = 12
    base_length: float = 1.0
    length_step: float = 0.035
    release_angle_deg: float = 12.0
    gravity: float = 9.81

    @property
    def release_angle_rad(self) -> float:
        """Initial release angle in radians."""

        return np.deg2rad(self.release_angle_deg)

    def lengths(self) -> np.ndarray:
        """Return an array of string lengths, clipped to positive values."""

        indices = np.arange(self.num_pendulums, dtype=float)
        lengths = self.base_length + indices * self.length_step
        # Prevent zero or negative lengths, which have no physical meaning.
        return np.clip(lengths, 0.05, None)

    def angular_frequencies(self, lengths: np.ndarray | None = None) -> np.ndarray:
        """Return natural angular frequencies for each pendulum.

        Parameters
        ----------
        lengths:
            Optional pre-computed lengths to reuse.
        """

        if lengths is None:
            lengths = self.lengths()
        # Small-angle simple pendulum approximation omega = sqrt(g / L).
        return np.sqrt(self.gravity / lengths)


class PendulumSnakePlot:
    """Helper that manages artists for the pendulum snake animation."""

    def __init__(self, ax: Axes, params: PendulumParameters) -> None:
        self.ax = ax
        self.params = params
        self.string_artists: List[plt.Line2D] = []
        self.bob_artists: List[plt.Line2D] = []
        self._setup_axes()
        self._create_artists()

    def _setup_axes(self) -> None:
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("#0f1a29")
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Pendulum Snake Explorer", color="white", fontsize=14)
        self._update_limits()

    def _update_limits(self) -> None:
        max_length = float(np.max(self.params.lengths()))
        # A little padding so the bobs never leave the viewport.
        padding = max_length * 0.2
        self.ax.set_xlim(-max_length - padding, max_length + padding)
        self.ax.set_ylim(-max_length - padding, padding)

    def _create_artists(self) -> None:
        for artist in itertools.chain(self.string_artists, self.bob_artists):
            artist.remove()
        self.string_artists = []
        self.bob_artists = []

        colors = plt.cm.plasma(np.linspace(0.1, 0.9, self.params.num_pendulums))
        for color in colors:
            (string_line,) = self.ax.plot([], [], lw=2.5, color=color, alpha=0.85)
            (bob_point,) = self.ax.plot([], [], "o", color=color, ms=8)
            self.string_artists.append(string_line)
            self.bob_artists.append(bob_point)

    def refresh(self) -> None:
        """Recreate artists when the number of pendulums changes."""

        self._update_limits()
        self._create_artists()

    def draw(self, time_s: float) -> Sequence[plt.Artist]:
        lengths = self.params.lengths()
        omegas = self.params.angular_frequencies(lengths)
        angles = self.params.release_angle_rad * np.cos(omegas * time_s)
        x_positions = lengths * np.sin(angles)
        y_positions = -lengths * np.cos(angles)

        for idx, (string, bob) in enumerate(zip(self.string_artists, self.bob_artists)):
            x = x_positions[idx]
            y = y_positions[idx]
            string.set_data([0.0, x], [0.0, y])
            bob.set_data([x], [y])

        return [*self.string_artists, *self.bob_artists]


def _make_sliders(
    fig: plt.Figure,
    params: PendulumParameters,
    on_change: Callable[[], None],
) -> Tuple[Slider, Slider, Slider, Slider, Slider]:
    """Create Matplotlib sliders bound to the pendulum parameters."""

    slider_width = 0.78
    slider_height = 0.035
    slider_left = 0.12

    ax_count = fig.add_axes([slider_left, 0.28, slider_width, slider_height], facecolor="#101820")
    ax_base = fig.add_axes([slider_left, 0.22, slider_width, slider_height], facecolor="#101820")
    ax_step = fig.add_axes([slider_left, 0.16, slider_width, slider_height], facecolor="#101820")
    ax_angle = fig.add_axes([slider_left, 0.10, slider_width, slider_height], facecolor="#101820")
    ax_gravity = fig.add_axes([slider_left, 0.04, slider_width, slider_height], facecolor="#101820")

    count_slider = Slider(
        ax=ax_count,
        label="Pendulums",
        valmin=5,
        valmax=24,
        valinit=params.num_pendulums,
        valstep=1,
    )
    base_slider = Slider(ax=ax_base, label="Base length (m)", valmin=0.2, valmax=1.8, valinit=params.base_length, valstep=0.01)
    step_slider = Slider(ax=ax_step, label="Length step (m)", valmin=0.0, valmax=0.15, valinit=params.length_step, valstep=0.005)
    angle_slider = Slider(ax=ax_angle, label="Release angle (°)", valmin=2.0, valmax=25.0, valinit=params.release_angle_deg, valstep=0.5)
    gravity_slider = Slider(ax=ax_gravity, label="Gravity (m/s²)", valmin=1.5, valmax=15.0, valinit=params.gravity, valstep=0.1)

    def _update_from_sliders(_value: float) -> None:
        params.num_pendulums = int(count_slider.val)
        params.base_length = base_slider.val
        params.length_step = step_slider.val
        params.release_angle_deg = angle_slider.val
        params.gravity = gravity_slider.val
        on_change()

    for slider in (count_slider, base_slider, step_slider, angle_slider, gravity_slider):
        slider.on_changed(_update_from_sliders)

    return count_slider, base_slider, step_slider, angle_slider, gravity_slider


def _add_instructions(fig: plt.Figure) -> None:
    instruction_text = (
        "Play with the sliders to see how pendulum length, gravity, and release angle\n"
        "affect the travelling wave pattern. Longer strings swing slower, so a small\n"
        "length step (for example 0.03 m) keeps neighbouring pendulums nearly in phase.\n"
        "A larger step exaggerates the timing differences and produces faster-moving\n"
        "waves. Try lowering gravity to imagine the wave on the Moon!"
    )
    fig.text(0.12, 0.94, instruction_text, fontsize=9, color="#e0e6f8", va="top")


def main() -> None:
    print(__doc__)

    params = PendulumParameters()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.36)

    pendulum_plot = PendulumSnakePlot(ax, params)
    _add_instructions(fig)

    def on_slider_change() -> None:
        pendulum_plot.refresh()
        fig.canvas.draw_idle()

    _make_sliders(fig, params, on_slider_change)

    def frame_generator() -> Iterable[float]:
        time_s = 0.0
        timestep = 1 / 60  # seconds per frame
        while True:
            yield time_s
            time_s += timestep

    def animate(time_s: float):
        artists = pendulum_plot.draw(time_s)
        return artists

    FuncAnimation(fig, animate, frames=frame_generator(), interval=1000 / 60, blit=True)

    plt.show()


if __name__ == "__main__":
    main()
