"""Kid-friendly pendulum snake (wave pendulum) visualizer.

A pendulum snake is a row of swinging weights. Each string has a slightly
different length, so each bob keeps a slightly different beat. When all of the
beats are shown together, the row looks like a snake, a rainbow, or a wave.

Quick start
-----------

1. Install the requirements with ``pip install -r requirements.txt``.
2. Launch the viewer with ``python pendulum_snake.py``.
3. Press the preset buttons and move the sliders to ask "what if?" questions.

Demo ideas for kids
-------------------

* Start with ``Rainbow`` and ask: "Which pendulum gets back first?"
* Press ``Moon`` and notice that everything moves in slow motion.
* Move ``String gap`` to zero. The bobs match because all strings are equal.
* Move ``String gap`` up. The snake shape appears because every bob has a
  different swing time.

The math uses the small-angle pendulum approximation: omega = sqrt(g / L).
That means longer strings swing more slowly, and stronger gravity swings faster.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, Slider

BACKGROUND = "#08111f"
PANEL = "#101820"
TEXT = "#eef6ff"
SOFT_TEXT = "#b9d7ff"


@dataclass
class PendulumParameters:
    """Physical parameters for a group of simple pendulums."""

    num_pendulums: int = 12
    base_length: float = 0.75
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
        return np.clip(lengths, 0.05, None)

    def angular_frequencies(self, lengths: np.ndarray | None = None) -> np.ndarray:
        """Return natural angular frequencies for each pendulum."""

        if lengths is None:
            lengths = self.lengths()
        return np.sqrt(self.gravity / lengths)

    def periods(self, lengths: np.ndarray | None = None) -> np.ndarray:
        """Return each pendulum's swing period in seconds."""

        return 2 * np.pi / self.angular_frequencies(lengths)


PRESETS: Dict[str, PendulumParameters] = {
    "Rainbow": PendulumParameters(),
    "Moon": PendulumParameters(gravity=1.62, release_angle_deg=14.0),
    "Same beat": PendulumParameters(length_step=0.0, release_angle_deg=10.0),
}


def frame_generator() -> Iterator[int]:
    """Yield frame numbers forever for the live animation."""

    return itertools.count()


class PendulumSnakePlot:
    """Helper that manages artists for the pendulum snake animation."""

    def __init__(self, ax: Axes, params: PendulumParameters) -> None:
        self.ax = ax
        self.params = params
        self.string_artists: List[Line2D] = []
        self.bob_artists: List[Line2D] = []
        self.rail_artist: Line2D | None = None
        self.center_artist: Line2D | None = None
        self.fast_label = None
        self.slow_label = None
        self.fact_box = None
        self._setup_axes()
        self._create_artists()

    def _setup_axes(self) -> None:
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(BACKGROUND)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(
            "Pendulum Snake Playground", color=TEXT, fontsize=18, weight="bold", pad=12
        )
        self._update_limits()
        self._add_static_helpers()

    def _update_limits(self) -> None:
        max_length = float(np.max(self.params.lengths()))
        padding = max_length * 0.25
        self.ax.set_xlim(-max_length - padding, max_length + padding)
        self.ax.set_ylim(-max_length - padding, padding * 1.4)

    def _add_static_helpers(self) -> None:
        x_min, x_max = self.ax.get_xlim()
        y_top = self.ax.get_ylim()[1] * 0.18
        (self.rail_artist,) = self.ax.plot(
            [x_min * 0.65, x_max * 0.65], [0, 0], color="#6da6ff", lw=5, alpha=0.45
        )
        (self.center_artist,) = self.ax.plot(
            [0, 0], [0, self.ax.get_ylim()[0]], "--", color="#ffffff", lw=1, alpha=0.18
        )
        self.fast_label = self.ax.text(
            -0.96,
            y_top,
            "Short string\n= quicker beat",
            color=SOFT_TEXT,
            fontsize=10,
            ha="left",
            va="top",
            transform=self.ax.transData,
        )
        self.slow_label = self.ax.text(
            0.55,
            y_top,
            "Long string\n= slower beat",
            color=SOFT_TEXT,
            fontsize=10,
            ha="left",
            va="top",
            transform=self.ax.transData,
        )
        self.fact_box = self.ax.text(
            0.5,
            0.03,
            self._fact_text(),
            color=TEXT,
            fontsize=10,
            ha="center",
            va="bottom",
            transform=self.ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#13243a",
                edgecolor="#315a8a",
                alpha=0.88,
            ),
        )

    def _fact_text(self) -> str:
        periods = self.params.periods()
        return (
            "Science clue: shortest beat "
            f"{periods[0]:.2f}s • longest beat {periods[-1]:.2f}s. "
            "Different beats draw the snake!"
        )

    def _refresh_static_helpers(self) -> None:
        self._update_limits()
        if self.rail_artist is not None:
            x_min, x_max = self.ax.get_xlim()
            self.rail_artist.set_data([x_min * 0.65, x_max * 0.65], [0, 0])
        if self.center_artist is not None:
            self.center_artist.set_data([0, 0], [0, self.ax.get_ylim()[0]])
        y_top = self.ax.get_ylim()[1] * 0.18
        if self.fast_label is not None:
            self.fast_label.set_position((self.ax.get_xlim()[0] * 0.82, y_top))
        if self.slow_label is not None:
            self.slow_label.set_position((self.ax.get_xlim()[1] * 0.38, y_top))
        if self.fact_box is not None:
            self.fact_box.set_text(self._fact_text())

    def _create_artists(self) -> None:
        for artist in itertools.chain(self.string_artists, self.bob_artists):
            artist.remove()
        self.string_artists = []
        self.bob_artists = []

        colors = plt.cm.rainbow(np.linspace(0.02, 0.98, self.params.num_pendulums))
        for index, color in enumerate(colors, start=1):
            (string_line,) = self.ax.plot([], [], lw=2.5, color=color, alpha=0.82)
            (bob_point,) = self.ax.plot(
                [],
                [],
                "o",
                color=color,
                ms=11,
                markeredgecolor="white",
                markeredgewidth=1.2,
                label=f"Pendulum {index}",
            )
            string_line.set_animated(True)
            bob_point.set_animated(True)
            self.string_artists.append(string_line)
            self.bob_artists.append(bob_point)

    def refresh(self) -> None:
        """Recreate artists when parameters change."""

        self._refresh_static_helpers()
        self._create_artists()

    def draw(self, time_s: float) -> Sequence[Artist]:
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


class DemoControls:
    """Container for interactive widgets and update helpers."""

    def __init__(
        self, fig: plt.Figure, params: PendulumParameters, on_change: Callable[[], None]
    ) -> None:
        self.fig = fig
        self.params = params
        self.on_change = on_change
        self.sliders = self._make_sliders()
        self.preset_buttons = self._make_preset_buttons()

    def _make_sliders(self) -> Tuple[Slider, Slider, Slider, Slider, Slider]:
        slider_width = 0.78
        slider_height = 0.032
        slider_left = 0.12

        axes = [
            self.fig.add_axes(
                [slider_left, bottom, slider_width, slider_height], facecolor=PANEL
            )
            for bottom in (0.26, 0.21, 0.16, 0.11, 0.06)
        ]
        count_slider = Slider(
            axes[0], "How many?", 5, 24, valinit=self.params.num_pendulums, valstep=1
        )
        base_slider = Slider(
            axes[1],
            "First string",
            0.2,
            1.8,
            valinit=self.params.base_length,
            valstep=0.01,
        )
        step_slider = Slider(
            axes[2],
            "String gap",
            0.0,
            0.15,
            valinit=self.params.length_step,
            valstep=0.005,
        )
        angle_slider = Slider(
            axes[3],
            "Push size",
            2.0,
            25.0,
            valinit=self.params.release_angle_deg,
            valstep=0.5,
        )
        gravity_slider = Slider(
            axes[4], "Gravity", 1.5, 15.0, valinit=self.params.gravity, valstep=0.1
        )

        for slider in (
            count_slider,
            base_slider,
            step_slider,
            angle_slider,
            gravity_slider,
        ):
            slider.label.set_color(TEXT)
            slider.valtext.set_color(TEXT)
            slider.on_changed(self._update_from_sliders)

        return count_slider, base_slider, step_slider, angle_slider, gravity_slider

    def _make_preset_buttons(self) -> Tuple[Button, ...]:
        buttons: List[Button] = []
        for index, preset_name in enumerate(PRESETS):
            ax = self.fig.add_axes(
                [0.12 + index * 0.18, 0.315, 0.16, 0.048], facecolor=PANEL
            )
            button = Button(ax, preset_name, color="#18304a", hovercolor="#1f4068")
            button.label.set_color(TEXT)
            button.on_clicked(lambda _event, name=preset_name: self.apply_preset(name))
            buttons.append(button)
        return tuple(buttons)

    def _update_from_sliders(self, _value: float) -> None:
        count_slider, base_slider, step_slider, angle_slider, gravity_slider = (
            self.sliders
        )
        self.params.num_pendulums = int(count_slider.val)
        self.params.base_length = base_slider.val
        self.params.length_step = step_slider.val
        self.params.release_angle_deg = angle_slider.val
        self.params.gravity = gravity_slider.val
        self.on_change()

    def apply_preset(self, name: str) -> None:
        preset = PRESETS[name]
        values = (
            preset.num_pendulums,
            preset.base_length,
            preset.length_step,
            preset.release_angle_deg,
            preset.gravity,
        )
        for slider, value in zip(self.sliders, values):
            slider.set_val(value)


def _make_playback_buttons(
    fig: plt.Figure,
    on_start: Callable[[object], None],
    on_pause: Callable[[object], None],
    on_restart: Callable[[object], None],
) -> Tuple[Button, Button, Button]:
    """Create buttons to control the animation playback."""

    button_specs = (("Start", on_start), ("Pause", on_pause), ("Restart", on_restart))
    buttons: List[Button] = []
    for index, (label, callback) in enumerate(button_specs):
        ax = fig.add_axes([0.66 + index * 0.105, 0.315, 0.095, 0.048], facecolor=PANEL)
        button = Button(ax, label, color="#18304a", hovercolor="#1f4068")
        button.label.set_color(TEXT)
        button.on_clicked(callback)
        buttons.append(button)
    return tuple(buttons)  # type: ignore[return-value]


def _add_instructions(fig: plt.Figure) -> None:
    """Add kid-friendly explanation text above the animation."""

    fig.patch.set_facecolor(BACKGROUND)
    instruction_text = (
        "Ask a question, then test it: What happens with Moon gravity? What if every string is the same length?\n"
        "Kid clue: a pendulum is like a playground swing. Shorter strings make quicker swings; longer strings lag behind."
    )
    fig.text(0.12, 0.955, instruction_text, fontsize=10, color=TEXT, va="top")
    fig.text(
        0.12,
        0.012,
        "Presets: Rainbow = classic snake • Moon = slow motion • Same beat = all strings match",
        fontsize=9,
        color=SOFT_TEXT,
    )


def _add_card(fig: plt.Figure) -> None:
    """Draw a soft panel behind the controls."""

    card = FancyBboxPatch(
        (0.09, 0.025),
        0.84,
        0.355,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure,
        facecolor="#0d1b2d",
        edgecolor="#23415f",
        linewidth=1.2,
        alpha=0.95,
        zorder=-1,
    )
    fig.patches.append(card)


def build_demo() -> Tuple[plt.Figure, FuncAnimation]:
    """Build and return the Matplotlib figure and animation."""

    params = PendulumParameters()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.40)
    _add_card(fig)
    _add_instructions(fig)

    pendulum_plot = PendulumSnakePlot(ax, params)
    time_elapsed = 0.0
    timestep = 1 / 60
    is_running = True

    def on_slider_change() -> None:
        pendulum_plot.refresh()
        pendulum_plot.draw(time_elapsed)
        fig.canvas.draw_idle()

    controls = DemoControls(fig, params, on_slider_change)
    fig._pendulum_controls = controls  # type: ignore[attr-defined]

    def animate(_frame_index: int):
        nonlocal time_elapsed
        artists = pendulum_plot.draw(time_elapsed)
        time_elapsed += timestep
        return artists

    animation = FuncAnimation(
        fig,
        animate,
        frames=frame_generator(),
        interval=1000 / 60,
        blit=True,
        cache_frame_data=False,
    )
    fig._pendulum_animation = animation  # type: ignore[attr-defined]

    def start_animation(_event: object) -> None:
        nonlocal is_running
        if not is_running:
            animation.event_source.start()
            is_running = True

    def pause_animation(_event: object) -> None:
        nonlocal is_running
        if is_running:
            animation.event_source.stop()
            is_running = False

    def restart_animation(_event: object) -> None:
        nonlocal time_elapsed, is_running
        time_elapsed = 0.0
        pendulum_plot.draw(time_elapsed)
        fig.canvas.draw_idle()
        if not is_running:
            animation.event_source.start()
            is_running = True

    buttons = _make_playback_buttons(
        fig, start_animation, pause_animation, restart_animation
    )
    fig._pendulum_buttons = buttons  # type: ignore[attr-defined]

    return fig, animation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a kid-friendly pendulum snake demo."
    )
    parser.add_argument(
        "--save-preview",
        metavar="PATH",
        help="save a still preview image instead of opening the interactive window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, _animation = build_demo()
    if args.save_preview:
        fig.savefig(args.save_preview, dpi=160)
        print(f"Saved preview to {args.save_preview}")
        plt.close(fig)
    else:
        print(__doc__)
        plt.show()


if __name__ == "__main__":
    main()
