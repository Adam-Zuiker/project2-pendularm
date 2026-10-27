"""tools/pendularm_viz.py — Interactive visualization for the Pendularm
arm: live rendering, PID gain sliders, a 2-link/3-link switch, and
click-to-set starting position / PID setpoint.

Runs entirely locally against your own ArmRuntime/dynamics/PID/IK code —
no network/TCP involved, so it works whether or not the actual rosbridge
server (`make run`) is running. This is a debugging/exploration tool, not
part of the graded submission — it lives outside starter/python/ on
purpose, so make_submission.sh never bundles it (it depends on
matplotlib, which isn't standard library).

Everything here runs single-threaded (the animation callback is the only
thing ever calling into ArmRuntime), so reaching into ArmRuntime's private
_q/_qdot/_t fields directly for the "teleport" feature is safe — there's
no physics-loop thread to race with, unlike the real server.

Usage:
    python3 tools/pendularm_viz.py [2|3]

Controls:
    - kp / ki / kd sliders: applied uniformly to every joint
    - 2 links / 3 links radio buttons: restarts with a fresh arm
    - Enable PID: toggles the controller on/off
    - Pause / Resume: freezes the physics (same as /arm_sim/pause)
    - Reset to home: snaps back to the all-zero pose
    - Left-click on the plot: teleport the arm to that position (IK, no
      dynamics involved — an instant "starting position")
    - Right-click on the plot: command that position as the PID setpoint
      (drives there dynamically, same as /joint_trajectory)
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "starter", "python", "src"))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider

from arm_lib.integrators import INTEGRATORS
from arm_runtime import ArmRuntime
from kinematics import solve_ik

INTEGRATOR_NAMES = ("euler", "midpoint", "verlet", "rk4")
assert set(INTEGRATOR_NAMES) == set(INTEGRATORS), "keep this in sync with arm_lib.integrators"

STEPS_PER_FRAME = 4  # physics steps per redraw, for smoother/faster-than-realtime motion
TRACE_MAX_POINTS = 500


class PendularmViz:
    def __init__(self, n_links: int = 2) -> None:
        self.n_links = n_links
        self.runtime = ArmRuntime(n_links)
        self._target = None  # (x, y) last commanded via right-click

        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_axes((0.06, 0.35, 0.6, 0.6))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Pendularm")
        self._rescale_axes()

        (self.arm_line,) = self.ax.plot([], [], "o-", lw=4, markersize=8, color="#2563eb")
        (self.target_marker,) = self.ax.plot([], [], "x", markersize=12, color="#dc2626")
        (self.trace_line,) = self.ax.plot([], [], "-", lw=1, alpha=0.3, color="#2563eb")
        self.trace_x: list[float] = []
        self.trace_y: list[float] = []

        self.status_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, va="top", fontsize=9, family="monospace"
        )

        self._build_controls()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.anim = FuncAnimation(self.fig, self._update, interval=16, blit=False)

    def _rescale_axes(self) -> None:
        reach = sum(self.runtime.get_params()[1]) * 1.2
        self.ax.set_xlim(-reach, reach)
        self.ax.set_ylim(-reach, reach)

    def _build_controls(self) -> None:
        axcolor = "lightgoldenrodyellow"
        ax_kp = self.fig.add_axes((0.72, 0.90, 0.24, 0.03), facecolor=axcolor)
        ax_ki = self.fig.add_axes((0.72, 0.85, 0.24, 0.03), facecolor=axcolor)
        ax_kd = self.fig.add_axes((0.72, 0.80, 0.24, 0.03), facecolor=axcolor)
        self.s_kp = Slider(ax_kp, "kp", 0.0, 200.0, valinit=40.0)
        self.s_ki = Slider(ax_ki, "ki", 0.0, 50.0, valinit=5.0)
        self.s_kd = Slider(ax_kd, "kd", 0.0, 50.0, valinit=8.0)
        self.s_kp.on_changed(self._on_gains_changed)
        self.s_ki.on_changed(self._on_gains_changed)
        self.s_kd.on_changed(self._on_gains_changed)
        self._on_gains_changed(None)  # apply the initial slider values

        ax_integrator = self.fig.add_axes((0.72, 0.62, 0.2, 0.15), facecolor=axcolor)
        self.radio_integrator = RadioButtons(ax_integrator, INTEGRATOR_NAMES, active=INTEGRATOR_NAMES.index("rk4"))
        self.radio_integrator.on_clicked(self._on_integrator_changed)
        self._on_integrator_changed(self.radio_integrator.value_selected)

        ax_links = self.fig.add_axes((0.72, 0.50, 0.2, 0.10), facecolor=axcolor)
        self.radio_links = RadioButtons(ax_links, ("2 links", "3 links"), active=0 if self.n_links == 2 else 1)
        self.radio_links.on_clicked(self._on_links_changed)

        ax_enable = self.fig.add_axes((0.72, 0.42, 0.24, 0.05))
        self.btn_enable = Button(ax_enable, "Enable PID: OFF")
        self.btn_enable.on_clicked(self._on_toggle_pid)

        ax_pause = self.fig.add_axes((0.72, 0.35, 0.24, 0.05))
        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_pause.on_clicked(self._on_toggle_pause)

        ax_reset = self.fig.add_axes((0.72, 0.28, 0.24, 0.05))
        self.btn_reset = Button(ax_reset, "Reset to home")
        self.btn_reset.on_clicked(self._on_reset)

        self.fig.text(0.72, 0.18, "Left-click: teleport start\nRight-click: PID setpoint", fontsize=8)

    # -- control callbacks -------------------------------------------------

    def _on_gains_changed(self, _val) -> None:
        n = self.n_links
        kp, ki, kd = self.s_kp.val, self.s_ki.val, self.s_kd.val
        self.runtime.set_pid_gains(kp=[kp] * n, ki=[ki] * n, kd=[kd] * n)

    def _on_integrator_changed(self, label: str) -> None:
        self.runtime.set_integrator(method=label)

    def _on_links_changed(self, label: str) -> None:
        self.n_links = 2 if label == "2 links" else 3
        self.runtime = ArmRuntime(self.n_links)
        self._on_gains_changed(None)
        self._on_integrator_changed(self.radio_integrator.value_selected)
        self._rescale_axes()
        self.trace_x, self.trace_y = [], []
        self._target = None
        self.btn_enable.label.set_text("Enable PID: OFF")
        self.btn_pause.label.set_text("Pause")

    def _on_toggle_pid(self, _event) -> None:
        new_state = not self.runtime.is_pid_enabled()
        self.runtime.enable_pid(new_state)
        self.btn_enable.label.set_text(f"Enable PID: {'ON' if new_state else 'OFF'}")

    def _on_toggle_pause(self, _event) -> None:
        new_state = not self.runtime.is_paused()
        self.runtime.set_pause(new_state)
        self.btn_pause.label.set_text("Resume" if new_state else "Pause")

    def _on_reset(self, _event) -> None:
        self.runtime.reset()
        self.trace_x, self.trace_y = [], []
        self._target = None

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        lengths = self.runtime.get_params()[1]
        try:
            q = solve_ik(lengths, x, y, None)
        except ValueError:
            return  # unreachable click, ignore
        if event.button == 1:  # left click: teleport starting position
            with self.runtime._lock:  # single-threaded GUI, safe -- see module docstring
                self.runtime._q = list(q)
                self.runtime._qdot = [0.0] * self.n_links
                self.runtime._t = 0.0
            self.trace_x, self.trace_y = [], []
            self._target = None
        elif event.button == 3:  # right click: PID setpoint
            self.runtime.set_joint_trajectory_setpoint(list(q), [0.0] * self.n_links)
            self._target = (x, y)

    # -- animation -----------------------------------------------------

    def _update(self, _frame):
        if not self.runtime.is_paused():
            for _ in range(STEPS_PER_FRAME):
                self.runtime.step()

        t, q, qdot, effort = self.runtime.get_joint_state()
        lengths = self.runtime.get_params()[1]

        xs, ys = [0.0], [0.0]
        phi, x, y = 0.0, 0.0, 0.0
        for length, qi in zip(lengths, q):
            phi += qi
            x += length * math.cos(phi)
            y += length * math.sin(phi)
            xs.append(x)
            ys.append(y)
        self.arm_line.set_data(xs, ys)

        self.trace_x.append(x)
        self.trace_y.append(y)
        if len(self.trace_x) > TRACE_MAX_POINTS:
            self.trace_x = self.trace_x[-TRACE_MAX_POINTS:]
            self.trace_y = self.trace_y[-TRACE_MAX_POINTS:]
        self.trace_line.set_data(self.trace_x, self.trace_y)

        if self._target is not None:
            self.target_marker.set_data([self._target[0]], [self._target[1]])
        else:
            self.target_marker.set_data([], [])

        self.status_text.set_text(
            f"t={t:6.2f}s\n"
            f"q={[round(v, 2) for v in q]}\n"
            f"qdot={[round(v, 2) for v in qdot]}\n"
            f"effort={[round(v, 1) for v in effort]}\n"
            f"PID={'ON' if self.runtime.is_pid_enabled() else 'OFF'}"
            f"{'  PAUSED' if self.runtime.is_paused() else ''}"
        )
        return self.arm_line, self.trace_line, self.target_marker, self.status_text

    def show(self) -> None:
        plt.show()


if __name__ == "__main__":
    n = 2
    if len(sys.argv) > 1 and sys.argv[1] in ("2", "3"):
        n = int(sys.argv[1])
    PendularmViz(n).show()
