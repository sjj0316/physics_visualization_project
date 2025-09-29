"""PyQt based desktop application for exploring triple pendulum dynamics."""
from __future__ import annotations

import math
import sys
from collections import deque
from typing import Deque, Iterable

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection

from .triple_pendulum import (
    TriplePendulumParams,
    TriplePendulumSimulator,
    compute_positions,
    total_energy,
)


class Pendulum2DCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(5, 5))
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.line, = self.ax.plot([], [], "o-", lw=2.5, color="tab:blue")
        self.trace, = self.ax.plot([], [], color="tab:red", alpha=0.6, lw=1.2)
        self.figure.tight_layout()

    def set_world_size(self, total_length: float) -> None:
        margin = 0.15 * total_length + 0.2
        limit = total_length + margin
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.draw()

    def update(self, positions: np.ndarray, trace: Iterable[np.ndarray]) -> None:
        xs = [0.0, positions[0, 0], positions[1, 0], positions[2, 0]]
        ys = [0.0, positions[0, 1], positions[1, 1], positions[2, 1]]
        self.line.set_data(xs, ys)

        if trace:
            trace_arr = np.array(trace)
            self.trace.set_data(trace_arr[:, 0], trace_arr[:, 1])
        else:
            self.trace.set_data([], [])

        self.draw_idle()


class Pendulum3DCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(5, 5))
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.view_init(elev=25.0, azim=-60.0)
        self.line, = self.ax.plot([], [], [], "o-", lw=2.5, color="tab:blue")
        self.trace, = self.ax.plot([], [], [], color="tab:orange", alpha=0.6, lw=1.0)
        self.ax.set_box_aspect([1, 1, 0.5])
        self.figure.tight_layout()

    def set_world_size(self, total_length: float) -> None:
        margin = 0.15 * total_length + 0.2
        limit = total_length + margin
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-0.5 * limit, 0.5 * limit)
        self.draw()

    def update(self, positions: np.ndarray, trace: Iterable[np.ndarray]) -> None:
        xs = [0.0, positions[0, 0], positions[1, 0], positions[2, 0]]
        ys = [0.0, positions[0, 1], positions[1, 1], positions[2, 1]]
        zs = [0.0, positions[0, 2], positions[1, 2], positions[2, 2]]
        self.line.set_data(xs, ys)
        self.line.set_3d_properties(zs)

        if trace:
            trace_arr = np.array(trace)
            self.trace.set_data(trace_arr[:, 0], trace_arr[:, 1])
            self.trace.set_3d_properties(trace_arr[:, 2])
        else:
            self.trace.set_data([], [])
            self.trace.set_3d_properties([])

        self.draw_idle()


class TriplePendulumWindow(QtWidgets.QWidget):
    TIMER_INTERVAL_MS = 16  # approx 60 FPS

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Triple Pendulum Playground")
        self.simulator: TriplePendulumSimulator | None = None
        self.trace_history: Deque[np.ndarray] = deque()
        self.steps_per_frame: int = 1
        self.is_running = False

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.TIMER_INTERVAL_MS)
        self.timer.timeout.connect(self._on_timer)

        self._build_ui()

    # ------------------------------------------------------------------ UI setup
    def _build_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)

        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self._build_parameters_group())
        controls_layout.addWidget(self._build_initial_conditions_group())
        controls_layout.addStretch(1)
        controls_layout.addWidget(self._build_status_panel())
        controls_layout.addWidget(self._build_buttons())

        figures_layout = QtWidgets.QVBoxLayout()
        self.canvas2d = Pendulum2DCanvas()
        self.canvas3d = Pendulum3DCanvas()
        figures_layout.addWidget(self.canvas2d, stretch=1)
        figures_layout.addWidget(self.canvas3d, stretch=1)

        main_layout.addLayout(controls_layout, stretch=0)
        main_layout.addLayout(figures_layout, stretch=1)

    def _build_parameters_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Physical parameters")
        layout = QtWidgets.QFormLayout(box)

        self.mass_spins = [self._create_spin_box(0.1, 10.0, 0.1, 1.0) for _ in range(3)]
        self.length_spins = [self._create_spin_box(0.1, 5.0, 0.05, 1.0) for _ in range(3)]
        self.gravity_spin = self._create_spin_box(0.1, 30.0, 0.1, 9.81)
        self.dt_spin = self._create_spin_box(0.0005, 0.05, 0.0005, 0.002, decimals=4)

        layout.addRow("Mass m₁ (kg)", self.mass_spins[0])
        layout.addRow("Mass m₂ (kg)", self.mass_spins[1])
        layout.addRow("Mass m₃ (kg)", self.mass_spins[2])
        layout.addRow("Length l₁ (m)", self.length_spins[0])
        layout.addRow("Length l₂ (m)", self.length_spins[1])
        layout.addRow("Length l₃ (m)", self.length_spins[2])
        layout.addRow("Gravity g (m/s²)", self.gravity_spin)
        layout.addRow("Time step dt (s)", self.dt_spin)

        return box

    def _build_initial_conditions_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Initial state")
        layout = QtWidgets.QFormLayout(box)

        # Angles provided in degrees for ease of input.
        self.angle_spins = [self._create_spin_box(-179.0, 179.0, 1.0, 120.0) for _ in range(3)]
        self.velocity_spins = [self._create_spin_box(-720.0, 720.0, 5.0, 0.0) for _ in range(3)]

        layout.addRow("θ₁ (deg)", self.angle_spins[0])
        layout.addRow("θ₂ (deg)", self.angle_spins[1])
        layout.addRow("θ₃ (deg)", self.angle_spins[2])
        layout.addRow("ω₁ (deg/s)", self.velocity_spins[0])
        layout.addRow("ω₂ (deg/s)", self.velocity_spins[1])
        layout.addRow("ω₃ (deg/s)", self.velocity_spins[2])

        return box

    def _build_status_panel(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Simulation status")
        layout = QtWidgets.QVBoxLayout(box)
        self.time_label = QtWidgets.QLabel("Time: 0.000 s")
        self.energy_label = QtWidgets.QLabel("Energy: 0.000 J")
        layout.addWidget(self.time_label)
        layout.addWidget(self.energy_label)
        return box

    def _build_buttons(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.start_button = QtWidgets.QPushButton("Start / Restart")
        self.start_button.clicked.connect(self.start_simulation)
        layout.addWidget(self.start_button)

        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)
        layout.addWidget(self.pause_button)

        return widget

    def _create_spin_box(
        self,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        *,
        decimals: int = 3,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    # ---------------------------------------------------------------- simulation
    def start_simulation(self) -> None:
        params = TriplePendulumParams(
            m1=self.mass_spins[0].value(),
            m2=self.mass_spins[1].value(),
            m3=self.mass_spins[2].value(),
            l1=self.length_spins[0].value(),
            l2=self.length_spins[1].value(),
            l3=self.length_spins[2].value(),
            g=self.gravity_spin.value(),
        )

        total_length = params.l1 + params.l2 + params.l3
        self.canvas2d.set_world_size(total_length)
        self.canvas3d.set_world_size(total_length)

        theta = [math.radians(spin.value()) for spin in self.angle_spins]
        omega = [math.radians(spin.value()) for spin in self.velocity_spins]
        initial_state = [theta[0], omega[0], theta[1], omega[1], theta[2], omega[2]]

        dt = self.dt_spin.value()
        self.simulator = TriplePendulumSimulator(params, initial_state, dt=dt)
        self.steps_per_frame = max(1, int(round((self.TIMER_INTERVAL_MS / 1000.0) / dt)))
        history_length = max(150, int(8.0 / dt))
        self.trace_history = deque(maxlen=history_length)
        self._append_trace_point()

        self.time_label.setText("Time: 0.000 s")
        energy = total_energy(self.simulator.state, params)
        self.energy_label.setText(f"Energy: {energy:0.3f} J")

        self.pause_button.setEnabled(True)
        self.pause_button.setText("Pause")
        self.is_running = True
        self.timer.start()

    def toggle_pause(self) -> None:
        if not self.simulator:
            return

        if self.is_running:
            self.timer.stop()
            self.is_running = False
            self.pause_button.setText("Resume")
        else:
            self.timer.start()
            self.is_running = True
            self.pause_button.setText("Pause")

    def _append_trace_point(self) -> None:
        if not self.simulator:
            return
        positions = compute_positions(self.simulator.state, self.simulator.params)
        self.trace_history.append(positions[-1].copy())

    def _on_timer(self) -> None:
        if not self.simulator:
            return

        self.simulator.step_many(self.steps_per_frame)
        params = self.simulator.params
        positions = compute_positions(self.simulator.state, params)
        self._append_trace_point()

        self.canvas2d.update(positions, self.trace_history)
        self.canvas3d.update(positions, [(p[0], p[1], 0.0) for p in self.trace_history])

        self.time_label.setText(f"Time: {self.simulator.time:0.3f} s")
        energy = total_energy(self.simulator.state, params)
        self.energy_label.setText(f"Energy: {energy:0.3f} J")

    # ---------------------------------------------------------------------- misc
    def closeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 (Qt naming)
        if self.timer.isActive():
            self.timer.stop()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = TriplePendulumWindow()
    window.resize(1200, 700)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
