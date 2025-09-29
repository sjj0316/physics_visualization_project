"""Minimal command-line animation for the triple pendulum."""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from .triple_pendulum import (
    TriplePendulumParams,
    TriplePendulumSimulator,
    compute_positions,
)


def animate(simulator: TriplePendulumSimulator, interval_ms: int = 16) -> None:
    params = simulator.params
    total_length = params.l1 + params.l2 + params.l3

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    margin = 0.15 * total_length + 0.2
    limit = total_length + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(True, linestyle="--", alpha=0.3)

    line, = ax.plot([], [], "o-", lw=2.5, color="tab:blue")
    tail, = ax.plot([], [], color="tab:red", alpha=0.6, lw=1.0)
    trace = []

    def update(_):
        simulator.step()
        positions = compute_positions(simulator.state, simulator.params)
        xs = [0.0, positions[0, 0], positions[1, 0], positions[2, 0]]
        ys = [0.0, positions[0, 1], positions[1, 1], positions[2, 1]]
        line.set_data(xs, ys)
        trace.append(positions[-1, :2].copy())
        if len(trace) > 600:
            trace.pop(0)
        trace_arr = np.array(trace)
        tail.set_data(trace_arr[:, 0], trace_arr[:, 1])
        return line, tail

    FuncAnimation(fig, update, interval=interval_ms, blit=True)
    plt.show()


if __name__ == "__main__":
    params = TriplePendulumParams()
    theta = np.radians([120.0, -10.0, 20.0])
    omega = np.radians([0.0, 0.0, 0.0])
    state = [theta[0], omega[0], theta[1], omega[1], theta[2], omega[2]]
    simulator = TriplePendulumSimulator(params, state, dt=0.002)
    animate(simulator)
