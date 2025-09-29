"""Core physics model for a planar triple pendulum.

This module exposes tools for numerically integrating the motion of a
three-link pendulum where each link is assumed to be massless and each mass is
concentrated at the joint.  The equations of motion are derived symbolically
with :mod:`sympy` and exported to efficient :mod:`numpy` functions.

The public API is intentionally small so that it can be consumed from a UI or a
command line driver alike.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Tuple

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class TriplePendulumParams:
    """Physical constants for the triple pendulum.

    Attributes
    ----------
    m1, m2, m3:
        Mass of the bob at each link, in kilograms.
    l1, l2, l3:
        Length of each link, in metres.
    g:
        Gravitational acceleration, in metres per second squared.
    """

    m1: float = 1.0
    m2: float = 1.0
    m3: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    l3: float = 1.0
    g: float = 9.81

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.m1, self.m2, self.m3, self.l1, self.l2, self.l3, self.g)


StateVector = np.ndarray


def _derive_acceleration_function():
    """Return a numpy callable computing angular accelerations.

    The derivation relies on the Lagrangian formulation.  We keep the routine
    isolated so that the symbolic work happens at module import only once.
    """

    m1, m2, m3, l1, l2, l3, g = sp.symbols("m1 m2 m3 l1 l2 l3 g", positive=True)
    theta1, theta2, theta3 = sp.symbols("theta1 theta2 theta3")
    omega1, omega2, omega3 = sp.symbols("omega1 omega2 omega3")
    alpha1, alpha2, alpha3 = sp.symbols("alpha1 alpha2 alpha3")

    thetas = (theta1, theta2, theta3)
    omegas = (omega1, omega2, omega3)
    alphas = (alpha1, alpha2, alpha3)
    masses = (m1, m2, m3)
    lengths = (l1, l2, l3)

    # Cartesian coordinates of each mass centre.
    x_coords = []
    y_coords = []
    for idx in range(3):
        x_k = sum(lengths[i] * sp.sin(thetas[i]) for i in range(idx + 1))
        y_k = -sum(lengths[i] * sp.cos(thetas[i]) for i in range(idx + 1))
        x_coords.append(x_k)
        y_coords.append(y_k)

    # Velocities are obtained via d/dt of the coordinates.  Because the angles
    # are the generalised coordinates the velocity components can be written as
    # linear combinations of angular velocities.
    vx = []
    vy = []
    for idx in range(3):
        vx_k = sum(lengths[i] * omegas[i] * sp.cos(thetas[i]) for i in range(idx + 1))
        vy_k = sum(lengths[i] * omegas[i] * sp.sin(thetas[i]) for i in range(idx + 1))
        vx.append(vx_k)
        vy.append(vy_k)

    kinetic = sp.Rational(1, 2) * sum(
        masses[i] * (vx[i] ** 2 + vy[i] ** 2) for i in range(3)
    )
    potential = sum(masses[i] * g * y_coords[i] for i in range(3))
    lagrangian = kinetic - potential

    equations = []
    for i in range(3):
        d_l_domega = sp.diff(lagrangian, omegas[i])
        d_l_dtheta = sp.diff(lagrangian, thetas[i])

        total_time_derivative = 0
        for j in range(3):
            total_time_derivative += sp.diff(d_l_domega, thetas[j]) * omegas[j]
            total_time_derivative += sp.diff(d_l_domega, omegas[j]) * alphas[j]

        equations.append(sp.simplify(total_time_derivative - d_l_dtheta))

    solution = sp.solve(equations, alphas, simplify=True, rational=False)

    accelerations = [sp.simplify(solution[alpha]) for alpha in alphas]

    variables = (
        theta1,
        theta2,
        theta3,
        omega1,
        omega2,
        omega3,
        m1,
        m2,
        m3,
        l1,
        l2,
        l3,
        g,
    )

    return sp.lambdify(variables, accelerations, modules="numpy")


@lru_cache(maxsize=1)
def _get_acceleration_function():
    return _derive_acceleration_function()


def triple_pendulum_derivatives(state: Iterable[float], params: TriplePendulumParams) -> np.ndarray:
    """Return the time derivatives for the given state."""

    theta1, omega1, theta2, omega2, theta3, omega3 = state
    acc_func = _get_acceleration_function()
    alpha1, alpha2, alpha3 = acc_func(
        theta1,
        theta2,
        theta3,
        omega1,
        omega2,
        omega3,
        params.m1,
        params.m2,
        params.m3,
        params.l1,
        params.l2,
        params.l3,
        params.g,
    )
    return np.array([omega1, alpha1, omega2, alpha2, omega3, alpha3], dtype=float)


class TriplePendulumSimulator:
    """Simple fourth-order Runge--Kutta integrator for the triple pendulum."""

    def __init__(self, params: TriplePendulumParams, initial_state: Iterable[float], dt: float = 0.005):
        self.params = params
        self.state = np.array(initial_state, dtype=float)
        self.dt = float(dt)
        self.time = 0.0

    def step(self) -> np.ndarray:
        """Advance the simulation by a single integration step."""

        y = self.state
        dt = self.dt
        k1 = triple_pendulum_derivatives(y, self.params)
        k2 = triple_pendulum_derivatives(y + 0.5 * dt * k1, self.params)
        k3 = triple_pendulum_derivatives(y + 0.5 * dt * k2, self.params)
        k4 = triple_pendulum_derivatives(y + dt * k3, self.params)
        self.state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.time += dt
        return self.state

    def step_many(self, steps: int) -> np.ndarray:
        """Advance the simulation by ``steps`` integration steps."""

        for _ in range(int(steps)):
            self.step()
        return self.state

    def copy_state(self) -> np.ndarray:
        return np.array(self.state, dtype=float)


def compute_positions(state: Iterable[float], params: TriplePendulumParams) -> np.ndarray:
    """Return cartesian coordinates for each mass.

    The result is an ``(3, 3)`` array containing ``(x, y, z)`` for each mass.
    All masses move in the same vertical plane, so the ``z`` component is zero.
    """

    theta1, _, theta2, _, theta3, _ = state
    l1, l2, l3 = params.l1, params.l2, params.l3

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    x3 = x2 + l3 * np.sin(theta3)
    y3 = y2 - l3 * np.cos(theta3)

    positions = np.array(
        [
            [x1, y1, 0.0],
            [x2, y2, 0.0],
            [x3, y3, 0.0],
        ],
        dtype=float,
    )
    return positions


def total_energy(state: Iterable[float], params: TriplePendulumParams) -> float:
    """Compute the total mechanical energy of the system."""

    theta1, omega1, theta2, omega2, theta3, omega3 = state
    m1, m2, m3 = params.m1, params.m2, params.m3
    l1, l2, l3 = params.l1, params.l2, params.l3
    g = params.g

    positions = compute_positions(state, params)
    velocities = []
    omegas = np.array([omega1, omega2, omega3], dtype=float)
    lengths = np.array([l1, l2, l3], dtype=float)
    thetas = np.array([theta1, theta2, theta3], dtype=float)

    for idx in range(3):
        vx = np.sum(lengths[: idx + 1] * omegas[: idx + 1] * np.cos(thetas[: idx + 1]))
        vy = np.sum(lengths[: idx + 1] * omegas[: idx + 1] * np.sin(thetas[: idx + 1]))
        velocities.append((vx, vy))

    velocities = np.array(velocities)
    masses = np.array([m1, m2, m3], dtype=float)

    kinetic = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    potential = np.sum(masses * g * positions[:, 1])
    return float(kinetic + potential)


__all__ = [
    "TriplePendulumParams",
    "TriplePendulumSimulator",
    "compute_positions",
    "total_energy",
    "triple_pendulum_derivatives",
]
