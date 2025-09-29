# Physics Visualisation Project

An interactive toolkit for exploring the chaotic motion of a planar triple
pendulum.  The project offers both a fully fledged desktop user interface and a
lightweight command-line animation so you can examine the dynamics in 2D or 3D
with adjustable physical parameters.

## Features

- Accurate equations of motion derived with symbolic calculus (via SymPy).
- Real-time 2D and 3D visualisations with Matplotlib embedded in a PyQt5
  interface.
- Interactive control of masses, rod lengths, gravity, integration time-step,
  starting angles, and initial angular velocities.
- Energy read-out and trajectory trail to help diagnose chaotic behaviour.
- Auxiliary command-line script for a quick 2D Matplotlib animation without the
  full UI.

## Getting started

The application targets Python 3.9+ and runs on Windows, macOS, and Linux.  To
keep your system tidy we recommend creating a virtual environment before
installing the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
pip install -r requirements.txt
```

On Windows you can use the pre-installed launcher:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Launching the desktop interface

Once the dependencies are in place, start the PyQt5 application:

```bash
python -m src.app
```

A window titled **Triple Pendulum Playground** will open with side panels for
adjusting the masses, lengths, gravity, time-step, and initial conditions.  Use
“Start / Restart” to apply the current configuration and “Pause” to temporarily
halt the integration.  Both 2D and 3D views update in real time and display the
trajectory of the final pendulum bob.

### Running the lightweight 2D animation

If you only need a simple Matplotlib animation without the UI, execute:

```bash
python -m src.triple_sympy
```

This starts a simulation with default parameters and displays a 2D animation of
the triple pendulum’s motion.

## Repository structure

- `src/triple_pendulum.py` — core physics model, numerical integration, and
  energy helpers.
- `src/app.py` — PyQt5 desktop application with embedded Matplotlib canvases.
- `src/triple_sympy.py` — command-line helper for a quick 2D animation.
- `requirements.txt` — third-party dependencies.

## Notes

The symbolic derivation of the equations happens when `src.triple_pendulum` is
imported for the first time.  This may take a second or two on the first run but
is cached afterwards.
