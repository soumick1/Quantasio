![Python 3.8](https://img.shields.io/badge/python-3.10-green.svg)

# Quantasio: A Generalized Neural Framework for Solving Temporal Partial Differential Equations and Navier-Stokes Dynamics in Three Spatial Dimensions

Welcome to Quantasio! This tool leverages deep learning to approximate solutions for any general temporal PDE as well as fluid dynamics problems using the Navier-Stokes equations. The solver is designed to handle 2D, 3D, and 4D (time-dependent) cases and is capable of solving a range of scenarios from basic flows to turbulent regimes and flows around obstacles.

## Features
### General PDE Solver:

Supports solving temporal problems in:
1) 2D: Solves equations of the form u(x, y, t).
2) 3D: Extends to u(x, y, z, t).
3) 4D: Approximates vector fields (u, v, w)(x, y, z, t).

### Navier-Stokes Applications:

Solves the Navier-Stokes equations for incompressible fluids.
Handles boundary conditions such as inflow, outflow, and no-slip surfaces.
#### Scenarios:

Basic Navier-Stokes Problem: Steady or unsteady flows in a regular domain.
Turbulent Flow: Approximates turbulent-like behavior using adjusted initial and boundary conditions.
Flow Around Obstacles: Simulates flows in domains with internal obstacles, such as a sphere.
Visualization:

Visualize the solutions with animations showing the evolution of velocity fields over time.
Support for rendering vector fields in 2D, 3D, and 4D domains with obstacles.
