![Python 3.13](https://img.shields.io/badge/python-3.13-green.svg)

# Quantasio: A Generalized Neural Framework for Solving 3D Navier-Stokes Dynamics

Welcome to Quantasio! This tool leverages deep learning to approximate solutions for any general temporal PDE as well as fluid dynamics problems using the Navier-Stokes equations. The solver is designed to handle 2D, 3D, and 4D (time-dependent) cases and is capable of solving a range of scenarios from basic flows to turbulent regimes and flows around obstacles.

## Features
### General PDE Solver:

Supports solving temporal problems in:
1) 2D: Solves equations of the form u(x, t).
2) 3D: Extends to u(x, y, t).
3) 4D: Extends to u(x, y, z, t).
4) NS Solver: Approximates vector fields (u, v, w)(x, y, z, t) by solving the Navier Stokes Equation.

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

### Case Study 1: Visualizing a Basic Flow field by solving NS Equation
For a velocity field **u** = (u, v, w) in a 3D domain:
```math
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} - \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}\right) = 0
```
```math
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + w \frac{\partial v}{\partial z} - \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} + \frac{\partial^2 v}{\partial z^2}\right) = 0
```
```math
\frac{\partial w}{\partial t} + u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} - \nu \left(\frac{\partial^2 w}{\partial x^2} + \frac{\partial^2 w}{\partial y^2} + \frac{\partial^2 w}{\partial z^2}\right) = 0
```
Where:
- $\( u, v, w \)$: Velocity components in $\(x\)$, $\(y\)$, and $\(z\)$ directions.
- $\( \nu \)$: Kinematic viscosity.

### Initial Conditions:
The initial velocity field is specified as:

```math
u_0(x, y, z) = \sin(\pi x), \quad v_0(x, y, z) = 0, \quad w_0(x, y, z) = 0
```
### Domain:
The domain is defined as:
```math
x \in [0, 1], \quad y \in [0, 1], \quad z \in [0, 1], \quad t \in [0, 1]
```
### Residuals:
The deep learning model minimizes the residuals of the Navier-Stokes equations, which are defined as:

```math
f_u = \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} - \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}\right)
```

```math
f_v = \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + w \frac{\partial v}{\partial z} - \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} + \frac{\partial^2 v}{\partial z^2}\right)
```

```math
f_w = \frac{\partial w}{\partial t} + u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} - \nu \left(\frac{\partial^2 w}{\partial x^2} + \frac{\partial^2 w}{\partial y^2} + \frac{\partial^2 w}{\partial z^2}\right)
```

### Visualizing and animating the flow field using Quantasio

![](./images/navier_stokes_animation.gif)

### Case Study 2: Visualizing a Turbulent flow field with oscillatory boundary conditions

### Initial Conditions:
The initial velocity field is given by sinusoidal variations:

```math
u_0(x, y, z) = \sin(\pi x) \cos(\pi y)
```

```math
v_0(x, y, z) = \sin(\pi y) \cos(\pi z)
```

```math
w_0(x, y, z) = \sin(\pi z) \cos(\pi x)
```

### Oscillatory Boundary Conditions:
At the domain boundaries, the velocity field exhibits oscillatory behavior:

```math
u_b(x, y, z, t) = 0.5 \sin(2\pi t) \cos(\pi x)
```

```math
v_b(x, y, z, t) = 0.5 \sin(2\pi t) \cos(\pi y)
```

```math
w_b(x, y, z, t) = 0.5 \sin(2\pi t) \cos(\pi z)
```

### Domain:
The domain is defined as:

```math
x \in [0, 1], \quad y \in [0, 1], \quad z \in [0, 1], \quad t \in [0, 1]
```

### Visualizing and animating the flow field using Quantasio
![](./images/navier_stokes_turbulence.gif)

### Case Study 3: Visualizing Turbulent flow over a spherical obstacle

### Initial Conditions:
The velocity field is initially zero throughout the domain:

```math
u_0(x, y, z) = 0, \quad v_0(x, y, z) = 0, \quad w_0(x, y, z) = 0
```

### Boundary Conditions:
1. **Inflow (Uniform Flow):**
```math
   u(x, 0, z, t) = 1, \quad v(x, 0, z, t) = 0, \quad w(x, 0, z, t) = 0
```
2. **No-slip condition on obstacle surface:**
   On the spherical obstacle centered at $\( (0.5, 0.5, 0.5) \)$ with radius $\( r = 0.1 \)$,
```math
   u = v = w = 0
```
3. **Outflow (Zero-gradient boundary condition):**
   At the outflow boundary, the velocity gradients are zero.

### Domain:
The computational domain is defined as:
```math
x \in [0, 1], \quad y \in [0, 1], \quad z \in [0, 1], \quad t \in [0, 1]
```

### Obstacle Representation:
The obstacle is represented as a sphere:
```math
(x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2 \leq 0.1^2
```

### Visualizing and animating the flow field using Quantasio
![](./images/navier_stokes_obstacle.gif)


## 📄 Citation

[Paper Link](https://ieeexplore.ieee.org/abstract/document/10927473)

If you find this repository useful in your research, please cite my work as follows:

```bibtex
@ARTICLE{11045321,
  author={Sarker, Soumick and Chakraborty, Sudipto},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={Quantasio: A Generalized Neural Framework for Solving 3D Navier-Stokes Dynamics}, 
  year={2025},
  volume={},
  number={},
  pages={1-10},
  keywords={Mathematical models;Training;Boundary conditions;Neural networks;Convergence;Accuracy;Modulation;Scalability;Real-time systems;Geometry;Quantasio;Physics-Informed Neural Networks;Partial Differential Equations;Navier-Stokes Equations;Turbulent Flow},
  doi={10.1109/TAI.2025.3581506}}
