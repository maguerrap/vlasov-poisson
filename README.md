# Vlasov-Poisson solver

A solver for the **1D–1D Vlasov–Poisson system** using a semi-Lagrangian
scheme, built with [JAX](https://github.com/google/jax) for high-performance
computing and automatic differentiation.

This project enables forward simulation of the Vlasov–Poisson equation
and optimization over external fields, with examples for
Two-Stream and Bump-on-Tail equilibria.

---

## Mathematical Model

We solve the **Vlasov–Poisson system** in one spatial and one velocity
dimension, given by

$$
\left\leftbrace\begin{array}{lc}
   \partial_{t}f + v\partial_{x}f - (E_{f}+H) \partial_{v}f = 0 \,,\\
   E_{f} = \partial_{x}V_{f} \,, \\
   \partial_{xx} V_{f} = 1 - \rho_{f} =1 - \int  f \,\mathrm{d}v\,.
\end{array}\right.
$$

where:
- $f(t, x, v)$ is the plasma distribution function,
- $E(t, x)$ is the self-consistent electric field,
- $H(x)$ is an external electric field (control),
- $\rho_{f}(t,x)$ is the charged density,
- $f_{\text{eq}}(v)$ is the equilibrium distribution,
- $x \in [0, L_x]$ and $v \in [-L_v, L_v]$.

---

## Features

- ⚡ Semi-Lagrangian solver for the 1D–1D Vlasov–Poisson system  
- 🔁 Built with [JAX](https://github.com/google/jax) for GPU/TPU acceleration  
- 📈 Includes cost functions:
  - KL divergence vs. equilibrium
  - Final electric energy
  - Time-integrated electric energy   
- 📊 Visualization utilities for distributions and fields  
- 📓 Example Jupyter notebooks for experiments  

---

## Installation

### Requirements
- Python **3.12+**
- [pip](https://pip.pypa.io/en/stable/) for installation  
- NVIDIA GPU with recent drivers (if using GPU JAX)

### CPU Version (default)
To install the solver with CPU JAX:

```bash
pip install git+https://github.com/maguerrap/vlasov-poisson.git@main
```

### GPU Version (recommended)
For GPU acceleration, first install JAX with CUDA support.
Check [JAX installation docs](https://docs.jax.dev/en/latest/installation.html) for instructions.

Then install the solver:

```bash
pip install git+https://github.com/maguerrap/vlasov-poisson.git@main
```

## Usage

After installation, you can import the solver:
```python
from vp_solver import Mesh, VlasovPoissonSolver
```

Examples
We provide Jupyter notebooks in the examples/ folder:

- Two-Stream Equilibrium
- Bump-on-Tail Equilibrium

In these notebooks, we:

- Run forward simulations of the Vlasov–Poisson system
- Use [Optax](https://github.com/google-deepmind/optax/tree/main) to solve a PDE-constrained optimization problem:
-The goal is to design the external field  $H(x)$ that minimizes a chosen objective (e.g. KL divergence or electric energy).

To run them, launch Jupyter:

```bash
jupyter notebook examples/
```

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/maguerrap/vlasov-poisson/blob/main/LICENSE).
