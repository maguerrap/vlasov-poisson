# vp_solver

A solver for the **1D–1D Vlasov–Poisson system** using a semi-Lagrangian
scheme, built with [JAX](https://github.com/google/jax) for high-performance
computing and automatic differentiation.

This project enables forward simulation of the Vlasov–Poisson equation
and optimization over external fields, with examples for
Two-Stream and Bump-on-Tail equilibria.

---

## Features

- ⚡ Semi-Lagrangian solver for the 1D–1D Vlasov–Poisson system  
- 🔁 Built with [JAX](https://github.com/google/jax) for GPU/TPU acceleration  
- 📈 Includes cost functions (KL divergence, electric energy, etc.)  
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

### Usage

After installation, you can import the solver:
```python
from vp_solver import Mesh, VlasovPoissonSolver
```

Examples
We provide Jupyter notebooks in the examples/ folder:

- Two-Stream Equilibrium
- Bump-on-Tail Equilibrium
To run them, launch Jupyter:

```bash
jupyter notebook examples/
```

### License

This project is licensed under the MIT License. See [LICENSE](https://github.com/maguerrap/vlasov-poisson/blob/main/LICENSE).
