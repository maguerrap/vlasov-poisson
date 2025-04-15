# Vlasov Poisson

Here are the codes used for the simulation of the Vlasov-Poisson system in the paper [What metric to optimize for suppressing instability in a Vlasov-Poisson system?](https://arxiv.org/abs/2504.10435).

This repository contains:

- A solver for the 1D-1D Vlasov-Poisson equation using via Semi-Lagrangian with linear interpolation using `JAX`.
- A solver for the inverse problem of suppressing instability of the Vlasov-Poisson using an external electric field via PDE-constrained optimization using `optax`
