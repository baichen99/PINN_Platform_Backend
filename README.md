# PINNs Playground

## Done and Todo

### Methods

- [x] Inverse Problem
- [x] Residual-based adaptive refinement
- [x] Self-adaptive activation function
- [x] Self-adaptive loss weights
- [x] Causal sampling
- [x] RAR-D
- [ ] More methods/algorithms

### PDE examples

- [x] Burgers
- [x] NavierStocks2D
- [x] NavierStocks3D

### Others

- [x] Add checkpoint
- [x] MLP model
- [x] Add log file
- [x] fastapi handle request
- [x] Visualization
- [ ] More Models

## Usage

1. Install dependencies

2. write your pde in `pdes`, and modify `config.common.py` 's function `pde_fn(cls):` to return your pde

3. create a train script, import `PINN` and `config`, and run `PINN(config, model).train()`. Check `train_burgers.py` for example.
4. for inverse problem, just add `params_init` in config, and modify `pde_fn` accept `params` as input, check `train_ns_inverse.py` for example.

## Add modules

This framework is designed to be modular, so you can add your own modules easily. We provide a `Callback` class, which can be used to add your own callback functions.