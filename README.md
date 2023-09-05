# PINNs Playground ✨



**Project Highlights** ✨

1. Automatically generate training history upon completion 📊
2. Easy to extend with minimal code, enabling rapid verification ✅

## Done and Todo ✅🔥

### Methods ✔️

- [x] Inverse Problem
- [x] Residual-based adaptive refinement
- [x] Self-adaptive activation function
- [x] Self-adaptive loss weights
- [x] Causal sampling
- [x] RAR-D
- [ ] More methods/algorithms

### PDE examples 📝

- [x] Burgers
- [x] NavierStocks2D
- [x] NavierStocks3D

### Others 🛠️

- [x] Add checkpoint
- [x] MLP model
- [x] Add log file
- [x] fastapi handle request
- [x] Visualization
- [ ] More Models

## Usage 🚀

1. Install dependencies 🛠️
   
   ```shell
   ... install pytorch
   >>> pip install fastapi tqdm rich matplotlib numpy pandas imageio
   ```

2. Write your PDE in `pdes`, and modify `config.common.py`'s function `pde_fn(cls):` to return your PDE

3. Create a train script, import `PINN` and `config`, and run `PINN(config, model).train()`. Check `train_burgers.py` for an example.

4. For inverse problems, just add `params_init` in the config, and modify `pde_fn` to accept `params` as input. Check `train_ns_inverse.py` for an example.

## Add modules 🧩

This framework is designed to be modular, so you can easily add your own modules. We provide a `Callback` class, which can be used to add your own callback functions.
