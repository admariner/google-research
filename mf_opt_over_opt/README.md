# Xin Gu's SRP (Summer 26)

---

- `mca/`: Monte Carlo Accounting for Approximate Gaussian DP
- `dpmf/`: Beyond RMSE for DP-MF Objectives

> **Note**: Invocations below assume you are running from the root of the
> repository (or have `google_research` added to your `PYTHONPATH`).

## Monte Carlo Accounting (`mca/`)

All `algo_` files (`alg5_jax.py`, `alg5_numpy.py`, `algo4p5_jax.py`,
`algo4p5_numpy.py`, `algo4p9_jax.py`, `algo4p9_numpy.py`, `algo6_jax.py`,
`algo6_numpy.py`, `algo7_jax.py`, `algo7_numpy.py`) implement:
> Monte Carlo accounting for approximate GDP, using samples from a Gaussian
> mechanism's PLD

### Example: Running Algorithm 7 (JAX)

To verify approximate $\mu$-GDP parameters using `algo7_jax`:

```bash
python -m mf_opt_over_opt.mca.algo7_jax \
  --mu_input=1.0 \
  --n=100000 \
  --gamma=1e-5 \
  --tau=1e-9 \
  --mu_max=2.0 \
  --mu_step=0.05
```

---

## DP Matrix Factorization (`dpmf/`)

The `dpmf/` directory contains tools to optimize and evaluate differentially
private matrix factorization (DP-MF) noising mechanisms and train models using
them:

- `dpmf/generate_noising_matrix/`: Core library for defining and optimizing
  matrix strategies (including `noisecurve_factorizer`, `optimization`, and
  `matrix_strategy`).
- `dpmf/find_C/`: Optimization and evaluation scripts for finding optimal DP
  noising matrices (including `analytic_obj_evaluation`, `extract_dna`, and
  sandbox evaluators).
- `dpmf/train/`: Model architectures (`bert_model.py`) and training loops
  (`training_loop_bert.py`) supporting DP-MF mechanisms.

### Example 1: Optimizing a DP Noising Matrix

You can optimize a noising matrix using `noisecurve_factorizer`:

```python
import jax.numpy as jnp
import numpy as np
from mf_opt_over_opt.dpmf.generate_noising_matrix.factorizer import noisecurve_factorizer

# 1. Load precomputed preconditioner (e.g. Hessian diagonal)
H = jnp.array(np.load('path/to/flat_precond.npy'))
T = 1000  # Number of iterations
bands = 8  # Band width
lr = 0.01  # Learning rate

# 2. Build the workload matrix A
A = noisecurve_factorizer.build_workload(lr=lr, T=T, H=H)

# 3. Optimize banded strategy parameters
params_star = noisecurve_factorizer.optimize_noisecurve_banded(
    n=T, bands=bands, workload_matrix=A, max_optimizer_steps=250
)

# 4. Save the optimized parameters to disk
np.save('path/to/optimized_matrix.npy', np.array(params_star))
```

Alternatively, you can run the optimization CLI:

```bash
python -m mf_opt_over_opt.dpmf.find_C.analytic_obj_evaluation \
  --matrix_family=banded_opt \
  --method=noisecurve \
  --t=1000 \
  --bands=8 \
  --preconditioner_path=path/to/flat_precond.npy \
  --lr=0.01
```

### Example 2: Training TinyBERT with the Optimized Matrix

Train TinyBERT using `training_loop_bert` with the optimized DP noising matrix:

```bash
python -m mf_opt_over_opt.dpmf.train.training_loop_bert \
  --matrix_family=banded_opt \
  --coef_path=path/to/optimized_matrix.npy \
  --min_sep=8 \
  --iterations=1000 \
  --batch_size=512 \
  --learning_rate=0.01 \
  --noise_multiplier=0.5 \
  --batch_l2_norm_clip=1.0 \
  --dataset_path=path/to/arxiv_splits/train-tf \
  --valid_dataset_path=path/to/arxiv_splits/valid-tf \
  --test_dataset_path=path/to/arxiv_splits/test-tf \
  --initial_checkpoint_path=path/to/model_checkpoints/tiny_bert.npz \
  --root_output_dir=path/to/output_dir \
  --run_name=tiny_bert_dp_banded_opt
```