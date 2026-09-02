# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analytic objective evaluation for banded X matrix optimization."""

# pylint: disable=invalid-name
import os
from typing import Sequence

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from jax.scipy.sparse.linalg import cg
import numpy as np


_MATRIX_FAMILY = flags.DEFINE_enum(
    'matrix_family',
    'banded_opt',
    ['blt', 'banded_opt'],
    'Matrix family to use (only banded_opt is supported in this script).',
)

_METHOD = flags.DEFINE_enum(
    'method',
    'noisecurve',
    ['noisecurve', 'rmse'],
    'Which evaluation to run.',
)
_T = flags.DEFINE_integer('t', 1000, 'Number of iterations.')
_PRECONDITIONER_PATH = flags.DEFINE_string(
    'preconditioner_path',
    '/path/to/checkpoints/flat_precond.npy',
    'Path to the preconditioner file (Hessian diagonal flat_precond.npy).',
)
_LR = flags.DEFINE_float('lr', 0.01, 'Learning rate (for noisecurve workload).')
_OBJECTIVE = flags.DEFINE_enum(
    'objective',
    'avg',
    ['final', 'avg'],
    'Objective to minimize (final loss or average loss).',
)

# Banded Parameters
_BANDS = flags.DEFINE_integer(
    'bands', 8, 'Number of bands for matrix strategy.'
)
_SOLVE_PARAMS_STEPS = flags.DEFINE_integer(
    'solve_params_steps',
    5,
    'Number of outer optimizer steps to solve for strategy parameters'
    ' (compatibility).',
)
_INIT_PARAMS_SEED = flags.DEFINE_integer(
    'init_params_seed',
    42,
    'PRNG seed for strategy parameter initialization (compatibility).',
)
_WORK_DIR = flags.DEFINE_string(
    'work_dir', None, 'Optional directory to save optimized parameters.'
)


# 1. Force 64-bit precision (Essential for Newton method stability)
jax.config.update('jax_enable_x64', True)


def build_banded_optimizer(n, b):
  """Creates JIT-compiled optimization primitives for the banded matrix space.

  Using a closure statically bakes the shapes (n, b) into XLA compilation.

  Args:
    n: Matrix dimension.
    b: Band width.

  Returns:
    A tuple of (num_vars, mat, evaluate_objective, newton_cg_step).
  """
  # Precompute indices for strictly upper b-band elements
  i_grid = jnp.arange(n)[:, None]
  j_grid = i_grid + 1 + jnp.arange(b)[None, :]
  mask = j_grid < n
  rows = jnp.broadcast_to(i_grid, (n, b))[mask]
  cols = j_grid[mask]
  num_vars = len(rows)

  @jax.jit
  def mat(x):
    """Maps vector x into a symmetric n x n matrix with 0 on the diagonal."""
    Z = jnp.zeros((n, n), dtype=jnp.float64)
    Z = Z.at[rows, cols].set(x)
    Z = Z.at[cols, rows].set(x)
    return Z

  @jax.jit
  def vec(A):
    """Extracts strictly upper b-band elements from matrix A."""
    return A[rows, cols]

  @jax.jit
  def evaluate_objective(x, t, W):
    """Evaluates F_t(x) = t * Tr(W X^-1) - log(det(X)).

    Safely catches out-of-domain (non-PD) matrices during line-search by
    returning jnp.inf without crashing the XLA compiler.
    """
    X = jnp.eye(n) + mat(x)
    L = jnp.linalg.cholesky(X)
    is_pd = ~jnp.isnan(L).any()

    # Safe fallback L to prevent NaN errors during JIT evaluation if not PD
    L_safe = jnp.where(is_pd, L, jnp.eye(n))
    C_safe = cho_solve((L_safe, True), jnp.eye(n))

    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L_safe)))

    # O(n^2) trick: Tr(W X^-1) is exactly the dot product of W and C
    # (since both are real symmetric matrices), vastly faster than W @ C.
    trace_term = jnp.sum(W * C_safe)

    obj_val = t * trace_term - logdet
    return jnp.where(is_pd, obj_val, jnp.inf)

  @jax.jit
  def newton_cg_step(x, t, W):
    """Computes exact gradient and applies the analytical inverse Hessian via CG."""
    X = jnp.eye(n) + mat(x)
    L = jnp.linalg.cholesky(X)

    # We assume x is PD here (guaranteed by line search), but add safety for JIT
    L_safe = jnp.where(jnp.isnan(L), jnp.eye(n), L)
    C = cho_solve((L_safe, True), jnp.eye(n))

    # Enforce exact symmetry (prevents tiny asymmetric float drift)
    C = (C + C.T) / 2.0
    D = C @ W @ C
    D = (D + D.T) / 2.0

    M = t * D + 0.5 * C
    M = (M + M.T) / 2.0

    # 1. Exact closed-form Vector Gradient (g)
    G_mat = -t * D - C
    g = 2.0 * vec(G_mat)

    # 2. Exact closed-form Hessian-Vector Product Closure
    def hvp(v):
      V = mat(v)
      # Analytical Hessian: MVC + CVM.
      # Optimization: since (MVC).T == CVM, we compute MVC once!
      MVC = M @ (V @ C)
      H_mat = MVC + MVC.T
      return 2.0 * vec(H_mat)

    # 3. Truncated Newton CG tolerance: adapts dynamically based on
    # gradient norm.
    g_norm = jnp.linalg.norm(g)
    cg_tol = jnp.clip(jnp.sqrt(g_norm), 1e-9, 0.5)

    # Solve exact step natively: H * dx = -g
    dx, _ = cg(hvp, -g, tol=cg_tol, maxiter=min(num_vars, 250))

    # 4. Newton decrement: lambda^2 = -g^T dx
    lambda_sq = jnp.maximum(0.0, jnp.dot(-g, dx))
    return dx, lambda_sq, g

  return num_vars, mat, evaluate_objective, newton_cg_step


def solve_banded_optimization(W, b, t0=1.0, mu=10.0, eps_gap=1e-6):
  """Main Outer Log-Barrier Loop and Inner Newton-CG Loop.

  Control logic is kept in Python to allow easy monitoring and dynamic
  backtracking. Because the mathematical kernels are JITted above, Python
  overhead is negligible.

  Args:
    W: Workload matrix.
    b: Band width.
    t0: Initial barrier parameter.
    mu: Barrier parameter multiplier.
    eps_gap: Duality gap tolerance.

  Returns:
    A tuple of (x, C, best_loss, total_inner).
  """
  n = W.shape[0]
  num_vars, mat_fn, obj_fn, newton_fn = build_banded_optimizer(n, b)

  # Initialization: x = 0 strictly maps to X = Identity Matrix.
  x = jnp.zeros(num_vars, dtype=jnp.float64)
  t = float(t0)

  print(f"{'Outer':<6} | {'Inner':<6} | {'Gap (n/t)':<10} | {'Objective'}")
  print('-' * 45)

  outer_iters = 0
  while (n / t) > eps_gap:
    inner_iters = 0
    f_x = obj_fn(x, t, W)

    while True:
      # 1. Evaluate exact Newton direction via JAX CG
      dx, lambda_sq, g = newton_fn(x, t, W)

      # 2. Local Convergence Check for this barrier step
      if lambda_sq / 2.0 <= 1e-5:
        break

      # 3. Armijo Backtracking Line Search
      alpha = 1.0
      g_dot_dx = jnp.dot(g, dx)

      for _ in range(30):
        x_new = x + alpha * dx
        f_new = obj_fn(x_new, t, W)

        # Check for PD validity (!isinf) AND Armijo sufficient decrease
        if not jnp.isinf(f_new) and f_new <= f_x + 0.01 * alpha * g_dot_dx:
          x = x_new
          f_x = f_new
          break
        alpha *= 0.5
      else:
        print('Warning: Line search failed to find valid step.')
        break

      inner_iters += 1
      if inner_iters >= 100:
        break

    print(f'{outer_iters:<6} | {inner_iters:<6} | {n/t:<10.3e} | {f_x:.5f}')

    # 4. Barrier update step
    t *= mu
    outer_iters += 1

  # Reconstruct and return the optimal dense matrix
  return jnp.eye(n) + mat_fn(x)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _MATRIX_FAMILY.value != 'banded_opt':
    raise ValueError(
        f'Only banded_opt is supported, got {_MATRIX_FAMILY.value}'
    )

  T = _T.value
  b = _BANDS.value - 1

  # Load H
  with open(_PRECONDITIONER_PATH.value, 'rb') as f:
    H = np.load(f)

  H = np.sqrt(np.abs(H))
  H = H / np.max(np.abs(H))
  H = jnp.array(H)

  # Build W
  if _METHOD.value == 'noisecurve':
    # Inline implementation of W
    mus = np.array(H)
    mus = mus / np.max(mus)
    V = np.vander(1.0 - _LR.value * mus, T)
    VM = V * mus[:, None]
    VTM2V = np.dot(VM.T, VM)
    J_col = np.arange(T)[:, None]
    K_row = np.arange(T)[None, :]
    T_mat = (T - np.maximum(J_col, K_row)) / T
    W = T_mat * VTM2V
    W = W / T  # Average
  elif _METHOD.value == 'rmse':
    if _OBJECTIVE.value == 'avg':
      J_col = jnp.arange(T)[:, None]
      K_row = jnp.arange(T)[None, :]
      W = T - jnp.maximum(J_col, K_row)
      W = W / T  # Average
    elif _OBJECTIVE.value == 'final':
      W = jnp.ones((T, T))
    else:
      raise ValueError(f'Unknown objective: {_OBJECTIVE.value}')
  else:
    raise ValueError(f'Unknown method: {_METHOD.value}')

  W = jnp.array(W)

  print('Optimizing banded X matrix...')
  X_optimal = solve_banded_optimization(W, b)

  # Compute final loss: Tr(W X^{-1})
  L = jnp.linalg.cholesky(X_optimal)
  C_inv = cho_solve((L, True), jnp.eye(T))
  loss = jnp.sum(W * C_inv)

  print(f'Final Analytic Loss ({_METHOD.value}): {loss:.6f}')

  # Save Optimized X and C
  if _WORK_DIR.value is not None:
    save_dir = _WORK_DIR.value
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    # Compute C via Reverse Cholesky
    X_rev = X_optimal[::-1, ::-1]
    L_rev = jnp.linalg.cholesky(X_rev)
    C_optimal = L_rev.T[::-1, ::-1]

    optimized_X_path = os.path.join(save_dir, 'optimized_X.npy')
    with open(optimized_X_path, 'wb') as f:
      np.save(f, np.array(X_optimal))
    print(f'Saved optimized X to {optimized_X_path}')

    optimized_C_path = os.path.join(save_dir, 'optimized_C.npy')
    with open(optimized_C_path, 'wb') as f:
      np.save(f, np.array(C_optimal))
    print(f'Saved optimized C to {optimized_C_path}')


if __name__ == '__main__':
  app.run(main)
