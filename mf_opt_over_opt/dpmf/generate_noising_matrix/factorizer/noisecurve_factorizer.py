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

"""Noise curve factorizer helper."""

# pylint: disable=invalid-name
from collections.abc import Callable
import functools

import chex
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import banded
from jax_privacy.matrix_factorization import toeplitz
import numpy as np
import optax

from mf_opt_over_opt.dpmf.generate_noising_matrix import optimization


def build_workload(lr, T, H):
  """Builds the workload matrix A for NoiseCurve using NumPy on CPU.

  Args:
    lr: The learning rate (eta).
    T: The number of iterations.
    H: The Hessian diagonal of shape (d,) (JAX array).

  Returns:
    A lower triangular NumPy array A of shape (T, T) such that A^T A = W,
    where W is the workload Gram matrix defined in Theorem 4.1 Eq 2.
  """
  # Force conversion to CPU NumPy array
  mus = np.array(H)
  mus = mus / np.max(mus)

  V = np.vander(1.0 - lr * mus, T)  # Shape (p, T)

  # V^T M^2 V where M = Diag(mus)
  VM = V * mus[:, None]  # Shape (p, T)
  VTM2V = np.dot(VM.T, VM)  # Shape (T, T)

  # T_mat[j, k] = (T - max(j, k)) / T
  J_col = np.arange(T)[:, None]
  K_row = np.arange(T)[None, :]
  T_mat = (T - np.maximum(J_col, K_row)) / T

  # W = T_mat * (V^T M^2 V)
  W = T_mat * VTM2V

  # Reverse Cholesky to get lower triangular A such that A^T A = W
  W_rev = W[::-1, ::-1]

  # Add a small eps for numerical stability if W is not strictly PD
  eps = 1e-6
  W_rev = W_rev + eps * np.eye(T)

  L = np.linalg.cholesky(W_rev)
  A = L.T[::-1, ::-1]

  return A


def loss(
    strategy_coef,
    n = None,
    reduction_fn = jnp.mean,
    workload_matrix = None,
):
  """Error of C on workload under single participation."""
  if reduction_fn is jnp.max:
    reduction_fn = lambda v: v[-1]
  strategy_coef, n = toeplitz._reconcile(strategy_coef, n)  # pylint: disable=protected-access

  if workload_matrix is not None:
    workload_matrix = jnp.array(workload_matrix)
    chex.assert_shape(workload_matrix, (n, n))

    # vmap over columns of workload_matrix (in_axes=1)
    pq_err_fn = lambda col: toeplitz.per_query_error(
        strategy_coef=strategy_coef, n=n, workload_coef=col
    )
    S = jax.vmap(pq_err_fn, in_axes=1, out_axes=1)(workload_matrix)

    # S[k, i] = \sum_{j=0}^k (B'_{j, i})^2
    # Sum over columns to get cumulative row norms: S[k] = \sum_i S[k, i]
    cum_row_norms = jnp.sum(S, axis=1)

    # Recover row norms: R_k = S[k] - S[k-1]
    row_norms = jnp.diff(cum_row_norms, prepend=0.0)

    error = reduction_fn(row_norms)
  else:
    error = reduction_fn(
        toeplitz.per_query_error(strategy_coef=strategy_coef, n=n)
    )

  sens_squared = toeplitz.sensitivity_squared(strategy_coef, n)
  return error * sens_squared


def loss_banded(
    params,
    n,
    workload_matrix,
    reduction_fn = jnp.mean,
):
  """Error of C on workload for general banded matrix (column normalized)."""
  if reduction_fn is jnp.max:
    reduction_fn = lambda v: v[-1]

  C = banded.ColumnNormalizedBanded(params=params)

  workload_matrix = jnp.array(workload_matrix)
  chex.assert_shape(workload_matrix, (n, n))

  C_dense = C.materialize()
  # Solve B C = A => C^T B^T = A^T
  B_T = jax.scipy.linalg.solve_triangular(
      C_dense.T, workload_matrix.T, lower=False
  )
  B = B_T.T
  row_norms = jnp.sum(B**2, axis=1)
  error = reduction_fn(row_norms)

  return error


def optimize_noisecurve_banded(
    n,
    bands,
    workload_matrix,
    params = None,
    max_optimizer_steps = 250,
    reduction_fn = jnp.mean,
    optimizer = optimization.DEFAULT_OPTIMIZER,
):
  """Optimize over the space of banded strategies on a workload."""
  loss_fn = functools.partial(
      loss_banded,
      n=n,
      reduction_fn=reduction_fn,
      workload_matrix=workload_matrix,
  )

  if params is None:
    C_default = banded.ColumnNormalizedBanded.default(n, bands)
    params = C_default.params

  loss_and_grad_fn = jax.value_and_grad(loss_fn)

  optimized_params = optimization.optimize(
      loss_and_grad=loss_and_grad_fn,
      params=params,
      max_optimizer_steps=max_optimizer_steps,
      value_fn=loss_fn,
      optimizer=optimizer,
  )
  return optimized_params


def optimize_noisecurve_banded_toeplitz(
    n,
    bands,
    strategy_coef = None,
    max_optimizer_steps = 250,
    reduction_fn = jnp.mean,
    workload_matrix = None,
    optimizer = optimization.DEFAULT_OPTIMIZER,
):
  """Optimize over the space of banded Toeplitz strategies on a workload."""
  loss_fn = functools.partial(
      loss, n=n, reduction_fn=reduction_fn, workload_matrix=workload_matrix
  )

  if strategy_coef is None:
    strategy_coef = toeplitz.optimal_max_error_strategy_coefs(bands)
  if strategy_coef.shape[0] != bands:
    raise ValueError(f'{strategy_coef.shape=} != {bands=}')

  loss_and_grad_fn = jax.value_and_grad(loss_fn)

  params = optimization.optimize(
      loss_and_grad=loss_and_grad_fn,
      params=strategy_coef,
      max_optimizer_steps=max_optimizer_steps,
      value_fn=loss_fn,
      optimizer=optimizer,
  )
  return params / jnp.linalg.norm(params)


def optimize_noisecurve_single_param_toeplitz(
    n,
    bands,
    init_c = 0.5,
    max_optimizer_steps = 250,
    reduction_fn = jnp.mean,
    workload_matrix = None,
    optimizer = optimization.DEFAULT_OPTIMIZER,
):
  """Optimize over a single parameter Toeplitz strategy (c^k) on a workload."""

  def single_param_loss(c):
    strategy_coef = jnp.power(c, jnp.arange(bands))
    return loss(
        strategy_coef=strategy_coef,
        n=n,
        reduction_fn=reduction_fn,
        workload_matrix=workload_matrix,
    )

  loss_and_grad_fn = jax.value_and_grad(single_param_loss)

  opt_c = optimization.optimize(
      loss_and_grad=loss_and_grad_fn,
      params=jnp.array(init_c),
      max_optimizer_steps=max_optimizer_steps,
      value_fn=single_param_loss,
      optimizer=optimizer,
  )
  return opt_c
