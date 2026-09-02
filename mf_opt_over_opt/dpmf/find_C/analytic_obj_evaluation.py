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

"""Analytic objective evaluation for DP noising matrix strategies."""

# pylint: disable=invalid-name
import dataclasses
import functools
import os
from typing import Any, Callable, Sequence, TypeAlias

from absl import app
from absl import flags
import chex
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import streaming_matrix
from jax_privacy.matrix_factorization import toeplitz
import numpy as np
import optax

from mf_opt_over_opt.dpmf.generate_noising_matrix.factorizer import noisecurve_factorizer

jax.config.update('jax_enable_x64', True)

_MATRIX_FAMILY = flags.DEFINE_enum(
    'matrix_family',
    'blt',
    ['blt', 'banded_opt', 'toeplitz'],
    'Matrix family to use.',
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

# BLT Parameters (assuming 2 buffers for consistency with xm_launch.py)
_BUF_DECAY_0 = flags.DEFINE_float('buf_decay_0', 0.5, 'buf_decay_0')
_BUF_DECAY_1 = flags.DEFINE_float('buf_decay_1', 0.5, 'buf_decay_1')
_OUTPUT_SCALE_0 = flags.DEFINE_float('output_scale_0', 0.5, 'output_scale_0')
_OUTPUT_SCALE_1 = flags.DEFINE_float('output_scale_1', 0.5, 'output_scale_1')

# Banded Parameters
_BANDS = flags.DEFINE_integer(
    'bands', 8, 'Number of bands for matrix strategy (for banded_opt).'
)
_SOLVE_PARAMS_STEPS = flags.DEFINE_integer(
    'solve_params_steps',
    5,
    'Number of outer optimizer steps to solve for strategy parameters.',
)
_INIT_PARAMS_SEED = flags.DEFINE_integer(
    'init_params_seed', 42, 'PRNG seed for strategy parameter initialization.'
)

_MAX_LOSS_THRESHOLD = flags.DEFINE_float(
    'max_loss_threshold',
    10000.0,
    'Stop optimization early if loss exceeds this threshold.',
)
_WORK_DIR = flags.DEFINE_string(
    'work_dir', None, 'Optional directory to save optimized parameters.'
)

_DEBUG = flags.DEFINE_bool('debug', False, 'Enable debug logging.')


def compute_blt_C_coefs(buf_decay, output_scale, T):
  powers = jnp.arange(T - 1)
  tmp = buf_decay ** powers[:, None] * output_scale
  return jnp.concatenate([
      jnp.array([1.0], dtype=buf_decay.dtype),
      jnp.sum(tmp, axis=1),
  ])


def compute_blt_analytic_loss(buf_decay, output_scale, A, T, objective='avg'):
  """Computes analytic loss for BLT strategy."""
  C_coefs = compute_blt_C_coefs(buf_decay, output_scale, T)
  col_norms = jnp.sqrt(jnp.cumsum(C_coefs**2))[::-1]

  Cinv_coefs = toeplitz.inverse_coef(C_coefs, n=T)

  A_scaled = A * col_norms[None, :]
  Cinv_unnorm_mat = toeplitz.materialize_lower_triangular(Cinv_coefs, n=T)
  B = A_scaled @ Cinv_unnorm_mat
  row_norms = jnp.sum(B**2, axis=1)

  if objective == 'final':
    return row_norms[-1]
  elif objective == 'avg':
    return jnp.mean(row_norms)
  else:
    raise ValueError(f'Unknown objective: {objective}')


@functools.partial(jax.jit, static_argnames=['b', 'T', 'alpha'])
def generate_C_banded_opt(params, b, T, alpha=0.99):
  """Maps unconstrained params to a strictly valid banded matrix."""
  col_indices = jnp.expand_dims(jnp.arange(T), 1)
  sub_indices = jnp.expand_dims(jnp.arange(1, b + 1), 0)
  mask = (col_indices + sub_indices) < T
  V = params * mask.astype(jnp.float32)

  L1 = jnp.sum(jnp.abs(V), axis=1)
  L2_sq = jnp.sum(V**2, axis=1)
  S = L1**2 + L2_sq

  S_safe = jnp.where(S > 1e-12, S, 1.0)
  N = jnp.where(S > 1e-12, jnp.sqrt(S_safe), 0.0)

  N_safe = jnp.where(N < 1e-6, 1.0, N)
  scale = alpha * jnp.where(N < 1e-6, jnp.ones_like(N), jnp.tanh(N) / N_safe)
  U = V * jnp.expand_dims(scale, 1)

  U_sq = jnp.sum(U**2, axis=1)
  c_diag = jnp.sqrt(jnp.maximum(1.0 - U_sq, 1e-12))

  C = jnp.diag(c_diag)
  for i in range(1, b + 1):
    sub_diag_elements = U[: T - i, i - 1]
    C = C + jnp.diag(sub_diag_elements, k=-i)

  return C


def compute_banded_analytic_loss(V, A, b, T, alpha=0.99, objective='avg'):
  """Computes analytic loss for banded strategy."""
  C = generate_C_banded_opt(V, b, T, alpha)
  # B = A @ C^{-1}
  # Solve C^T B^T = A^T
  BT = jax.scipy.linalg.solve_triangular(C, A.T, lower=True, trans='T')
  B = BT.T
  row_norms = jnp.sum(B**2, axis=1)

  if objective == 'final':
    return row_norms[-1]
  elif objective == 'avg':
    return jnp.mean(row_norms)
  else:
    raise ValueError(f'Unknown objective: {objective}')


@functools.partial(jax.jit, static_argnames=['bands', 'T'])
def generate_C_toeplitz(params, bands, T):
  """Maps free params in R^(bands-1) to a uniformly scaled Toeplitz matrix C."""
  a_off_diag = params
  a = jnp.concatenate([jnp.ones((1,), dtype=params.dtype), a_off_diag], axis=0)
  C = jnp.eye(T, dtype=params.dtype)
  for i in range(1, bands):
    C = C + jnp.diag(jnp.full((T - i,), a[i], dtype=params.dtype), k=-i)
  norm_a = jnp.linalg.norm(a, ord=2)
  return C / norm_a


def compute_toeplitz_analytic_loss(
    params,
    A,
    bands,
    T,
    objective='avg',
):
  """Computes analytic loss for Toeplitz strategy."""
  C = generate_C_toeplitz(params, bands, T)
  BT = jax.scipy.linalg.solve_triangular(C, A.T, lower=True, trans='T')
  B = BT.T
  row_norms = jnp.sum(B**2, axis=1)

  if objective == 'final':
    return row_norms[-1]
  elif objective == 'avg':
    return jnp.mean(row_norms)
  else:
    raise ValueError(f'Unknown objective: {objective}')


# Optimization Utilities (L-BFGS)
DEFAULT_OPTIMIZER = optax.lbfgs(
    memory_size=1, linesearch=optax.scale_by_backtracking_linesearch(128)
)


@dataclasses.dataclass
class CallbackArgs:
  step: int
  loss: jnp.ndarray
  grad: chex.ArrayTree | None
  params: chex.ArrayTree
  state: Any


CallbackFnType: TypeAlias = Callable[[CallbackArgs], None | bool]


def jax_enable_x64(fn):
  def wrapped_fn(*args, **kwargs):
    with jax.enable_x64():
      return fn(*args, **kwargs)

  return wrapped_fn


@jax_enable_x64
def optimize(
    loss_and_grad,
    params,
    *,
    max_optimizer_steps = 250,
    value_fn = None,
    callback = lambda _: None,
    optimizer = DEFAULT_OPTIMIZER,
    projection_fn = None,
):
  """Optimize a differentiable loss function."""
  if projection_fn is not None and value_fn is not None:
    user_value_fn = value_fn
    value_fn = lambda p: user_value_fn(projection_fn(p))

  def single_step(params, opt_state):
    value, grad = loss_and_grad(params)
    updates, opt_state = optimizer.update(
        grad, opt_state, params, value=value, grad=grad, value_fn=value_fn
    )
    new_params = optax.apply_updates(params, updates)
    if projection_fn is not None:
      new_params = projection_fn(new_params)
    return value, grad, new_params, opt_state

  original_dtypes = jax.tree.map(lambda x: x.dtype, params)

  params = jax.tree.map(jnp.float64, params)
  if projection_fn is not None:
    params = projection_fn(params)
  state = optimizer.init(params)
  for i in range(max_optimizer_steps):
    loss, grad, params, state = single_step(params, state)
    if callback(
        CallbackArgs(step=i, loss=loss, grad=grad, params=params, state=state)
    ):
      break

  return jax.tree.map(jnp.astype, params, original_dtypes)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  T = _T.value

  # Load H
  with open(_PRECONDITIONER_PATH.value, 'rb') as f:
    H = np.load(f)

  H = np.sqrt(np.abs(H))
  H = H / np.max(np.abs(H))
  H = jnp.array(H)

  # Build Workload A
  if _METHOD.value == 'noisecurve':
    A = noisecurve_factorizer.build_workload(lr=_LR.value, T=T, H=H)
  elif _METHOD.value == 'rmse':
    A = streaming_matrix.prefix_sum().materialize(T)
  else:
    raise ValueError(f'Unknown method: {_METHOD.value}')

  A = jnp.array(A)

  # Matrix Family Branching
  if _MATRIX_FAMILY.value == 'blt':
    # BLT Params
    buf_decay = jnp.array([_BUF_DECAY_0.value, _BUF_DECAY_1.value])
    output_scale = jnp.array([_OUTPUT_SCALE_0.value, _OUTPUT_SCALE_1.value])

    # Compute Loss
    loss = compute_blt_analytic_loss(
        buf_decay, output_scale, A, T, objective=_OBJECTIVE.value
    )
    print(f'Analytic Loss ({_METHOD.value}): {loss:.6f}')

  elif _MATRIX_FAMILY.value in ('banded_opt', 'toeplitz'):
    obj_type = 'avg' if _METHOD.value == 'noisecurve' else _OBJECTIVE.value
    if _MATRIX_FAMILY.value == 'banded_opt':
      b = _BANDS.value - 1
      init_V = jnp.zeros((T, b))
      objective_fn = lambda V: compute_banded_analytic_loss(
          V, A, b, T, objective=obj_type
      )
      projection_fn = None
    else:
      init_V = jnp.zeros((_BANDS.value - 1,), dtype=jnp.float64)
      objective_fn = lambda V: compute_toeplitz_analytic_loss(
          V,
          A,
          _BANDS.value,
          T,
          objective=obj_type,
      )
      projection_fn = None

    loss_and_grad_fn = jax.jit(jax.value_and_grad(objective_fn))

    def callback_fn(args):
      grad_norm = optax.global_norm(args.grad)
      print(
          f'Step {args.step:2d}, loss: {args.loss:.9f}, '
          f'grad norm: {grad_norm:.9f}'
      )

      if (
          jnp.isnan(args.loss)
          or jnp.isnan(grad_norm)
          or args.loss > _MAX_LOSS_THRESHOLD.value
      ):
        print(
            f'Loss ({args.loss}) exceeded threshold'
            f' ({_MAX_LOSS_THRESHOLD.value}) or NaN detected! Stopping.'
        )
        return True
      return False

    print(f'Optimizing {_MATRIX_FAMILY.value} matrix parameters...')
    optimized_V = optimize(
        loss_and_grad=loss_and_grad_fn,
        params=init_V,
        max_optimizer_steps=_SOLVE_PARAMS_STEPS.value,
        value_fn=objective_fn,
        callback=callback_fn,
        projection_fn=projection_fn,
    )

    # Compute final loss with optimized V
    loss = objective_fn(optimized_V)
    print(f'Final Analytic Loss ({_METHOD.value}): {loss:.6f}')

    # Save Optimized V
    if _WORK_DIR.value is not None:
      save_dir = _WORK_DIR.value
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      optimized_V_path = os.path.join(save_dir, 'optimized_V.npy')
      with open(optimized_V_path, 'wb') as f:
        np.save(f, np.array(optimized_V))
      print(f'Saved optimized V to {optimized_V_path}')

  else:
    raise ValueError(f'Unknown matrix family: {_MATRIX_FAMILY.value}')


if __name__ == '__main__':
  app.run(main)
