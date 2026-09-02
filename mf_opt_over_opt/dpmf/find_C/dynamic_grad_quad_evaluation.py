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

"""Quadratic proxy training and DP-MF matrix optimization.

Extends fixed_precond_banded_toep_sandbox.py by replacing standard quadratic
gradients with the dynamic AR(1) gradient update rule derived from real
model DNA (mu, sigma, rho, h).
"""

# pylint: disable=invalid-name
import abc
from collections.abc import Callable, Sequence
import dataclasses
import functools
import os
from typing import Any, TypeAlias, TypeVar, cast

from absl import app
from absl import flags
import chex
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import banded
from jax_privacy.matrix_factorization import streaming_matrix
import numpy as np
import optax

jax.config.update('jax_enable_x64', True)


class MatrixStrategy(abc.ABC):
  """Abstract base class for a tunable DP noising matrix strategy."""

  @abc.abstractmethod
  def init_params(self):
    """Returns the initial guess for the parameters to optimize."""
    raise NotImplementedError

  @abc.abstractmethod
  def build_noising_matrix(
      self, params
  ):
    """Constructs the DP scaled C^{-1} matrix given the tunable parameters."""
    raise NotImplementedError

  @abc.abstractmethod
  def build_normalized_noising_matrix(
      self, params
  ):
    """Constructs the DP scaled C^{-1} matrix with sensitivity=1."""
    raise NotImplementedError

  def project_params(self, params):
    """Optionally restricts/projects parameters to a valid constraint manifold."""
    return params

  def project_grad(self, params, grad):
    """Optionally projects Euclidean gradient onto the tangent space."""
    del params
    return grad

  def constraint_fn(self, params):
    """Computes constraint violation h(params) = 0 for ALM."""
    del params
    return jnp.zeros((1,), dtype=jnp.float64)


def memory_friendly_toeplitz_inverse(
    coef,
    scale = None,
):
  """Creates C^{-1} as a memory-efficient StreamingMatrix object.

  Unlike toeplitz.inverse_as_streaming_matrix, this implementation uses a ring
  buffer state with in-place updates (bufs.at[k].set(xi)) rather than jnp.roll.
  This avoids storing full state tensor rolls across scan steps during
  reverse-mode automatic differentiation.

  Args:
    coef: Coefficients for Toeplitz matrix.
    scale: Optional scale array.

  Returns:
    A StreamingMatrix object representing C^{-1}.
  """
  bands = coef.shape[0]

  def init(abstract_yi):
    dtype = jnp.promote_types(abstract_yi.dtype, coef.dtype)
    zero = jnp.zeros_like(abstract_yi, dtype=dtype)
    buffers = jnp.broadcast_to(zero, (bands - 1,) + zero.shape)
    return jnp.array(0), buffers

  def _next(yi, state):
    if bands == 1:
      res = yi / coef[0]
      if scale is not None:
        res = res * scale
      return res, state

    index, bufs = state
    r = jnp.arange(1, bands)
    slots = (index - r) % (bands - 1)
    inner = jnp.tensordot(coef[1:], bufs[slots], axes=((0,), (0,)))
    xi = (yi - inner) / coef[0]

    k = index % (bands - 1)
    updated_bufs = bufs.at[k].set(xi)

    output = xi if scale is None else xi * scale
    return output, (index + 1, updated_bufs)

  return streaming_matrix.StreamingMatrix.from_array_implementation(init, _next)


class BandedToeplitzOptimizationStrategy(MatrixStrategy):
  """A strategy using Banded Toeplitz Optimization (supporting Projected & ALM)."""

  def __init__(
      self,
      T,
      bands,
      init_c = None,
      unconstrained = False,
  ):
    self.T = T
    self.bands = bands
    self.unconstrained = unconstrained
    if init_c is None:
      self.init_c = jnp.zeros((bands,), dtype=jnp.float64).at[0].set(1.0)
    else:
      self.init_c = init_c

  def init_params(self):
    if self.unconstrained:
      return self.init_c
    return self.project_params(self.init_c)

  @staticmethod
  @functools.partial(jax.jit, static_argnames=['bands'])
  def generate_C_bands(params, bands):
    del bands
    norm = jnp.linalg.norm(params, ord=2)
    norm_safe = jnp.where(norm < 1e-12, 1.0, norm)
    c = params / norm_safe
    c = jnp.where(c[0] < 0, -c, c)
    return c, jnp.linalg.norm(c, ord=2)

  def project_params(self, params):
    """Projects parameters onto the unit sphere S^{bands-1} with c_0 >= 0."""
    c, _ = self.generate_C_bands(params, self.bands)
    return c

  def project_grad(self, params, grad):
    """Projects Euclidean gradient onto the tangent space of S^{bands-1}."""
    c = self.project_params(params)
    inner = jnp.dot(c, grad)
    return grad - inner * c

  def constraint_fn(self, params):
    """Constraint violation ||c||_2^2 - 1 = 0 for ALM."""
    return jnp.sum(params**2) - 1.0

  def build_noising_matrix(
      self, params
  ):
    if self.unconstrained:
      c0 = jnp.maximum(params[0], 1e-6)
      c = jnp.concatenate([jnp.array([c0], dtype=params.dtype), params[1:]])
      return memory_friendly_toeplitz_inverse(c)

    c = self.project_params(params)
    cinv = memory_friendly_toeplitz_inverse(c)
    if self.T <= self.bands:
      full_coef = c[: self.T]
    else:
      full_coef = jnp.pad(c, (0, self.T - self.bands))
    col_norms = jnp.sqrt(jnp.cumsum(full_coef**2))[::-1]
    return streaming_matrix.scale_rows_and_columns(cinv, row_scale=col_norms)

  def build_normalized_noising_matrix(
      self, params
  ):
    c = self.project_params(params)
    cinv = memory_friendly_toeplitz_inverse(c)
    if self.T <= self.bands:
      full_coef = c[: self.T]
    else:
      full_coef = jnp.pad(c, (0, self.T - self.bands))
    col_norms = jnp.sqrt(jnp.cumsum(full_coef**2))[::-1]
    return streaming_matrix.scale_rows_and_columns(cinv, row_scale=col_norms)


class BandedOptimizationStrategy(MatrixStrategy):
  """A strategy using Banded Optimization from banded_optimization.py."""

  def __init__(
      self,
      T,
      bands,
      alpha = 0.99,
      init_V = None,
  ):
    """Initializes the strategy.

    Args:
      T: Dimension of the matrix.
      bands: Number of bands (including diagonal).
      alpha: Squashing parameter.
      init_V: Initial value for the unconstrained parameters (V) of shape (T,
        bands - 1).
    """
    self.T = T
    self.bands = bands
    self.alpha = alpha
    if init_V is None:
      self.init_V = jnp.zeros((T, bands - 1))
    else:
      self.init_V = init_V

  def init_params(self):
    return self.init_V

  @staticmethod
  @functools.partial(jax.jit, static_argnames=['b', 'T', 'alpha'])
  def generate_C_bands(params, b, T, alpha):
    """Maps unconstrained params to strictly valid banded matrix parameters.

    Args:
        params: Unconstrained JAX array (V) of shape (T, b)
        b: Number of sub-diagonals (integer)
        T: Dimension of the matrix (integer)
        alpha: Squashing parameter

    Returns:
        Params array of shape (T, b + 1) representing the banded matrix.
    """
    col_indices = jnp.expand_dims(jnp.arange(T), 1)
    sub_indices = jnp.expand_dims(jnp.arange(1, b + 1), 0)
    mask = (col_indices + sub_indices) < T
    V = params * mask.astype(jnp.float32)

    L1 = jnp.sum(jnp.abs(V), axis=1)
    L2_sq = jnp.sum(V**2, axis=1)
    S = L1**2 + L2_sq

    # Safe sqrt for N
    S_safe = jnp.where(S > 1e-12, S, 1.0)
    N = jnp.where(S > 1e-12, jnp.sqrt(S_safe), 0.0)

    # Safe division for scale
    N_safe = jnp.where(N < 1e-6, 1.0, N)
    scale = alpha * jnp.where(N < 1e-6, jnp.ones_like(N), jnp.tanh(N) / N_safe)
    U = V * jnp.expand_dims(scale, 1)

    # Direct sum of squares instead of norm**2 to avoid NaN gradient at 0
    U_sq = jnp.sum(U**2, axis=1)
    c_diag = jnp.sqrt(jnp.maximum(1.0 - U_sq, 1e-12))

    return jnp.concatenate([jnp.expand_dims(c_diag, 1), U], axis=1)

  def build_noising_matrix(
      self, params
  ):
    b = self.bands - 1
    extracted_bands = self.generate_C_bands(params, b, self.T, self.alpha)
    strategy = banded.ColumnNormalizedBanded(params=extracted_bands)
    return strategy.inverse_as_streaming_matrix()

  def build_normalized_noising_matrix(
      self, params
  ):
    return self.build_noising_matrix(params)


def loss_fn(theta, h, mu):
  quad = (4_400_000 / 5000) * 0.5 * jnp.dot(theta, h * theta)
  return quad + jnp.dot(mu, theta)


def optim(
    noising_matrix,
    w0,
    key,
    T,
    stddev,
    optimizer,
    mu,
    sigma,
    rho,
    h,
    l2_norm_clip = 1.0,
    use_checkpoint = False,
    objective = 'final',
    max_loss_threshold = 10000.0,
):
  """Runs DP optimizer on quadratic proxy using model DNA."""
  if mu.ndim == 1:
    mu = jnp.expand_dims(mu, 0)
    sigma = jnp.expand_dims(sigma, 0)
    rho = jnp.expand_dims(rho, 0)
    h = jnp.expand_dims(h, 0)
  M = mu.shape[0]
  d = w0.shape[0]
  params = {'q': w0}

  opt_state = optimizer.init(params)
  matrix_state = noising_matrix.init_multiply(jnp.zeros(d, dtype=w0.dtype))
  z_init = jnp.zeros(d, dtype=w0.dtype)

  init_loss = loss_fn(w0, h[0], mu[0])
  diverged_init = jnp.isnan(init_loss) | (init_loss > max_loss_threshold)

  @jax.jit
  def step_fn(carry, i):
    params_state, opt_state_val, key_t, matrix_state_val, z_prev, diverged = (
        carry
    )
    key_t, subkey_eps, subkey_noise = jax.random.split(key_t, 3)

    k = jnp.minimum((i * M) // T, M - 1)
    h_k = h[k]
    mu_k = mu[k]
    rho_k = rho[k]
    sigma_k = sigma[k]

    noise_scale_k = sigma_k * jnp.sqrt(jnp.maximum(1.0 - rho_k**2, 0.0))

    # 1. AR(1) Noise Process for gradient signal fluctuation
    eps = jax.random.normal(subkey_eps, shape=(d,), dtype=w0.dtype)
    epsilon_t = eps * noise_scale_k
    z_current = rho_k * z_prev + epsilon_t
    s_current = mu_k + z_current

    # 2. Simulated gradient: g_t = (4_400_000 / 5000) * h_k * theta + s_t
    theta = params_state['q']
    g_current = (4_400_000 / 5000) * h_k * theta + s_current
    grads = {'q': g_current}

    # 3. DP Clipping and DP-MF Noise Addition
    g_norm = optax.global_norm(grads) * jnp.sqrt(4_400_000 / 5000)
    divisor = jnp.maximum(g_norm / l2_norm_clip, 1.0)
    grads = jax.tree.map(lambda t: t / divisor, grads)

    Z_t = jax.random.normal(subkey_noise, shape=(d,), dtype=w0.dtype)
    E_t_raw, next_matrix_state = noising_matrix.multiply_next(
        Z_t, matrix_state_val
    )
    E_t = jax.tree.map(lambda e: (stddev * e).astype(w0.dtype), E_t_raw)

    noisy_grads = jax.tree.map(lambda g: g + E_t, grads)

    # Sanitize gradients to prevent NaN/Inf contamination in optimizer state
    noisy_grads_bad = jax.tree_util.tree_reduce(
        lambda acc, x: acc | jnp.isnan(x).any() | jnp.isinf(x).any(),
        noisy_grads,
        False,
    )
    safe_noisy_grads = jax.tree.map(
        lambda g: jnp.where(diverged | noisy_grads_bad, jnp.zeros_like(g), g),
        noisy_grads,
    )

    # 4. Proposed Optimizer update
    updates, next_opt_state = optimizer.update(
        safe_noisy_grads, opt_state_val, params_state
    )
    next_params = optax.apply_updates(params_state, updates)
    next_theta = cast(dict[str, jax.Array], next_params)['q']

    # Evaluate loss on proposed parameters
    next_loss = loss_fn(next_theta, h_k, mu_k)
    is_now_bad = (
        jnp.isnan(next_loss)
        | (next_loss > max_loss_threshold)
        | noisy_grads_bad
    )
    is_diverged = diverged | is_now_bad

    # If diverged (previously or now), freeze state and emit max_loss_threshold
    params_state = jax.tree.map(
        lambda old, new: jnp.where(is_diverged, old, new),
        params_state,
        next_params,
    )
    opt_state_val = jax.tree.map(
        lambda old, new: jnp.where(is_diverged, old, new),
        opt_state_val,
        next_opt_state,
    )
    matrix_state_val = jax.tree.map(
        lambda old, new: jnp.where(is_diverged, old, new),
        matrix_state_val,
        next_matrix_state,
    )
    loss_val = jnp.where(is_diverged, max_loss_threshold, next_loss)

    return (
        params_state,
        opt_state_val,
        key_t,
        matrix_state_val,
        z_current,
        is_diverged,
    ), loss_val

  init_carry = (
      params,
      opt_state,
      key,
      matrix_state,
      z_init,
      diverged_init,
  )

  if use_checkpoint:
    import equinox as eqx  # pylint: disable=g-import-not-at-top, import-outside-toplevel
    (final_params, _, _, _, _, diverged_final), step_losses = (
        eqx.internal.scan(
            step_fn,
            init_carry,
            xs=jnp.arange(T),
            kind='checkpointed',
            checkpoints=4,
        )
    )
  else:
    (final_params, _, _, _, _, diverged_final), step_losses = (
        jax.lax.scan(
            step_fn,
            init_carry,
            xs=jnp.arange(T),
        )
    )

  if objective == 'final':
    final_theta = final_params['q']
    final_loss = loss_fn(final_theta, h[-1], mu[-1])
    diverged = (
        diverged_final
        | jnp.isnan(final_loss)
        | (final_loss > max_loss_threshold)
    )
    return jnp.where(
        diverged,
        max_loss_threshold,
        final_loss,
    )
  elif objective == 'avg':
    initial_loss = loss_fn(w0, h[0], mu[0])
    initial_loss = jnp.where(
        jnp.isnan(initial_loss) | (initial_loss > max_loss_threshold),
        max_loss_threshold,
        initial_loss,
    )
    all_losses = jnp.concatenate([jnp.array([initial_loss]), step_losses])
    avg_loss = jnp.mean(all_losses)
    return jnp.where(
        jnp.isnan(avg_loss) | (avg_loss > max_loss_threshold),
        max_loss_threshold,
        avg_loss,
    )
  else:
    raise ValueError(f'Unknown objective: {objective}')


def compute_objective_f(
    noising_matrix,
    w0,
    keys,
    T,
    mu,
    sigma,
    rho,
    h,
    stddev = 0.1,
    lr = 0.01,
    optimizer_factory = optax.sgd,
    microbatch_size = 100,
    l2_norm_clip = 1.0,
    use_checkpoint = False,
    objective = 'final',
    max_loss_threshold = 10000.0,
):
  """Computes the expected final loss f over k independent noise realizations."""
  optimizer = optimizer_factory(lr)

  optim_fn = lambda key_i: optim(
      noising_matrix=noising_matrix,
      w0=w0,
      key=key_i,
      T=T,
      stddev=stddev,
      optimizer=optimizer,
      mu=mu,
      sigma=sigma,
      rho=rho,
      h=h,
      l2_norm_clip=l2_norm_clip,
      use_checkpoint=use_checkpoint,
      objective=objective,
      max_loss_threshold=max_loss_threshold,
  )

  devices = jax.devices()
  num_devices = len(devices)
  num_keys = len(keys)

  if num_devices > 1 and num_keys % num_devices == 0:
    mesh = jax.sharding.Mesh(devices, ('x',))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('x', None)
    )
    sharded_keys = jax.device_put(keys, sharding)
    global_microbatch_size = microbatch_size * num_devices
  else:
    sharded_keys = keys
    global_microbatch_size = microbatch_size

  mean_loss = optax.microbatching.micro_vmap(
      optim_fn,
      accumulator=optax.microbatching.AccumulationType.MEAN,
      microbatch_size=min(global_microbatch_size, num_keys),
  )(sharded_keys)
  return mean_loss


def compute_strategy_loss(
    strategy,
    params,
    lr,
    w0,
    keys,
    T,
    mu,
    sigma,
    rho,
    h,
    stddev,
    optimizer_factory = optax.sgd,
    microbatch_size = 100,
    l2_norm_clip = 1.0,
    use_checkpoint = False,
    objective = 'final',
    max_loss_threshold = 10000.0,
):
  """Computes expected final loss over k independent noise realizations."""

  def optim_fn(params_val, key_i):
    optimizer = optimizer_factory(lr)
    matrix = strategy.build_noising_matrix(params_val)
    return optim(
        noising_matrix=matrix,
        w0=w0,
        key=key_i,
        T=T,
        stddev=stddev,
        optimizer=optimizer,
        mu=mu,
        sigma=sigma,
        rho=rho,
        h=h,
        l2_norm_clip=l2_norm_clip,
        use_checkpoint=use_checkpoint,
        objective=objective,
        max_loss_threshold=max_loss_threshold,
    )

  devices = jax.devices()
  num_devices = len(devices)
  num_keys = len(keys)

  if num_devices > 1 and num_keys % num_devices == 0:
    mesh = jax.sharding.Mesh(devices, ('x',))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('x', None)
    )
    sharded_keys = jax.device_put(keys, sharding)
    global_microbatch_size = microbatch_size * num_devices
  else:
    sharded_keys = keys
    global_microbatch_size = microbatch_size

  mean_loss = optax.microbatching.micro_vmap(
      optim_fn,
      in_axes=(None, 0),  # type: ignore[arg-type]
      accumulator=optax.microbatching.AccumulationType.MEAN,
      microbatch_size=min(global_microbatch_size, num_keys),
  )(params, sharded_keys)
  return mean_loss


def compute_strategy_loss_and_grad(
    strategy,
    params,
    lr,
    w0,
    keys,
    T,
    mu,
    sigma,
    rho,
    h,
    stddev,
    optimizer_factory = optax.sgd,
    microbatch_size = 100,
    l2_norm_clip = 1.0,
    use_checkpoint = False,
    objective = 'final',
    max_loss_threshold = 10000.0,
):
  """Computes expected final loss and its gradient wrt strategy params."""

  def optim_fn(params_val, key_i):
    optimizer = optimizer_factory(lr)
    matrix = strategy.build_noising_matrix(params_val)
    return optim(
        noising_matrix=matrix,
        w0=w0,
        key=key_i,
        T=T,
        stddev=stddev,
        optimizer=optimizer,
        mu=mu,
        sigma=sigma,
        rho=rho,
        h=h,
        l2_norm_clip=l2_norm_clip,
        use_checkpoint=use_checkpoint,
        objective=objective,
        max_loss_threshold=max_loss_threshold,
    )

  single_loss_and_grad = jax.value_and_grad(optim_fn, argnums=0)

  devices = jax.devices()
  num_devices = len(devices)
  num_keys = len(keys)

  if num_devices > 1 and num_keys % num_devices == 0:
    mesh = jax.sharding.Mesh(devices, ('x',))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('x', None)
    )
    sharded_keys = jax.device_put(keys, sharding)
    global_microbatch_size = microbatch_size * num_devices
  else:
    sharded_keys = keys
    global_microbatch_size = microbatch_size

  mean_loss, mean_grad = optax.microbatching.micro_vmap(
      single_loss_and_grad,
      in_axes=(None, 0),  # type: ignore[arg-type]
      accumulator=(
          optax.microbatching.AccumulationType.MEAN,
          optax.microbatching.AccumulationType.MEAN,
      ),
      microbatch_size=min(global_microbatch_size, num_keys),
  )(
      params, sharded_keys
  )

  return mean_loss, mean_grad


def evaluate_setup(
    strategy,
    params,
    lr,
    T,
    w0,
    mu,
    sigma,
    rho,
    h,
    stddev,
    optimizer_factory = optax.adamw,
    k_eval = 1000,
    rng_seed = 42,
    microbatch_size = 100,
    l2_norm_clip = 1.0,
    use_checkpoint = False,
    objective = 'final',
    max_loss_threshold = 10000.0,
):
  """Evaluates the expected final loss for a given generalized noising matrix setup."""
  rng = jax.random.PRNGKey(rng_seed)
  eval_keys = jax.random.split(rng, k_eval)

  @jax.jit
  def eval_fn(p):
    return compute_objective_f(
        noising_matrix=strategy.build_normalized_noising_matrix(p),
        w0=w0,
        keys=eval_keys,
        T=T,
        mu=mu,
        sigma=sigma,
        rho=rho,
        h=h,
        stddev=stddev,
        lr=lr,
        optimizer_factory=optimizer_factory,
        microbatch_size=microbatch_size,
        l2_norm_clip=l2_norm_clip,
        use_checkpoint=use_checkpoint,
        objective=objective,
        max_loss_threshold=max_loss_threshold,
    )

  loss = eval_fn(params)
  return float(loss)


# ==============================================================================
# Outer Optimization Utilities
# ==============================================================================

ParamT = TypeVar('ParamT', bound=chex.ArrayTree)

DEFAULT_OPTIMIZER = optax.lbfgs(
    memory_size=1, linesearch=optax.scale_by_backtracking_linesearch(128)
)


@dataclasses.dataclass
class CallbackArgs:
  """Information passed to the callback function on each optimization step."""

  step: int
  loss: jnp.ndarray
  grad: chex.ArrayTree | None
  params: chex.ArrayTree
  state: Any
  raw_loss: jnp.ndarray | None = None
  penalty: jnp.ndarray | None = None
  constraint_violation: jnp.ndarray | None = None
  rho: float | jnp.ndarray | None = None
  multiplier: jnp.ndarray | None = None


CallbackFnType: TypeAlias = Callable[[CallbackArgs], None | bool]


def optimize(
    loss_and_grad,
    params,
    *,
    max_optimizer_steps = 250,
    value_fn = None,
    callback = lambda _: None,
    optimizer = DEFAULT_OPTIMIZER,
    projection_fn = None,
    tangent_projection_fn = None,
):
  """Optimize a differentiable loss function on a constrained manifold."""
  if projection_fn is not None:
    params = projection_fn(params)
    if value_fn is not None:
      user_value_fn = value_fn
      value_fn = lambda p: user_value_fn(projection_fn(p))

  def single_step(params, opt_state):
    value, grad = loss_and_grad(params)
    if tangent_projection_fn is not None:
      grad = tangent_projection_fn(params, grad)
    updates, opt_state = optimizer.update(
        grad, opt_state, params, value=value, grad=grad, value_fn=value_fn
    )
    new_params = optax.apply_updates(params, updates)
    if projection_fn is not None:
      new_params = projection_fn(cast(ParamT, new_params))
    return value, grad, new_params, opt_state

  original_dtypes = jax.tree.map(lambda x: x.dtype, params)

  params = jax.tree.map(jnp.float64, params)
  state = optimizer.init(params)
  for i in range(max_optimizer_steps):
    loss, grad, params, state = single_step(params, state)
    if callback(
        CallbackArgs(step=i, loss=loss, grad=grad, params=params, state=state)
    ):
      break

  return jax.tree.map(jnp.astype, params, original_dtypes)


def optimize_alm(
    loss_and_grad_fn,
    constraint_fn,
    params,
    *,
    outer_steps = 5,
    inner_steps = 5,
    rho_init = 1.0,
    rho_multiplier = 2.0,
    rho_max = 100.0,
    inner_optimizer = optax.adam(learning_rate=0.005),
    callback = lambda _: None,
    projection_fn = None,
):
  """Optimizes parameters with Augmented Lagrangian Method (ALM)."""
  original_dtypes = jax.tree.map(lambda x: x.dtype, params)
  params = jax.tree.map(jnp.float64, params)

  init_viol = constraint_fn(params)
  lambda_val = jnp.zeros_like(init_viol)  # scalar multiplier \lambda
  rho_val = float(rho_init)               # scalar penalty \rho
  prev_viol_norm = jnp.linalg.norm(init_viol)

  opt_state = inner_optimizer.init(params)
  global_step = 0

  def make_penalty_fn(cur_lam, cur_rho):
    def penalty_fn(p):
      viol = constraint_fn(p)
      return jnp.sum(cur_lam * viol) + 0.5 * cur_rho * jnp.sum(viol**2)
    return penalty_fn

  for _ in range(outer_steps):
    penalty_fn = make_penalty_fn(lambda_val, rho_val)
    penalty_grad_fn = jax.grad(penalty_fn)

    for _ in range(inner_steps):
      raw_loss, raw_grad = loss_and_grad_fn(params)
      penalty_val = penalty_fn(params)
      penalty_grad = penalty_grad_fn(params)

      total_loss = raw_loss + penalty_val
      total_grad = jax.tree.map(lambda g, pg: g + pg, raw_grad, penalty_grad)

      updates, opt_state = inner_optimizer.update(
          total_grad, opt_state, params
      )
      params = cast(ParamT, optax.apply_updates(params, updates))

      current_viol = constraint_fn(params)
      stop_early = callback(
          CallbackArgs(
              step=global_step,
              loss=total_loss,
              raw_loss=raw_loss,
              penalty=penalty_val,
              constraint_violation=current_viol,
              rho=rho_val,
              multiplier=lambda_val,
              grad=total_grad,
              params=params,
              state=opt_state,
          )
      )
      global_step += 1
      if stop_early:
        break

    # Dual multiplier update (Ascent)
    viol = constraint_fn(params)
    viol_norm = jnp.linalg.norm(viol)
    lambda_val = lambda_val + rho_val * viol

    # Penalty update with rho_max capping
    if viol_norm > 0.25 * prev_viol_norm:
      rho_val = min(rho_val * float(rho_multiplier), float(rho_max))
    prev_viol_norm = viol_norm

  final_viol = float(jnp.linalg.norm(constraint_fn(params)))

  if projection_fn is not None:
    params = projection_fn(params)

  params = jax.tree.map(jnp.astype, params, original_dtypes)
  return params, lambda_val, rho_val, final_viol


def optimize_lagrangian(
    loss_and_grad_fn,
    constraint_fn,
    params,
    *,
    max_optimizer_steps = 250,
    lagrangian_lambda_init = 0.1,
    lagrangian_lambda_final = 10.0,
    optimizer = optax.adam(learning_rate=0.005),
    callback = lambda _: None,
    projection_fn = None,
):
  """Optimizes parameters with an annealed Lagrangian penalty on the first column norm."""
  original_dtypes = jax.tree.map(lambda x: x.dtype, params)
  params = jax.tree.map(jnp.float64, params)

  opt_state = optimizer.init(params)

  def make_penalty_fn(lam):
    def penalty_fn(p):
      viol = constraint_fn(p)  # scalar violation: ||c||_2^2 - 1
      return 0.5 * lam * jnp.sum(viol**2)
    return penalty_fn

  for step in range(max_optimizer_steps):
    if (
        max_optimizer_steps <= 1
        or lagrangian_lambda_init == lagrangian_lambda_final
    ):
      curr_lambda = lagrangian_lambda_final
    else:
      ratio = lagrangian_lambda_final / max(lagrangian_lambda_init, 1e-12)
      curr_lambda = lagrangian_lambda_init * (
          ratio ** (step / (max_optimizer_steps - 1))
      )

    penalty_fn = make_penalty_fn(curr_lambda)
    penalty_grad_fn = jax.grad(penalty_fn)

    raw_loss, raw_grad = loss_and_grad_fn(params)
    penalty_val = penalty_fn(params)
    penalty_grad = penalty_grad_fn(params)

    total_loss = raw_loss + penalty_val
    total_grad = jax.tree.map(lambda g, pg: g + pg, raw_grad, penalty_grad)

    updates, opt_state = optimizer.update(total_grad, opt_state, params)
    params = cast(ParamT, optax.apply_updates(params, updates))

    current_viol = constraint_fn(params)
    stop_early = callback(
        CallbackArgs(
            step=step,
            loss=total_loss,
            raw_loss=raw_loss,
            penalty=penalty_val,
            constraint_violation=current_viol,
            rho=curr_lambda,
            grad=total_grad,
            params=params,
            state=opt_state,
        )
    )
    if stop_early:
      break

  final_viol = float(jnp.linalg.norm(constraint_fn(params)))

  if projection_fn is not None:
    params = projection_fn(params)

  params = jax.tree.map(jnp.astype, params, original_dtypes)
  return params, final_viol


# ==============================================================================
# Flags and Main Function
# ==============================================================================

_DNA_PATH = flags.DEFINE_string(
    'dna_path',
    None,
    'Path to the extracted DNA NPZ file containing mu, sigma, rho, h.',
)

_T = flags.DEFINE_integer('t', 1000, 'Number of iterations.')

_LR = flags.DEFINE_float('lr', 1e-3, 'Learning rate.')

_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    'adamw',
    ['sgd', 'sgdm', 'adamw'],
    'Optimizer to use.',
)

_OUTER_OPTIM_METHOD = flags.DEFINE_enum(
    'outer_optim_method',
    'alm',
    ['alm', 'projected', 'lagrangian'],
    'Outer optimization method for strategy parameters (alm, projected, or'
    ' lagrangian).',
)

_ALM_OUTER_STEPS = flags.DEFINE_integer(
    'alm_outer_steps', 5, 'Number of outer multiplier update steps for ALM.'
)

_ALM_INNER_STEPS = flags.DEFINE_integer(
    'alm_inner_steps',
    5,
    'Number of inner optimization steps per ALM outer step.',
)

_ALM_RHO_INIT = flags.DEFINE_float(
    'alm_rho_init', 1.0, 'Initial penalty parameter rho for ALM.'
)

_ALM_RHO_MULTIPLIER = flags.DEFINE_float(
    'alm_rho_multiplier',
    2.0,
    'Multiplier for rho in ALM when constraint violation does not decrease.',
)

_ALM_RHO_MAX = flags.DEFINE_float(
    'alm_rho_max',
    100.0,
    'Maximum penalty parameter rho for ALM to prevent ill-conditioning.',
)

_LAGRANGIAN_LAMBDA_INIT = flags.DEFINE_float(
    'lagrangian_lambda_init',
    0.1,
    'Initial penalty hyperparameter lambda at step 0 for Lagrangian'
    ' constraint.',
)

_LAGRANGIAN_LAMBDA_FINAL = flags.DEFINE_float(
    'lagrangian_lambda_final',
    10.0,
    'Final penalty hyperparameter lambda at last step for Lagrangian'
    ' constraint.',
)

_NOISE_MULTIPLIER = flags.DEFINE_float(
    'noise_multiplier', 4.0, 'Noise multiplier.'
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 64, 'Dummy batch size for noise calculation.'
)

_K_TRAIN = flags.DEFINE_integer(
    'k_train', 1000, 'Number of independent noise realizations for training.'
)

_K_EVAL = flags.DEFINE_integer(
    'k_eval', 1000, 'Number of independent noise realizations for evaluation.'
)

_RNG_SEED = flags.DEFINE_integer(
    'rng_seed', 42, 'PRNG seed for evaluation noise.'
)

_TRAIN_SEED = flags.DEFINE_integer(
    'train_seed', 123, 'PRNG seed for training noise.'
)

_MICROBATCH_SIZE = flags.DEFINE_integer(
    'microbatch_size', 125, 'Microbatch size for evaluation.'
)

_BANDS = flags.DEFINE_integer(
    'bands', 8, 'Number of bands for matrix strategy.'
)

_SOLVE_PARAMS_STEPS = flags.DEFINE_integer(
    'solve_params_steps',
    5,
    'Number of outer optimizer steps to solve for strategy parameters.',
)

_L2_NORM_CLIP = flags.DEFINE_float(
    'l2_norm_clip', 1.0, 'L2 norm clip for gradient.'
)

_OBJECTIVE = flags.DEFINE_enum(
    'objective',
    'final',
    ['final', 'avg'],
    'Objective type: final or avg loss.',
)

_MATRIX_FAMILY = flags.DEFINE_enum(
    'matrix_family',
    'toeplitz',
    ['toeplitz', 'banded'],
    'Matrix family to optimize.',
)

_ALPHA = flags.DEFINE_float(
    'alpha',
    0.9,
    'Alpha squashing parameter for banded matrix optimization.',
)

_MAX_LOSS_THRESHOLD = flags.DEFINE_float(
    'max_loss_threshold',
    10000.0,
    'Stop optimization early if loss exceeds this threshold.',
)

_WORK_DIR = flags.DEFINE_string(
    'work_dir', None, 'Optional directory to save optimized parameters.'
)

_USE_CHECKPOINT = flags.DEFINE_bool(
    'use_checkpoint', False, 'Use Equinox checkpointed scan to save memory.'
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # 1. Load DNA (mu, sigma, rho, h)
  if _DNA_PATH.value and os.path.exists(_DNA_PATH.value):
    with open(_DNA_PATH.value, 'rb') as f:
      dna_data = np.load(f)
      mu = jnp.array(dna_data['mu'], dtype=jnp.float64)
      sigma = jnp.array(dna_data['sigma'], dtype=jnp.float64)
      rho = jnp.array(dna_data['rho'], dtype=jnp.float64)
      h = jnp.array(dna_data['h'], dtype=jnp.float64)
    print(f'Loaded DNA from {_DNA_PATH.value} with shape {mu.shape}')
  else:
    d = 1000
    print(f'DNA path not provided or not found; using synthetic DNA with d={d}')
    mu = jnp.zeros((1, d), dtype=jnp.float64)
    sigma = jnp.ones((1, d), dtype=jnp.float64) * 0.1
    rho = jnp.ones((1, d), dtype=jnp.float64) * 0.9
    h = jnp.ones((1, d), dtype=jnp.float64)

  if mu.ndim == 1:
    mu = jnp.expand_dims(mu, 0)
    sigma = jnp.expand_dims(sigma, 0)
    rho = jnp.expand_dims(rho, 0)
    h = jnp.expand_dims(h, 0)

  d = mu.shape[-1]
  w0 = jax.random.normal(jax.random.PRNGKey(0), (d,), dtype=jnp.float64)

  if _OPTIMIZER.value == 'sgd':
    optimizer_factory = optax.sgd
  elif _OPTIMIZER.value == 'sgdm':
    optimizer_factory = lambda lr: optax.sgd(learning_rate=lr, momentum=0.9)
  elif _OPTIMIZER.value == 'adamw':
    optimizer_factory = lambda lr: optax.adamw(learning_rate=lr, eps_root=1e-7)
  else:
    raise ValueError(f'Unknown optimizer: {_OPTIMIZER.value}')

  if _MATRIX_FAMILY.value == 'toeplitz':
    strategy = BandedToeplitzOptimizationStrategy(
        T=_T.value,
        bands=_BANDS.value,
        unconstrained=(_OUTER_OPTIM_METHOD.value == 'alm'),
    )
  elif _MATRIX_FAMILY.value == 'banded':
    strategy = BandedOptimizationStrategy(
        T=_T.value, bands=_BANDS.value, alpha=_ALPHA.value
    )
  else:
    raise ValueError(f'Unknown matrix family: {_MATRIX_FAMILY.value}')

  # Optimize strategy parameters
  train_rng = jax.random.PRNGKey(_TRAIN_SEED.value)
  train_keys = jax.random.split(train_rng, _K_TRAIN.value)

  stddev = _NOISE_MULTIPLIER.value * _L2_NORM_CLIP.value / _BATCH_SIZE.value

  def train_objective(params):
    return compute_strategy_loss_and_grad(
        strategy=strategy,
        params=params,
        lr=_LR.value,
        w0=w0,
        keys=train_keys,
        T=_T.value,
        mu=mu,
        sigma=sigma,
        rho=rho,
        h=h,
        stddev=stddev,
        optimizer_factory=optimizer_factory,
        microbatch_size=_MICROBATCH_SIZE.value,
        l2_norm_clip=_L2_NORM_CLIP.value,
        objective=_OBJECTIVE.value,
        use_checkpoint=_USE_CHECKPOINT.value,
        max_loss_threshold=_MAX_LOSS_THRESHOLD.value,
    )

  def train_objective_val(params):
    return compute_strategy_loss(
        strategy=strategy,
        params=params,
        lr=_LR.value,
        w0=w0,
        keys=train_keys,
        T=_T.value,
        mu=mu,
        sigma=sigma,
        rho=rho,
        h=h,
        stddev=stddev,
        optimizer_factory=optimizer_factory,
        microbatch_size=_MICROBATCH_SIZE.value,
        l2_norm_clip=_L2_NORM_CLIP.value,
        objective=_OBJECTIVE.value,
        use_checkpoint=_USE_CHECKPOINT.value,
        max_loss_threshold=_MAX_LOSS_THRESHOLD.value,
    )

  def callback_fn(args, writer = None):
    grad_norm = (
        float(optax.global_norm(args.grad))
        if args.grad is not None
        else float('nan')
    )
    raw_loss_val = (
        float(args.raw_loss) if args.raw_loss is not None else float(args.loss)
    )
    total_loss_val = float(args.loss)

    msg = (
        f'Step {args.step:2d}, raw_loss: {raw_loss_val:.9f},'
        f' total_loss: {total_loss_val:.9f}, grad norm: {grad_norm:.9f}'
    )
    if args.constraint_violation is not None:
      msg += f', viol: {float(args.constraint_violation):.6e}'
    if args.rho is not None:
      msg += f', rho: {float(args.rho):.2e}'
    print(msg)

    if writer is not None:
      log_dict = {
          'outer_step': int(args.step),
          'outer_loss': raw_loss_val,
          'total_loss': total_loss_val,
          'outer_grad_norm': grad_norm,
      }
      if args.constraint_violation is not None:
        log_dict['constraint_violation'] = float(args.constraint_violation)
      if args.rho is not None:
        log_dict['rho'] = float(args.rho)
      if args.multiplier is not None:
        log_dict['multiplier'] = float(args.multiplier)
      writer.write(log_dict)

    if (
        jnp.isnan(args.loss)
        or jnp.isnan(grad_norm)
        or raw_loss_val > _MAX_LOSS_THRESHOLD.value
    ):
      print(
          f'Loss ({raw_loss_val:.2f}) exceeded threshold'
          f' ({_MAX_LOSS_THRESHOLD.value}) or NaN detected! Stopping.'
      )
      return True
    return False

  if _OUTER_OPTIM_METHOD.value == 'alm':
    print(
        'Optimizing strategy parameters with ALM'
        f' ({_ALM_OUTER_STEPS.value} outer steps x'
        f' {_ALM_INNER_STEPS.value} inner steps,'
        f' rho_init={_ALM_RHO_INIT.value}, rho_max={_ALM_RHO_MAX.value})...'
    )
    optimized_V, final_lambda, final_rho, final_viol = optimize_alm(
        loss_and_grad_fn=train_objective,
        constraint_fn=strategy.constraint_fn,
        params=strategy.init_params(),
        outer_steps=_ALM_OUTER_STEPS.value,
        inner_steps=_ALM_INNER_STEPS.value,
        rho_init=_ALM_RHO_INIT.value,
        rho_multiplier=_ALM_RHO_MULTIPLIER.value,
        rho_max=_ALM_RHO_MAX.value,
        inner_optimizer=optax.adam(learning_rate=0.005),
        callback=callback_fn,
        projection_fn=strategy.project_params,
    )
    print(f'Optimized V (after ALM + retraction): {optimized_V}')
    print(f'Final ALM Lambda: {final_lambda}, Final Rho: {final_rho}')
    print(f'Final Constraint Violation (||c||^2 - 1): {final_viol:.6e}')
  elif _OUTER_OPTIM_METHOD.value == 'lagrangian':
    print(
        'Optimizing strategy parameters with Lagrangian penalty (lambda:'
        f' {_LAGRANGIAN_LAMBDA_INIT.value} ->'
        f' {_LAGRANGIAN_LAMBDA_FINAL.value})...'
    )
    optimized_V, final_viol = optimize_lagrangian(
        loss_and_grad_fn=train_objective,
        constraint_fn=strategy.constraint_fn,
        params=strategy.init_params(),
        max_optimizer_steps=_SOLVE_PARAMS_STEPS.value,
        lagrangian_lambda_init=_LAGRANGIAN_LAMBDA_INIT.value,
        lagrangian_lambda_final=_LAGRANGIAN_LAMBDA_FINAL.value,
        optimizer=optax.adam(learning_rate=0.005),
        callback=callback_fn,
        projection_fn=strategy.project_params,
    )
    print(f'Optimized V (after Lagrangian + retraction): {optimized_V}')
    print(f'Final Constraint Violation (||c||^2 - 1): {final_viol:.6e}')
  else:
    print('Optimizing strategy parameters on dynamic DNA quadratic proxy...')
    optimized_V = optimize(
        loss_and_grad=train_objective,
        params=strategy.init_params(),
        max_optimizer_steps=_SOLVE_PARAMS_STEPS.value,
        value_fn=train_objective_val,
        callback=callback_fn,
        optimizer=optax.adam(learning_rate=0.005),
        projection_fn=strategy.project_params,
        tangent_projection_fn=strategy.project_grad,
    )
    print(f'Optimized V: {optimized_V}')

  if _WORK_DIR.value is not None:
    save_dir = _WORK_DIR.value
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    optimized_V_path = os.path.join(save_dir, 'optimized_V.npy')
    with open(optimized_V_path, 'wb') as f:
      np.save(f, np.array(optimized_V))
    print(f'Saved optimized V to {optimized_V_path}')

  # Evaluate
  print('Evaluating optimized strategy...')

  eval_kwargs = dict(
      T=_T.value,
      w0=w0,
      mu=mu,
      sigma=sigma,
      rho=rho,
      h=h,
      stddev=stddev,
      optimizer_factory=optimizer_factory,
      k_eval=_K_EVAL.value,
      rng_seed=_RNG_SEED.value,
      microbatch_size=_MICROBATCH_SIZE.value,
      l2_norm_clip=_L2_NORM_CLIP.value,
      objective=_OBJECTIVE.value,
      use_checkpoint=_USE_CHECKPOINT.value,
  )

  loss = evaluate_setup(
      strategy=strategy,
      params=optimized_V,
      lr=_LR.value,
      **eval_kwargs,
  )
  print(f'Optimized Loss: {loss:.6f}')

  # Evaluate baseline (Identity / unconstrained init)
  baseline_params = strategy.init_params()
  loss_baseline = evaluate_setup(
      strategy=strategy,
      params=baseline_params,
      lr=_LR.value,
      **eval_kwargs,
  )
  print(
      'Baseline Loss (Identity / init params, fixed LR):'
      f' {loss_baseline:.6f}'
  )


if __name__ == '__main__':
  app.run(main)
