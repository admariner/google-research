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

"""Sandbox for DP noising matrix optimization with fixed preconditioners."""

# pylint: disable=invalid-name
import abc
import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import os
from typing import Any, TypeAlias, TypeVar

from absl import app
from absl import flags
import chex
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import banded
from jax_privacy.matrix_factorization import streaming_matrix
import numpy as np
import optax


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
    """Optionally restricts/projects parameters to a safe zone.

    Args:
      params: Tunable matrix strategy parameters.

    Returns:
      Projected safe parameters.
    """
    return params


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


# ==============================================================================
# Optimization Utilities
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


CallbackFnType: TypeAlias = Callable[[CallbackArgs], None | bool]


def jax_enable_x64(fn):
  """Decorator to enable x64 precision for a function."""

  def wrapped_fn(*args, **kwargs):
    with jax.enable_x64():
      return fn(*args, **kwargs)

  return wrapped_fn


def optim(
    loss_fn,
    noising_matrix,
    w0,
    key,
    T,
    base_noise_std,
    optimizer,
    scale_constant = 1.0,
    use_checkpoint = False,
):
  """Runs DP optimizer for T iterations on a model PyTree."""
  del use_checkpoint
  d = w0.shape[0]
  params = {'q': w0}

  grad_fn = jax.grad(loss_fn)

  opt_state = optimizer.init(params)
  matrix_state = noising_matrix.init_multiply(jnp.zeros(d, dtype=w0.dtype))

  @jax.jit
  def step_fn(carry, i):
    del i  # Unused.
    params_state, opt_state_val, key_t, matrix_state_val = carry
    key_t, subkey = jax.random.split(key_t)
    Z_t = jax.random.normal(subkey, shape=(d,), dtype=w0.dtype)
    E_t, matrix_state_val = noising_matrix.multiply_next(Z_t, matrix_state_val)
    E_t = base_noise_std * E_t
    E_t = E_t.astype(w0.dtype)

    grads = grad_fn(params_state)

    g_norm = optax.global_norm(grads)
    divisor = jnp.maximum(g_norm / scale_constant, 1.0)
    grads = jax.tree.map(lambda t: t / divisor, grads)
    noisy_grads = jax.tree.map(lambda g: g + scale_constant * E_t, grads)

    updates, opt_state_val = optimizer.update(
        noisy_grads, opt_state_val, params_state
    )
    params_state = optax.apply_updates(params_state, updates)
    return (params_state, opt_state_val, key_t, matrix_state_val), None

  import equinox as eqx  # pylint: disable=g-import-not-at-top, import-outside-toplevel

  (final_params, _, _, _), _ = eqx.internal.scan(
      step_fn,
      (params, opt_state, key, matrix_state),
      xs=jnp.arange(T),
      kind='checkpointed',
      checkpoints=4,
  )
  final_loss = loss_fn(final_params)
  return final_loss


def compute_objective_f(
    noising_matrix,
    loss_fn,
    w0,
    keys,
    T,
    base_noise_std = 0.1,
    lr = 0.01,
    optimizer_factory = optax.sgd,
    microbatch_size = 100,
    scale_constant = 1.0,
    use_checkpoint = False,
):
  """Computes the expected final loss f over k independent noise realizations."""
  optimizer = optimizer_factory(lr)

  optim_fn = lambda key_i: optim(
      loss_fn=loss_fn,
      noising_matrix=noising_matrix,
      w0=w0,
      key=key_i,
      T=T,
      base_noise_std=base_noise_std,
      optimizer=optimizer,
      scale_constant=scale_constant,
      use_checkpoint=use_checkpoint,
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

  losses = optax.microbatching.micro_vmap(
      optim_fn,
      accumulator=optax.microbatching.AccumulationType.CONCAT,
      microbatch_size=min(global_microbatch_size, num_keys),
  )(sharded_keys)
  return jnp.mean(losses, axis=0)


def compute_strategy_loss(
    strategy,
    params,
    lr,
    loss_fn,
    w0,
    keys,
    T,
    base_noise_std,
    optimizer_factory = optax.sgd,
    microbatch_size = 100,
    use_checkpoint = False,
):
  """Computes expected final loss over k independent noise realizations."""

  def optim_fn(params_val, key_i):
    optimizer = optimizer_factory(lr)
    matrix = strategy.build_noising_matrix(params_val)
    return optim(
        loss_fn=loss_fn,
        noising_matrix=matrix,
        w0=w0,
        key=key_i,
        T=T,
        base_noise_std=base_noise_std,
        optimizer=optimizer,
        use_checkpoint=use_checkpoint,
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

  losses = optax.microbatching.micro_vmap(
      optim_fn,
      in_axes=(None, 0),
      accumulator=optax.microbatching.AccumulationType.CONCAT,
      microbatch_size=min(global_microbatch_size, num_keys),
  )(params, sharded_keys)
  return jnp.mean(losses, axis=0)


def compute_strategy_loss_and_grad(
    strategy,
    params,
    lr,
    loss_fn,
    w0,
    keys,
    T,
    base_noise_std,
    optimizer_factory = optax.sgd,
    microbatch_size = 100,
    scale_constant = 1.0,
    use_checkpoint = False,
):
  """Computes expected final loss and its gradient wrt strategy params."""

  def optim_fn(params_val, key_i):
    optimizer = optimizer_factory(lr)
    matrix = strategy.build_noising_matrix(params_val)
    return optim(
        loss_fn=loss_fn,
        noising_matrix=matrix,
        w0=w0,
        key=key_i,
        T=T,
        base_noise_std=base_noise_std,
        optimizer=optimizer,
        scale_constant=scale_constant,
        use_checkpoint=use_checkpoint,
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

  losses, grads = optax.microbatching.micro_vmap(
      single_loss_and_grad,
      in_axes=(None, 0),
      accumulator=(
          optax.microbatching.AccumulationType.CONCAT,
          optax.microbatching.AccumulationType.CONCAT,
      ),
      microbatch_size=min(global_microbatch_size, num_keys),
  )(
      params, sharded_keys
  )

  mean_loss = jnp.mean(losses, axis=0)
  mean_grad = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads)
  return mean_loss, mean_grad


def evaluate_setup(
    strategy,
    params,
    lr,
    loss_fn,
    T,
    w0,
    base_noise_std,
    optimizer_factory = optax.adamw,
    k_eval = 1000,
    rng_seed = 42,
    microbatch_size = 100,
    scale_constant = 1.0,
    use_checkpoint = False,
):
  """Evaluates the expected final loss for a given generalized noising matrix setup."""
  rng = jax.random.PRNGKey(rng_seed)
  eval_keys = jax.random.split(rng, k_eval)

  @jax.jit
  def eval_fn(p):
    return compute_objective_f(
        noising_matrix=strategy.build_normalized_noising_matrix(p),
        loss_fn=loss_fn,
        w0=w0,
        keys=eval_keys,
        T=T,
        base_noise_std=base_noise_std,
        lr=lr,
        optimizer_factory=optimizer_factory,
        microbatch_size=microbatch_size,
        scale_constant=scale_constant,
        use_checkpoint=use_checkpoint,
    )

  loss = eval_fn(params)
  return float(loss)


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


def quad_model_loss(params, H):
  """Computes the generalized quadratic loss 1/2 * q^T H q."""
  q = params['q']
  return 0.5 * jnp.dot(q, H * q)


# ==============================================================================
# Flags and Main Function
# ==============================================================================

_INITIAL_PARAMS_PATH = flags.DEFINE_string(
    'initial_params_path',
    '/path/to/model_checkpoints/tiny_bert.npz',
    'Path to the initial model parameters checkpoint (NPZ format).',
)

_PRECONDITIONER_PATH = flags.DEFINE_string(
    'preconditioner_path',
    '/path/to/checkpoints/flat_precond.npy',
    'Path to the preconditioner file (Hessian diagonal flat_precond.npy).',
)

_T = flags.DEFINE_integer('t', 1000, 'Number of iterations.')

_LR = flags.DEFINE_float('lr', 1e-3, 'Learning rate.')

_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    'adamw',
    ['sgd', 'sgdm', 'adamw'],
    'Optimizer to use.',
)

_BASE_NOISE_STD = flags.DEFINE_float(
    'base_noise_std', 0.0625, 'Scale of the noise standard deviation.'
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

_SCALE_CONSTANT = flags.DEFINE_float(
    'scale_constant', 1.0, 'Scale constant (clip norm) for gradient.'
)

_INIT_PARAMS_SEED = flags.DEFINE_integer(
    'init_params_seed', 42, 'PRNG seed for strategy parameter initialization.'
)

_WORK_DIR = flags.DEFINE_string(
    'work_dir', None, 'Optional directory to save optimized parameters.'
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # 1. Load preconditioner H
  with open(_PRECONDITIONER_PATH.value, 'rb') as f:
    H = np.load(f)
  H = np.sqrt(np.abs(H))
  H = H / np.max(np.abs(H))
  H = jnp.array(H)

  # 2. Load w0
  jax_params = collections.defaultdict(dict)
  with open(_INITIAL_PARAMS_PATH.value, 'rb') as f:
    # pylint: disable-next=g-unsafe-pickle-load
    params_npz = np.load(f, allow_pickle=True)
    try:
      for key, value in params_npz.items():
        jax_params[key] = value.item()
      leaves = jax.tree_util.tree_leaves(jax_params)
      flat_leaves = [jnp.ravel(leaf) for leaf in leaves]
      w0 = jnp.concatenate(flat_leaves)
    except Exception as e:  # pylint: disable=broad-exception-caught
      try:
        key = list(params_npz.keys())[0]
        w0 = jnp.array(params_npz[key])
      except Exception as e_fallback:
        raise ValueError(
            f'Failed to load NPZ file: {e}. Fallback failed: {e_fallback}'
        ) from e

  quadratic = lambda p: jnp.dot(p['q'], H * p['q'])

  if _OPTIMIZER.value == 'sgd':
    optimizer_factory = optax.sgd
  elif _OPTIMIZER.value == 'sgdm':
    optimizer_factory = lambda lr: optax.sgd(learning_rate=lr, momentum=0.9)
  elif _OPTIMIZER.value == 'adamw':
    optimizer_factory = lambda lr: optax.adamw(learning_rate=lr, eps_root=1e-7)
  else:
    raise ValueError(f'Unknown optimizer: {_OPTIMIZER.value}')

  strategy = BandedOptimizationStrategy(T=_T.value, bands=_BANDS.value)

  # Optimize
  train_rng = jax.random.PRNGKey(_TRAIN_SEED.value)
  train_keys = jax.random.split(train_rng, _K_TRAIN.value)

  def train_objective(params):
    return compute_strategy_loss_and_grad(
        strategy=strategy,
        params=params,
        lr=_LR.value,
        loss_fn=quadratic,
        w0=w0,
        keys=train_keys,
        T=_T.value,
        base_noise_std=_BASE_NOISE_STD.value,
        optimizer_factory=optimizer_factory,
        microbatch_size=_MICROBATCH_SIZE.value,
        scale_constant=_SCALE_CONSTANT.value,
    )

  def train_objective_val(params):
    return compute_strategy_loss(
        strategy=strategy,
        params=params,
        lr=_LR.value,
        loss_fn=quadratic,
        w0=w0,
        keys=train_keys,
        T=_T.value,
        base_noise_std=_BASE_NOISE_STD.value,
        optimizer_factory=optimizer_factory,
        microbatch_size=_MICROBATCH_SIZE.value,
    )

  def callback_fn(args, writer=None):
    grad_norm = optax.global_norm(args.grad)
    print(
        f'Step {args.step:2d}, loss: {args.loss:.9f}, '
        f'grad norm: {grad_norm:.9f}'
    )

    if writer is not None:
      writer.write({
          'outer_step': int(args.step),
          'outer_loss': float(args.loss),
          'outer_grad_norm': float(grad_norm),
      })

    if jnp.isnan(args.loss) or jnp.isnan(grad_norm):
      print('NaN detected! Stopping.')
      return True
    return False

  print('Optimizing strategy parameters...')
  optimized_V = optimize(
      loss_and_grad=train_objective,
      params=strategy.init_params(),
      max_optimizer_steps=_SOLVE_PARAMS_STEPS.value,
      value_fn=train_objective_val,
      callback=callback_fn,
      optimizer=optax.adam(learning_rate=0.01),
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
      loss_fn=quadratic,
      T=_T.value,
      w0=w0,
      base_noise_std=_BASE_NOISE_STD.value,
      optimizer_factory=optimizer_factory,
      k_eval=_K_EVAL.value,
      rng_seed=_RNG_SEED.value,
      microbatch_size=_MICROBATCH_SIZE.value,
      scale_constant=_SCALE_CONSTANT.value,
  )

  loss = evaluate_setup(
      strategy=strategy,
      params=optimized_V,
      lr=_LR.value,
      **eval_kwargs,
  )
  print(f'Optimized Loss: {loss:.6f}')

  # Also evaluate baselines (V=0)
  # Baseline: Isotropic with Fixed LR
  baseline_params = jnp.zeros_like(optimized_V)
  loss_baseline = evaluate_setup(
      strategy=strategy,
      params=baseline_params,
      lr=_LR.value,
      **eval_kwargs,
  )
  print(f'Baseline Loss (V=0, fixed LR): {loss_baseline:.6f}')


if __name__ == '__main__':
  app.run(main)
