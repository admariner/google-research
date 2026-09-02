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

"""Evaluation script for BLT matrix strategy."""

# pylint: disable=invalid-name
import abc
import collections
from collections.abc import Callable, Mapping
from typing import Any, Sequence

from absl import app
from absl import flags
import chex
import jax
import jax.numpy as jnp
import jax.sharding
from jax_privacy.matrix_factorization import streaming_matrix
import numpy as np
import optax


jax.config.update('jax_enable_x64', True)


def blt_cinv_streaming_matrix(
    buf_decay, output_scale
):
  """Applies C^{-1} for a BLT matrix using Algorithm 3.

  Args:
    buf_decay: Buffer decay parameters of C.
    output_scale: Output scale parameters of C.

  Returns:
    A StreamingMatrix implementing C^{-1}.
  """

  def init(abstract_yi):
    num_buffers = buf_decay.shape[0]
    zero = jnp.zeros_like(abstract_yi, dtype=buf_decay.dtype)
    return jnp.broadcast_to(zero, (num_buffers,) + zero.shape)

  def _read(state):
    return jnp.tensordot(output_scale, state, axes=((0,), (0,)))

  def _update(state, xi):
    decay_expanded = jnp.expand_dims(
        buf_decay, axis=tuple(range(1, state.ndim))
    )
    state_decayed = state * decay_expanded
    return state_decayed + xi

  def next_fn(yi, state):
    xi = yi - _read(state)
    state = _update(state, xi)
    return xi, state

  return streaming_matrix.StreamingMatrix.from_array_implementation(
      init, next_fn
  )


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
    """Optionally restricts/projects parameters to a safe zone."""
    return params


class BLTStrategy(MatrixStrategy):
  """A strategy using Buffered Linear Toeplitz (BLT) matrices."""

  def __init__(self, T, num_buffers):
    self.T = T
    self.num_buffers = num_buffers

  def init_params(self):
    # Placeholder
    return jnp.zeros(self.num_buffers * 2)

  def build_noising_matrix(
      self, params
  ):
    return self.build_normalized_noising_matrix(params)

  def build_normalized_noising_matrix(
      self, params
  ):
    # params is [buf_decay_0, ..., buf_decay_{K-1},
    #            output_scale_0, ..., output_scale_{K-1}]
    buf_decay = params[: self.num_buffers]
    output_scale = params[self.num_buffers :]

    # Compute C coefficients for normalization
    powers = jnp.arange(self.T - 1)
    tmp = buf_decay**powers[:, None] * output_scale
    c_coefs = jnp.concatenate([
        jnp.array([1.0], dtype=buf_decay.dtype),
        jnp.sum(tmp, axis=1),
    ])

    col_norms = jnp.sqrt(jnp.cumsum(c_coefs**2))[::-1]

    cinv_streaming = blt_cinv_streaming_matrix(buf_decay, output_scale)
    return streaming_matrix.scale_rows_and_columns(
        cinv_streaming, row_scale=col_norms
    )

  def project_params(self, params):
    # Avoid exact 0 and 1 to prevent numerical issues in BLT
    return jnp.clip(params, 0.001, 0.999)


# ==============================================================================
# Optimization and Evaluation Utils (copied from sandbox)
# ==============================================================================


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
    enable_debug = False,
    is_debug = False,
    objective = 'final',
):
  """Runs DP optimization trajectory and returns final or average loss."""
  del use_checkpoint, enable_debug, is_debug
  d = w0.shape[0]
  params = {'q': w0}

  grad_fn = jax.grad(loss_fn)

  opt_state = optimizer.init(params)
  matrix_state = noising_matrix.init_multiply(jnp.zeros(d, dtype=w0.dtype))

  def step_fn(carry, _):
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

    loss_val = loss_fn(params_state)

    return (params_state, opt_state_val, key_t, matrix_state_val), loss_val

  (final_params, _, _, _), step_losses = jax.lax.scan(
      step_fn,
      (params, opt_state, key, matrix_state),
      xs=jnp.arange(T),
  )
  final_loss = loss_fn(final_params)

  if objective == 'final':
    return final_loss
  elif objective == 'avg':
    initial_loss = loss_fn(params)
    all_losses = jnp.concatenate([jnp.array([initial_loss]), step_losses])
    return jax.numpy.mean(all_losses)
  else:
    raise ValueError(f'Unknown objective: {objective}')


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
    enable_debug = False,
    objective = 'final',
):
  """Computes mean loss over multiple random seeds."""
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
      enable_debug=enable_debug,
      is_debug=jnp.all(key_i == keys[0]),
      objective=objective,
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


def evaluate_setup(
    strategy,
    params,
    loss_fn,
    T,
    lr,
    w0,
    base_noise_std,
    optimizer_factory = optax.adamw,
    k_eval = 1000,
    rng_seed = 42,
    microbatch_size = 100,
    scale_constant = 1.0,
    use_checkpoint = False,
    enable_debug = False,
    objective = 'final',
):
  """Evaluates strategy parameters on the given loss function."""
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
        enable_debug=enable_debug,
        objective=objective,
    )

  loss = eval_fn(params)
  return float(loss)


# ==============================================================================
# Main evaluation script logic
# ==============================================================================

_INITIAL_PARAMS_PATH = flags.DEFINE_string(
    'initial_params_path',
    '/path/to/model_checkpoints/tiny_bert.npz',
    'Path to the initial model parameters checkpoint (NPZ format).',
)

_PRECONDITIONER_PATH = flags.DEFINE_string(
    'preconditioner_path',
    '/path/to/checkpoints/percentiles.npy',
    'Path to the preconditioner file (sparse_H).',
)

_T = flags.DEFINE_integer('t', 1000, 'Number of iterations.')

_LR = flags.DEFINE_float('lr', 1e-3, 'Learning rate.')

_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    'adamw',
    ['sgd', 'sgdm', 'adamw', 'adamw_norm'],
    'Optimizer to use.',
)

_BASE_NOISE_STD = flags.DEFINE_float(
    'base_noise_std', 0.0625, 'Scale of the noise standard deviation.'
)

_K_EVAL = flags.DEFINE_integer(
    'k_eval', 1000, 'Number of independent noise realizations for evaluation.'
)

_RNG_SEED = flags.DEFINE_integer(
    'rng_seed', 42, 'PRNG seed for evaluation noise.'
)

_W0_SEED = flags.DEFINE_integer('w0_seed', 114, 'Seed for w0 initialization.')

_MICROBATCH_SIZE = flags.DEFINE_integer(
    'microbatch_size', 125, 'Microbatch size for evaluation.'
)

_BUF_DECAY_0 = flags.DEFINE_float('buf_decay_0', 0.5, 'Buf decay 0 parameter.')
_BUF_DECAY_1 = flags.DEFINE_float('buf_decay_1', 0.6, 'Buf decay 1 parameter.')
_OUTPUT_SCALE_0 = flags.DEFINE_float(
    'output_scale_0', 0.5, 'Output scale 0 parameter.'
)
_OUTPUT_SCALE_1 = flags.DEFINE_float(
    'output_scale_1', 0.4, 'Output scale 1 parameter.'
)

_WORK_DIR = flags.DEFINE_string(
    'work_dir', None, 'Optional directory to save evaluated losses.'
)

_DEBUG = flags.DEFINE_bool(
    'debug', False, 'Whether to enable step debugging and printing.'
)

_OBJECTIVE = flags.DEFINE_enum(
    'objective',
    'final',
    ['final', 'avg'],
    'Objective to minimize (final loss or average loss).',
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

  # 3. Define loss function
  quadratic = lambda p: jnp.dot(p['q'], H * p['q'])

  if _OPTIMIZER.value == 'sgd':
    optimizer_factory = optax.sgd
    lr = _LR.value
  elif _OPTIMIZER.value == 'sgdm':
    optimizer_factory = lambda lr: optax.sgd(learning_rate=lr, momentum=0.9)
    lr = _LR.value
  elif _OPTIMIZER.value == 'adamw':
    optimizer_factory = optax.adamw
    lr = _LR.value
  else:
    raise ValueError(f'Unknown optimizer: {_OPTIMIZER.value}')

  eval_kwargs = dict(
      loss_fn=quadratic,
      T=_T.value,
      lr=lr,
      w0=w0,
      base_noise_std=_BASE_NOISE_STD.value,
      optimizer_factory=optimizer_factory,
      k_eval=_K_EVAL.value,
      rng_seed=_RNG_SEED.value,
      microbatch_size=_MICROBATCH_SIZE.value,
      enable_debug=_DEBUG.value,
      objective=_OBJECTIVE.value,
  )

  buf_decay = [_BUF_DECAY_0.value, _BUF_DECAY_1.value]
  output_scale = [_OUTPUT_SCALE_0.value, _OUTPUT_SCALE_1.value]

  num_buffers = 2

  ms = BLTStrategy(T=_T.value, num_buffers=num_buffers)
  params = jnp.concatenate([jnp.array(buf_decay), jnp.array(output_scale)])

  loss = evaluate_setup(
      strategy=ms, params=params, **eval_kwargs
  )
  print(f'Loss: {loss:.6f}')


if __name__ == '__main__':
  app.run(main)
