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

"""Training loop on TinyBERT."""

# pylint: disable=invalid-name
import collections
import functools
import itertools
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
from jax_privacy import noise_addition
from jax_privacy.matrix_factorization import banded
from jax_privacy.matrix_factorization import toeplitz
import numpy as np
import optax
import tensorflow as tf

from mf_opt_over_opt.dpmf.generate_noising_matrix import matrix_strategy
from mf_opt_over_opt.dpmf.generate_noising_matrix.factorizer import noisecurve_factorizer
from mf_opt_over_opt.dpmf.train import bert_model

# Data processing flags.
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 512, 'Batch size')
_EVAL_BATCH_SIZE = flags.DEFINE_integer(
    'eval_batch_size', 32, 'Evaluation batch size'
)
_ITERATIONS = flags.DEFINE_integer('iterations', 10, 'Number of iterations')
_SHUFFLE = flags.DEFINE_bool('shuffle', False, 'Shuffle training data.')

# Optimizer flags.
_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    'sgd',
    ['adamw', 'sgd', 'sgdm'],
    'Optimizer to use.',
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 0.01, 'Learning rate for training'
)
_STABILITY_EPS = flags.DEFINE_float('stability_eps', 1e-8, 'Stability epsilon')

# Noise flags.
_BATCH_L2_NORM_CLIP = flags.DEFINE_float(
    'batch_l2_norm_clip', 0.0, 'L2 norm clip for batch gradients.'
)
_NOISE_MULTIPLIER = flags.DEFINE_float(
    'noise_multiplier',
    0.0,
    'Noise multiplier (applied to the batch clip norm).',
)
_MIN_SEP = flags.DEFINE_integer(
    'min_sep',
    1,
    'minimum separation for Toeplitz matrix',
)

# Seeds.
_SEED = flags.DEFINE_integer(
    'seed',
    0,
    'Seed for jax PRNG, affecting model training and noise generation',
)
_EVAL_SEED = flags.DEFINE_integer(
    'eval_seed', 0, 'Seed for jax PRNG used during evaluation'
)
_DATA_SEED = flags.DEFINE_integer('data_seed', 0, 'Seed for tf data shuffle')

# Task definition flags.
_DATASET_PATH = flags.DEFINE_string(
    'dataset_path',
    '/path/to/arxiv_splits/train-tf',
    'Dataset path',
)
_VALID_DATASET_PATH = flags.DEFINE_string(
    'valid_dataset_path',
    '/path/to/arxiv_splits/valid-tf',
    'Validation dataset path',
)
_TEST_DATASET_PATH = flags.DEFINE_string(
    'test_dataset_path',
    '/path/to/arxiv_splits/test-tf',
    'Test dataset path',
)
_INITIAL_CHECKPOINT_PATH = flags.DEFINE_string(
    'initial_checkpoint_path',
    '/path/to/model_checkpoints/tiny_bert.npz',
    'Initial checkpoint path',
)
_VALID_STEPS = flags.DEFINE_integer(
    'valid_steps', 1200, 'Number of steps between evaluations.'
)

# Output flags.
_ROOT_OUTPUT_DIR = flags.DEFINE_string(
    'root_output_dir', None, 'Root output directory'
)
_RUN_NAME = flags.DEFINE_string('run_name', None, 'Run name')
_MATRIX_FAMILY = flags.DEFINE_enum(
    'matrix_family',
    'toeplitz',
    [
        'toeplitz',
        'row_norm',
        'nc_plus_c',
        'toep_Cinv',
        'dense_toep',
        'blt',
        'banded_opt',
        'banded_X',
    ],
    'Matrix family to use.',
)
_PRECONDITIONER_PATH = flags.DEFINE_string(
    'preconditioner_path',
    None,
    'Path to the preconditioner file (Hessian diagonal). Required for'
    ' nc_plus_c.',
)
_COEF_PATH = flags.DEFINE_string(
    'coef_path',
    None,
    'Path to the saved coefficients or parameters file (npy). For toeplitz, it'
    ' can be the full coef file (size min_sep) or parameters file (V) of size'
    ' min_sep-1 (prepended with 1.0). For row_norm, it is the optimized z'
    ' file.',
)
_COEF_C = flags.DEFINE_float(
    'coef_c',
    None,
    'Parameter c for SingleParamToeplitz (coef = c ** np.arange(min_sep)).',
)
_COEF_A = flags.DEFINE_float(
    'coef_a',
    None,
    'Parameter a for TwoParamToeplitz [1, a, b, b^2, ...].',
)
_COEF_B = flags.DEFINE_float(
    'coef_b',
    None,
    'Parameter b for TwoParamToeplitz [1, a, b, b^2, ...].',
)
_COEF_U = flags.DEFINE_float(
    'coef_u',
    None,
    'Parameter u for TwoParamCinvToeplitz (folded to a).',
)
_COEF_V = flags.DEFINE_float(
    'coef_v',
    None,
    'Parameter v for TwoParamCinvToeplitz (folded to b).',
)
_COEF_BUF_DECAY_0 = flags.DEFINE_float(
    'coef_buf_decay_0', None, 'Parameter buf_decay_0 for BLT.'
)
_COEF_BUF_DECAY_1 = flags.DEFINE_float(
    'coef_buf_decay_1', None, 'Parameter buf_decay_1 for BLT.'
)
_COEF_OUTPUT_SCALE_0 = flags.DEFINE_float(
    'coef_output_scale_0', None, 'Parameter output_scale_0 for BLT.'
)
_COEF_OUTPUT_SCALE_1 = flags.DEFINE_float(
    'coef_output_scale_1', None, 'Parameter output_scale_1 for BLT.'
)
_USE_RMSE_TOEPLITZ = flags.DEFINE_bool(
    'use_rmse_toeplitz',
    False,
    'Whether to use default RMSE optimized toeplitz matrix when coef_path is'
    ' not provided.',
)
_PRECONDITIONER_SAVE_FREQUENCY = flags.DEFINE_integer(
    'preconditioner_save_frequency',
    0,
    'Frequency in steps to save the preconditioner. If 0, the'
    ' preconditioner is only saved at the end.',
)


@functools.partial(jax.jit, static_argnames=['b', 'T'])
def generate_C_row_norm(params, b, T):
  """Maps unconstrained params to a strictly valid banded matrix."""
  z_padded = jnp.append(params, 0.0)
  idx = jnp.arange(T)

  def step_fn(carry, xs):
    C, D = carry
    j, z_j = xs
    tilde_D_j = -jnp.dot(C[j, :], D)
    K_j = jnp.sum(tilde_D_j**2)
    c_min = jnp.sqrt(1.0 + K_j)
    MAX_SLACK = 5.0
    slack = MAX_SLACK * jax.nn.sigmoid(z_j)
    c_j = jnp.where(j < T - 1, c_min + slack, c_min)
    C_jj = c_min / c_j
    D_jj = c_j / c_min
    D_row = tilde_D_j * D_jj
    D_row = D_row.at[j].set(D_jj)
    D = D.at[j, :].set(D_row)
    rem_mass = 1.0 - C_jj**2
    num_sub = jnp.minimum(b, T - 1 - j)
    safe_mass = jnp.maximum(rem_mass, 1e-8)
    safe_num_sub = jnp.maximum(num_sub, 1)
    val = jnp.sqrt(safe_mass / safe_num_sub)
    mask = (idx > j) & (idx <= j + b)
    C_col = jnp.where(mask, val, 0.0)
    C_col = C_col.at[j].set(C_jj)
    C = C.at[:, j].set(C_col)
    return (C, D), None

  init_C = jnp.zeros((T, T), dtype=params.dtype)
  init_D = jnp.zeros((T, T), dtype=params.dtype)
  scan_xs = (idx, z_padded)

  (C_final, _), _ = jax.lax.scan(
      step_fn,
      (init_C, init_D),
      scan_xs,
  )
  return C_final


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

  C = jnp.diag(c_diag)
  for i in range(1, b + 1):
    sub_diag_elements = U[: T - i, i - 1]
    C = C + jnp.diag(sub_diag_elements, k=-i)

  return C


def extract_bands_row_norm(C, b):
  """Extracts bands from a dense banded matrix C into params format."""
  T = C.shape[0]
  C_padded = jnp.pad(C, ((0, b), (0, 0)))

  def get_bands(j):
    return jax.lax.dynamic_slice(C_padded, (j, j), (b + 1, 1)).flatten()

  return jax.vmap(get_bands)(jnp.arange(T))


def evaluate(
    dataset_path, loss_fn, params, key
):
  """Evaluate the model on the dataset."""
  eval_dataset = tf.data.Dataset.load(dataset_path)
  eval_dataset = eval_dataset.batch(_EVAL_BATCH_SIZE.value)
  eval_dataset = eval_dataset.as_numpy_iterator()
  total_eval_loss = 0.0
  total_count = 0
  while 1:
    try:
      eval_batch = next(eval_dataset)
      total_count += len(eval_batch)
      total_eval_loss += loss_fn(
          params,
          key,
          eval_batch,
          is_training=False,
      ) * len(eval_batch)
    except StopIteration:
      break
  ave_eval_loss = total_eval_loss / total_count
  logging.info('total evaluation count: %d', total_count)
  logging.info('average evaluation loss: %f', ave_eval_loss)
  return ave_eval_loss


def save_params(params, checkpoint_path):
  def _disabled_seek(*_):
    raise AttributeError('seek() is disabled on this object.')

  with open(checkpoint_path, 'wb') as f:
    setattr(f, 'seek', _disabled_seek)
    np.savez_compressed(f, **params)
  logging.info('params saved to %s', checkpoint_path)


def load_params(ckpt_path):
  jax_params = collections.defaultdict(dict)
  with open(ckpt_path, 'rb') as f:
    # pylint: disable-next=g-unsafe-pickle-load
    params_npz = np.load(f, allow_pickle=True)
    for key, value in params_npz.items():
      jax_params[key] = value.item()
  logging.info('params loaded from %s', ckpt_path)
  return jax_params


def main(_):
  """Training loop."""
  coef_a_val = _COEF_A.value
  coef_b_val = _COEF_B.value

  if (
      _COEF_PATH.value
      or _COEF_C.value is not None
      or (_COEF_A.value is not None and _COEF_B.value is not None)
      or (_COEF_U.value is not None and _COEF_V.value is not None)
      or _USE_RMSE_TOEPLITZ.value
  ):
    if flags.FLAGS.batch_l2_norm_clip == 0.0:
      flags.FLAGS.batch_l2_norm_clip = 1.0
      logging.info(
          'Defaulting batch_l2_norm_clip to 1.0 when coef_path, coef_c, or'
          ' two-param coefficients are provided or use_rmse_toeplitz is True.'
      )

  # Model initialization.
  key = jax.random.PRNGKey(_SEED.value)
  key, subkey = jax.random.split(key)
  loss_fn = bert_model.tiny_bert_loss_fn
  params = bert_model.tiny_bert_forward.init(
      subkey,
      jnp.zeros((1, 1), dtype=jnp.int64),
      jnp.zeros((1, 1), dtype=jnp.int64),
  )
  if _INITIAL_CHECKPOINT_PATH.value:
    params = load_params(_INITIAL_CHECKPOINT_PATH.value)
    logging.info('loading from checkpoint: %s', _INITIAL_CHECKPOINT_PATH.value)

  # Noise generation
  if (
      _NOISE_MULTIPLIER.value > 0.0
      or _COEF_PATH.value
      or _COEF_C.value is not None
      or (_COEF_A.value is not None and _COEF_B.value is not None)
      or (_COEF_U.value is not None and _COEF_V.value is not None)
      or _USE_RMSE_TOEPLITZ.value
  ):
    if _BATCH_L2_NORM_CLIP.value <= 0.0:
      raise ValueError(
          'batch_l2_norm_clip must be > 0 if noise_multiplier > 0, coef_path'
          ' or coef_c is provided or use_rmse_toeplitz is True.'
      )

    stddev_value = (
        _NOISE_MULTIPLIER.value * _BATCH_L2_NORM_CLIP.value / _BATCH_SIZE.value
    )

    if _MATRIX_FAMILY.value == 'row_norm':
      if not _COEF_PATH.value:
        raise ValueError('row_norm requires coef_path')
      with open(_COEF_PATH.value, 'rb') as f:
        z_params = jnp.array(np.load(f))
      logging.info('Loaded z from %s', _COEF_PATH.value)

      C = generate_C_row_norm(z_params, _MIN_SEP.value - 1, _ITERATIONS.value)
      extracted_bands = extract_bands_row_norm(C, _MIN_SEP.value - 1)
      strategy = banded.ColumnNormalizedBanded(params=extracted_bands)
      noising_matrix = strategy.inverse_as_streaming_matrix()

    elif _MATRIX_FAMILY.value == 'banded_opt':
      if not _COEF_PATH.value:
        raise ValueError('banded_opt requires coef_path')
      with open(_COEF_PATH.value, 'rb') as f:
        V_params = jnp.array(np.load(f))
      logging.info('Loaded V from %s', _COEF_PATH.value)

      C = generate_C_banded_opt(V_params, _MIN_SEP.value - 1, _ITERATIONS.value)
      extracted_bands = extract_bands_row_norm(C, _MIN_SEP.value - 1)
      strategy = banded.ColumnNormalizedBanded(params=extracted_bands)
      noising_matrix = strategy.inverse_as_streaming_matrix()

    elif _MATRIX_FAMILY.value == 'banded_X':
      if not _COEF_PATH.value:
        raise ValueError('banded_X requires coef_path')
      with open(_COEF_PATH.value, 'rb') as f:
        C = jnp.array(np.load(f))
      logging.info('Loaded C from %s', _COEF_PATH.value)

      extracted_bands = extract_bands_row_norm(C, _MIN_SEP.value - 1)
      strategy = banded.ColumnNormalizedBanded(params=extracted_bands)
      noising_matrix = strategy.inverse_as_streaming_matrix()

    elif _MATRIX_FAMILY.value == 'nc_plus_c':
      if not _PRECONDITIONER_PATH.value:
        raise ValueError('nc_plus_c requires preconditioner_path')
      if _COEF_C.value is None:
        raise ValueError('nc_plus_c requires coef_c')

      with open(_PRECONDITIONER_PATH.value, 'rb') as f:
        H = np.load(f)

      H = np.sqrt(np.abs(H))
      H = H / np.max(np.abs(H))
      H = jnp.array(H)

      A = noisecurve_factorizer.build_workload(
          _LEARNING_RATE.value, _ITERATIONS.value, H
      )
      params_star = noisecurve_factorizer.optimize_noisecurve_banded(
          _ITERATIONS.value, _MIN_SEP.value, A, max_optimizer_steps=30
      )

      C_star_elements = params_star / jnp.linalg.norm(
          params_star, axis=1, keepdims=True
      )

      c = _COEF_C.value
      params_C = c * C_star_elements
      params_C = params_C.at[:, 0].add(1 - c)

      strategy = banded.ColumnNormalizedBanded(params=params_C)
      noising_matrix = strategy.inverse_as_streaming_matrix()

    elif _MATRIX_FAMILY.value == 'toep_Cinv':
      if _COEF_U.value is not None and _COEF_V.value is not None:
        u = _COEF_U.value
        v = _COEF_V.value
        if u + v <= 1.0:
          coef_a_val = u
          coef_b_val = v
        else:
          coef_a_val = 1.0 - u
          coef_b_val = 1.0 - v
      elif _COEF_A.value is not None and _COEF_B.value is not None:
        coef_a_val = _COEF_A.value
        coef_b_val = _COEF_B.value
      else:
        raise ValueError(
            'toep_Cinv requires (coef_u, coef_v) or (coef_a, coef_b)'
        )

      strategy = matrix_strategy.TwoParamCinvToeplitz(
          T=_ITERATIONS.value,
          init_a=coef_a_val,
          init_b=coef_b_val,
      )
      noising_matrix = strategy.build_normalized_noising_matrix(
          jnp.array([coef_a_val, coef_b_val])
      )
    elif _MATRIX_FAMILY.value == 'dense_toep':
      if _COEF_U.value is not None and _COEF_V.value is not None:
        u = _COEF_U.value
        v = _COEF_V.value
        if u + v <= 1.0:
          coef_a_val = u
          coef_b_val = v
        else:
          coef_a_val = 1.0 - u
          coef_b_val = 1.0 - v
      elif _COEF_A.value is not None and _COEF_B.value is not None:
        coef_a_val = _COEF_A.value
        coef_b_val = _COEF_B.value
      else:
        raise ValueError(
            'dense_toep requires (coef_u, coef_v) or (coef_a, coef_b)'
        )

      strategy = matrix_strategy.TwoParamDenseToeplitz(
          T=_ITERATIONS.value,
          init_a=coef_a_val,
          init_b=coef_b_val,
      )
      noising_matrix = strategy.build_normalized_noising_matrix(
          jnp.array([coef_a_val, coef_b_val])
      )
    elif _MATRIX_FAMILY.value == 'blt':
      if (
          _COEF_BUF_DECAY_0.value is not None
          and _COEF_BUF_DECAY_1.value is not None
          and _COEF_OUTPUT_SCALE_0.value is not None
          and _COEF_OUTPUT_SCALE_1.value is not None
      ):
        buf_decay = [_COEF_BUF_DECAY_0.value, _COEF_BUF_DECAY_1.value]
        output_scale = [_COEF_OUTPUT_SCALE_0.value, _COEF_OUTPUT_SCALE_1.value]
      else:
        raise ValueError(
            'blt requires coef_buf_decay_0, coef_buf_decay_1,'
            ' coef_output_scale_0, coef_output_scale_1'
        )

      strategy = matrix_strategy.BLTStrategy(
          T=_ITERATIONS.value,
          num_buffers=2,
      )
      noising_matrix = strategy.build_normalized_noising_matrix(
          jnp.concatenate([jnp.array(buf_decay), jnp.array(output_scale)])
      )
    else:  # toeplitz
      if _COEF_PATH.value:  # DP-MF
        with open(_COEF_PATH.value, 'rb') as f:
          loaded_params = jnp.array(np.load(f))

        if loaded_params.ndim == 2:
          logging.warning(
              'Loaded 2D params of shape %s for toeplitz, taking first row.',
              loaded_params.shape,
          )
          loaded_params = loaded_params[0]

        if loaded_params.size == _MIN_SEP.value - 1:
          coef = jnp.concatenate([jnp.array([1.0]), loaded_params])
          logging.info(
              'Loaded params of size %d, prepended 1.0 to form coef of size %d'
              ' from %s',
              loaded_params.size,
              coef.size,
              _COEF_PATH.value,
          )
        elif loaded_params.size == _MIN_SEP.value:
          coef = loaded_params
          logging.info(
              'Loaded coef of size %d from %s', coef.size, _COEF_PATH.value
          )
        else:
          raise ValueError(
              f'Loaded parameters size {loaded_params.size} does not match'
              f' expected size {_MIN_SEP.value} or {_MIN_SEP.value - 1}'
          )
      elif _COEF_C.value is not None:
        coef = jnp.power(
            _COEF_C.value, jnp.arange(_MIN_SEP.value, dtype=jnp.float32)
        )
        logging.info('Loaded SingleParamToeplitz coef with c=%s', _COEF_C.value)
      elif _COEF_A.value is not None and _COEF_B.value is not None:
        if _MIN_SEP.value == 1:
          coef = jnp.array([1.0])
        elif _MIN_SEP.value == 2:
          coef = jnp.array([1.0, _COEF_A.value])
        else:
          b_part = jnp.power(
              _COEF_B.value,
              jnp.arange(1, _MIN_SEP.value - 1, dtype=jnp.float32),
          )
          coef = jnp.concatenate([jnp.array([1.0, _COEF_A.value]), b_part])
        logging.info(
            'Loaded TwoParamToeplitz coef with a=%s, b=%s',
            _COEF_A.value,
            _COEF_B.value,
        )
      elif _USE_RMSE_TOEPLITZ.value:
        coef = toeplitz.optimize_banded_toeplitz(
            n=_ITERATIONS.value,
            bands=_MIN_SEP.value,
        )
        logging.info('RMSE Toeplitz coefficients generated.')
      else:  # DP-SGD
        coef = jnp.array([1.0])

      noising_matrix = toeplitz.inverse_as_streaming_matrix(
          coef, _ITERATIONS.value
      )
    key, subkey = jax.random.split(key)
    correlated_noise = noise_addition.matrix_factorization_privatizer(
        noising_matrix,
        stddev=stddev_value,
        prng_key=subkey,
        dtype=jnp.float32,
    )
    noise_state = correlated_noise.init(params)

  else:
    noise_state = None
    correlated_noise = noise_addition.gaussian_privatizer(stddev=0.0)

  # Dataset loading.
  data_name = _DATASET_PATH.value.split('/')[-1]
  run_name = _RUN_NAME.value or os.environ.get(
      'XMANAGER_WORK_UNIT_ID', 'default_run'
  )
  hparams_str = (
      f'{run_name},dataset={data_name},optimizer={_OPTIMIZER.value},'
      f'lr={_LEARNING_RATE.value},bs={_BATCH_SIZE.value},'
      f'iter={_ITERATIONS.value},seed={_SEED.value}'
  )
  if (
      _NOISE_MULTIPLIER.value > 0.0
      or _COEF_PATH.value
      or _COEF_C.value is not None
      or (_COEF_A.value is not None and _COEF_B.value is not None)
      or _USE_RMSE_TOEPLITZ.value
  ):
    hparams_str += f',bclip={_BATCH_L2_NORM_CLIP.value},ms={_MIN_SEP.value},nm={_NOISE_MULTIPLIER.value}'
    if _COEF_PATH.value:
      hparams_str += f',coef={os.path.basename(_COEF_PATH.value)}'
    elif _COEF_C.value is not None:
      hparams_str += f',coef=c_{_COEF_C.value}'
    elif _COEF_A.value is not None and _COEF_B.value is not None:
      hparams_str += f',coef=a_{_COEF_A.value}_b_{_COEF_B.value}'
    elif _USE_RMSE_TOEPLITZ.value:
      hparams_str += ',coef=rmse_toeplitz'

  try:
    # Folder for final checkpoint and preconditioner.
    if _ROOT_OUTPUT_DIR.value is None:
      raise ValueError('root_output_dir flag must be set.')
    checkpoint_folder = os.path.join(_ROOT_OUTPUT_DIR.value, 'checkpoints')
    if not os.path.exists(checkpoint_folder):
      os.makedirs(checkpoint_folder)
    checkpoint_path = os.path.join(checkpoint_folder, hparams_str + '.npz')
    del checkpoint_path

    train_dataset = tf.data.Dataset.load(_DATASET_PATH.value)
    if _SHUFFLE.value:
      train_dataset = train_dataset.shuffle(
          len(train_dataset),
          reshuffle_each_iteration=False,
          seed=_DATA_SEED.value,
      )
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(_BATCH_SIZE.value)
    train_dataset = train_dataset.as_numpy_iterator()

    # Optimizer initialization.
    if _OPTIMIZER.value == 'sgd':
      optimizer = optax.sgd(learning_rate=_LEARNING_RATE.value)
      logging.info(
          'Using sgd optimizer with learning rate %s', _LEARNING_RATE.value
      )
    elif _OPTIMIZER.value == 'sgdm':
      optimizer = optax.sgd(
          learning_rate=_LEARNING_RATE.value,
          momentum=0.9,
      )
      logging.info(
          'Using sgd optimizer with learning rate %s and momentum 0.9',
          _LEARNING_RATE.value,
      )
    elif _OPTIMIZER.value == 'adamw':
      optimizer = optax.adamw(
          learning_rate=_LEARNING_RATE.value, eps=_STABILITY_EPS.value
      )
      logging.info(
          'Using adamw optimizer with learning rate %s',
          _LEARNING_RATE.value,
      )
    else:
      raise ValueError(
          'Unsupported/unknown optimizer: {}'.format(_OPTIMIZER.value)
      )

    # Gradient computation.
    @jax.jit
    def grad_fn(params, key, batch):
      return jax.grad(loss_fn)(params, key, batch)

    def update(key, params, opt_state, batch, noise_state):
      key, subkey = jax.random.split(key)
      gradient = grad_fn(params, subkey, batch)
      g_norm = optax.global_norm(gradient)
      if _BATCH_L2_NORM_CLIP.value > 0.0:
        divisor = jnp.maximum(g_norm / _BATCH_L2_NORM_CLIP.value, 1.0)
        gradient = jax.tree.map(lambda t: t / divisor, gradient)

      if noise_state is not None:
        noise, noise_state = correlated_noise.update(gradient, noise_state)
        gradient = jax.tree.map(lambda x, y: x + y, gradient, noise)
      updates, opt_state = optimizer.update(gradient, opt_state, params)
      params = optax.apply_updates(params, updates)
      return (params, opt_state, key, noise_state, g_norm)

    opt_state = optimizer.init(params)
    itercount = itertools.count()

    logging.info('\nStarting training...')
    for _ in range(_ITERATIONS.value):
      iteration = next(itercount) + 1
      batch = next(train_dataset)
      params, opt_state, key, noise_state, g_norm = update(
          key, params, opt_state, batch, noise_state
      )
      key, subkey = jax.random.split(key)
      train_loss = loss_fn(
          params,
          subkey,
          batch,
      )
      # if train_loss > 100:  # End runs that diverge early.
      #   raise RuntimeError(f'Training diverged: train_loss={train_loss}')
      metrics = {
          'train_loss': float(train_loss),
          'grad_norm': float(g_norm),
      }
      if iteration % _VALID_STEPS.value == 0:
        ave_valid_loss = evaluate(
            _VALID_DATASET_PATH.value,
            loss_fn,
            params,
            key=jax.random.PRNGKey(_EVAL_SEED.value),
        )
        metrics['valid_loss'] = ave_valid_loss

      if (
          _PRECONDITIONER_SAVE_FREQUENCY.value > 0
          and (iteration % _PRECONDITIONER_SAVE_FREQUENCY.value == 0)
          and _OPTIMIZER.value == 'adamw'
      ):
        preconditioner_path = os.path.join(
            checkpoint_folder,
            f'preconditioner_{iteration}.npz',
        )
        del preconditioner_path
        # save_params(
        #     opt_state[0].nu,
        #     preconditioner_path,
        # )
      metrics['step'] = iteration

    # Validation loss.
    ave_test_loss = evaluate(
        _TEST_DATASET_PATH.value,
        loss_fn,
        params,
        key=jax.random.PRNGKey(_EVAL_SEED.value),
    )
    metrics = {
        'test_loss': ave_test_loss,
    }
    metrics['step'] = next(itercount) + 1
    logging.info('Test loss: %f', ave_test_loss)

  finally:
    pass

  # save_params(params, checkpoint_path)
  # if _OPTIMIZER.value == 'adamw':
  #   preconditioner_path = os.path.join(
  #       checkpoint_folder, f'preconditioner_{_ITERATIONS.value}.npz'
  #   )
  #   save_params(
  #       opt_state[0].nu,
  #       preconditioner_path,
  #   )


if __name__ == '__main__':
  app.run(main)
