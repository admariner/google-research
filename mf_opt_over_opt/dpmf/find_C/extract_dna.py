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

"""Training loop on TinyBERT with gradient behavior (DNA) extraction.

Non-private training with gradient clipping and DNA extraction.
"""

# pylint: disable=invalid-name
import collections
import getpass
import itertools
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf

from mf_opt_over_opt.dpmf.train import bert_model

try:
  tf.config.set_visible_devices([], 'GPU')
except Exception:  # pylint: disable=broad-exception-caught
  pass

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
    'adamw',
    ['adamw', 'sgd', 'sgdm'],
    'Optimizer to use.',
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 0.01, 'Learning rate for training'
)
_STABILITY_EPS = flags.DEFINE_float('stability_eps', 1e-8, 'Stability epsilon')

# Gradient Clipping flag.
_BATCH_L2_NORM_CLIP = flags.DEFINE_float(
    'batch_l2_norm_clip', 1.0, 'L2 norm clip for batch gradients.'
)

# Seeds.
_SEED = flags.DEFINE_integer(
    'seed',
    0,
    'Seed for jax PRNG, affecting model training',
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

# DNA extraction flags.
_NUM_SAMPLED_PARAMS = flags.DEFINE_integer(
    'num_sampled_params',
    10000,
    'Number of parameters sampled evenly to extract grad DNA.',
)
_DNA_BETA = flags.DEFINE_float(
    'dna_beta',
    0.99,
    'Beta decay factor for exponential moving average in DNA extraction.',
)
_DNA_GRAD_SOURCE = flags.DEFINE_enum(
    'dna_grad_source',
    'clipped',
    ['raw', 'clipped'],
    'Which gradient to use for DNA extraction: raw (before clip) or clipped'
    ' (after clip).',
)
_DNA_WINDOW_SIZE = flags.DEFINE_integer(
    'dna_window_size',
    50,
    'Number of iterations per temporal window for saving time-varying DNA.',
)
_DNA_RESET_WINDOWS = flags.DEFINE_bool(
    'dna_reset_windows',
    True,
    'Whether to reset EMA statistics at the start of each new temporal window.',
)


def update_dna(
    mu,
    v,
    c,
    g_prev,
    target_grad,
    sampled_indices,
    beta = 0.99,
):
  """Updates exponential moving averages for mean, variance, and lag-1 covariance."""
  flat_grad, _ = jax.flatten_util.ravel_pytree(target_grad)
  grad_k = flat_grad[sampled_indices]
  new_mu = beta * mu + (1.0 - beta) * grad_k
  new_v = beta * v + (1.0 - beta) * (grad_k**2)
  new_c = beta * c + (1.0 - beta) * (grad_k * g_prev)
  new_g_prev = grad_k
  return new_mu, new_v, new_c, new_g_prev


def evaluate(
    dataset_path, loss_fn, params, key
):
  """Evaluate the model on the dataset."""

  @jax.jit
  def eval_batch_fn(p, k, b):
    return loss_fn(p, k, b, is_training=False)

  eval_dataset = tf.data.Dataset.load(dataset_path)
  eval_dataset = eval_dataset.batch(_EVAL_BATCH_SIZE.value)
  eval_dataset = eval_dataset.as_numpy_iterator()
  total_eval_loss = 0.0
  total_count = 0
  while 1:
    try:
      eval_batch = next(eval_dataset)
      total_count += len(eval_batch)
      loss_val = eval_batch_fn(params, key, eval_batch)
      total_eval_loss += float(loss_val) * len(eval_batch)
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
  """Training loop with DNA extraction."""
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

  # Step 1: Initialization for DNA extraction.
  flat_params, _ = jax.flatten_util.ravel_pytree(params)
  total_params = flat_params.size
  k_params = min(_NUM_SAMPLED_PARAMS.value, total_params)
  sampled_indices = jnp.array(
      np.linspace(0, total_params - 1, num=k_params, dtype=np.int32)
  )
  logging.info(
      'Sampling %d indices evenly across %d total model parameters for DNA'
      ' extraction.',
      k_params,
      total_params,
  )

  mu_dna = jnp.zeros((k_params,), dtype=jnp.float32)
  v_dna = jnp.zeros((k_params,), dtype=jnp.float32)
  c_dna = jnp.zeros((k_params,), dtype=jnp.float32)
  g_prev_dna = jnp.zeros((k_params,), dtype=jnp.float32)

  # Dataset loading.
  data_name = _DATASET_PATH.value.split('/')[-1]
  run_name = _RUN_NAME.value or os.environ.get(
      'XMANAGER_WORK_UNIT_ID', 'default_run'
  )
  hparams_str = (
      f'{run_name},dataset={data_name},optimizer={_OPTIMIZER.value},'
      f'lr={_LEARNING_RATE.value},bs={_BATCH_SIZE.value},'
      f'iter={_ITERATIONS.value},seed={_SEED.value},'
      f'bclip={_BATCH_L2_NORM_CLIP.value}'
  )

  try:
    # Folder for final checkpoint and DNA.
    root_dir = (
        _ROOT_OUTPUT_DIR.value or f'/tmp/mf_opt_over_opt_{getpass.getuser()}'
    )
    checkpoint_folder = os.path.join(root_dir, 'checkpoints')
    if not os.path.exists(checkpoint_folder):
      os.makedirs(checkpoint_folder)

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

    # Gradient & Loss computation.
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def update(key, params, opt_state, batch, mu, v, c, g_prev):
      key, subkey = jax.random.split(key)
      loss, raw_gradient = value_and_grad_fn(params, subkey, batch)
      g_norm = optax.global_norm(raw_gradient)

      gradient = raw_gradient
      if _BATCH_L2_NORM_CLIP.value > 0.0:
        divisor = jnp.maximum(g_norm / _BATCH_L2_NORM_CLIP.value, 1.0)
        gradient = jax.tree.map(lambda t: t / divisor, gradient)
      clipped_gradient = gradient

      if _DNA_GRAD_SOURCE.value == 'raw':
        target_grad = raw_gradient
      else:
        target_grad = clipped_gradient

      mu, v, c, g_prev = update_dna(
          mu, v, c, g_prev, target_grad, sampled_indices, _DNA_BETA.value
      )

      updates, opt_state = optimizer.update(gradient, opt_state, params)
      params = optax.apply_updates(params, updates)
      return (params, opt_state, key, g_norm, loss, mu, v, c, g_prev)

    opt_state = optimizer.init(params)
    itercount = itertools.count()

    mu_history = []
    sigma_history = []
    rho_history = []
    h_history = []
    window_steps = []

    logging.info('\nStarting training with DNA extraction...')
    for _ in range(_ITERATIONS.value):
      step_idx = next(itercount)
      iteration = step_idx + 1
      batch = next(train_dataset)
      (
          params,
          opt_state,
          key,
          g_norm,
          train_loss,
          mu_dna,
          v_dna,
          c_dna,
          g_prev_dna,
      ) = update(
          key,
          params,
          opt_state,
          batch,
          mu_dna,
          v_dna,
          c_dna,
          g_prev_dna,
      )
      metrics = {
          'train_loss': train_loss,
          'grad_norm': g_norm,
      }
      if iteration % _VALID_STEPS.value == 0:
        ave_valid_loss = evaluate(
            _VALID_DATASET_PATH.value,
            loss_fn,
            params,
            key=jax.random.PRNGKey(_EVAL_SEED.value),
        )
        metrics['valid_loss'] = ave_valid_loss
        cur_sigma2 = jnp.maximum(v_dna - (mu_dna**2), 1e-8)
        cur_sigma = jnp.sqrt(cur_sigma2)
        cur_covar = c_dna - (mu_dna**2)
        cur_rho = jnp.clip(cur_covar / cur_sigma2, -0.99, 0.99)
        metrics['dna_mean_rho'] = float(jnp.mean(cur_rho))
        metrics['dna_mean_sigma'] = float(jnp.mean(cur_sigma))

      metrics['step'] = iteration

      if _DNA_WINDOW_SIZE.value > 0 and (
          iteration % _DNA_WINDOW_SIZE.value == 0
          or iteration == _ITERATIONS.value
      ):
        cur_sigma2 = jnp.maximum(v_dna - (mu_dna**2), 1e-20)
        cur_sigma = jnp.sqrt(cur_sigma2)
        cur_covar = c_dna - (mu_dna**2)
        cur_rho = jnp.clip(cur_covar / cur_sigma2, -0.99, 0.99)
        if _OPTIMIZER.value == 'adamw' and hasattr(opt_state[0], 'nu'):
          flat_nu, _ = jax.flatten_util.ravel_pytree(opt_state[0].nu)
          cur_h = jnp.sqrt(flat_nu[sampled_indices])
        else:
          cur_h = cur_sigma
        mu_history.append(np.array(mu_dna))
        sigma_history.append(np.array(cur_sigma))
        rho_history.append(np.array(cur_rho))
        h_history.append(np.array(cur_h))
        window_steps.append(iteration)
        if _DNA_RESET_WINDOWS.value and iteration < _ITERATIONS.value:
          mu_dna = jnp.zeros_like(mu_dna)
          v_dna = jnp.zeros_like(v_dna)
          c_dna = jnp.zeros_like(c_dna)

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

    # Step 3: Post-Training (Extract the 'DNA'):
    if not mu_history:
      final_mu = mu_dna
      final_sigma2 = jnp.maximum(v_dna - (mu_dna**2), 1e-20)
      final_sigma = jnp.sqrt(final_sigma2)
      covar = c_dna - (mu_dna**2)
      final_rho = jnp.clip(covar / final_sigma2, -0.99, 0.99)

      if _OPTIMIZER.value == 'adamw' and hasattr(opt_state[0], 'nu'):
        flat_nu, _ = jax.flatten_util.ravel_pytree(opt_state[0].nu)
        final_h = jnp.sqrt(flat_nu[sampled_indices])
      else:
        logging.info(
            'Optimizer is not adamw or nu state not found; using final_sigma as'
            ' curvature proxy.'
        )
        final_h = final_sigma
      mu_history.append(np.array(final_mu))
      sigma_history.append(np.array(final_sigma))
      rho_history.append(np.array(final_rho))
      h_history.append(np.array(final_h))
      window_steps.append(_ITERATIONS.value)

    stacked_mu = np.stack(mu_history, axis=0)
    stacked_sigma = np.stack(sigma_history, axis=0)
    stacked_rho = np.stack(rho_history, axis=0)
    stacked_h = np.stack(h_history, axis=0)
    stacked_window_steps = np.array(window_steps, dtype=np.int32)

    logging.info('=== Extracted Grad DNA Summary ===')
    logging.info(
        'Mean(mu): %f, Std(mu): %f',
        float(np.mean(stacked_mu)),
        float(np.std(stacked_mu)),
    )
    logging.info(
        'Mean(sigma): %f, Std(sigma): %f',
        float(np.mean(stacked_sigma)),
        float(np.std(stacked_sigma)),
    )
    logging.info(
        'Mean(rho): %f, Std(rho): %f',
        float(np.mean(stacked_rho)),
        float(np.std(stacked_rho)),
    )
    logging.info(
        'Mean(h): %f, Std(h): %f',
        float(np.mean(stacked_h)),
        float(np.std(stacked_h)),
    )

    dna_path = os.path.join(checkpoint_folder, f'dna_{hparams_str}.npz')
    save_params(
        {
            'mu': stacked_mu,
            'sigma': stacked_sigma,
            'rho': stacked_rho,
            'h': stacked_h,
            'window_steps': stacked_window_steps,
            'sampled_indices': np.array(sampled_indices),
        },
        dna_path,
    )
    logging.info('Saved extracted DNA to %s', dna_path)

    dna_summary_metrics = {
        'step': _ITERATIONS.value,
        'dna_final_mean_mu': float(np.mean(stacked_mu)),
        'dna_final_mean_sigma': float(np.mean(stacked_sigma)),
        'dna_final_mean_rho': float(np.mean(stacked_rho)),
        'dna_final_mean_h': float(np.mean(stacked_h)),
    }
    logging.info('DNA summary metrics: %s', dna_summary_metrics)

  finally:
    pass


if __name__ == '__main__':
  app.run(main)
