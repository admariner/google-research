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

"""Algorithm 5: Approximate mu-GDP Verification.

Order Statistics Lower Bound (JAX/GPU version).
"""

# pylint: disable=invalid-name
import functools
import time
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import jax.scipy.stats as jax_stats
import numpy as np
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax

jax.config.update("jax_enable_x64", True)


_MU_INPUT = flags.DEFINE_float(
    "mu_input", 1.0, "Actual mechanism privacy parameter"
)
_N = flags.DEFINE_integer("n", 100000, "Sample size for verification")
_GAMMA = flags.DEFINE_float(
    "gamma", 1e-5, "Target failure probability of the statistical bound"
)
_TAU = flags.DEFINE_float("tau", 1e-9, "Tail probability for delta relaxation")
_NUM_SIMULATIONS = flags.DEFINE_integer(
    "num_simulations", 20000, "Number of MC runs for binary search solving q"
)
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")
_USE_SOLVED_Q = flags.DEFINE_boolean(
    "use_solved_q",
    False,
    "Whether to use solved q (True) or union bound q (False)",
)
_MU_MAX = flags.DEFINE_float("mu_max", 2.0, "Maximum mu to search in grid")
_MU_STEP = flags.DEFINE_float("mu_step", 0.05, "Step size for mu grid")


def G_mu(alpha, mu):
  """GDP trade-off function for Gaussian: G_mu(alpha)."""
  alpha = jnp.clip(alpha, 1e-15, 1.0 - 1e-15)
  val = -jax_stats.norm.ppf(alpha) - mu
  return jax_stats.norm.cdf(val)


def compute_delta(tau, mu_target):
  """Computes relaxation parameter delta = max(tau, G_mu(1 - tau, mu_target))."""
  z_tau_lower = jax_stats.norm.ppf(tau)
  g_val = jax_stats.norm.cdf(z_tau_lower - mu_target)
  return jnp.maximum(tau, g_val)


def _generate_L_sorted(key, mu_input, n):
  y = jax.random.normal(key, shape=(n,)) + mu_input
  L = y * mu_input - 0.5 * (mu_input**2)
  return jnp.sort(L, stable=False)


def _generate_L_prime_sorted(key, mu_input, n):
  y_prime = jax.random.normal(key, shape=(n,))
  L_prime = -(y_prime * mu_input - 0.5 * (mu_input**2))
  return jnp.sort(L_prime, stable=False)


# JIT helpers with static n to allow shape inferencing
generate_L_sorted = jax.jit(_generate_L_sorted, static_argnums=(2,))
generate_L_prime_sorted = jax.jit(_generate_L_prime_sorted, static_argnums=(2,))


# JIT compile the core logic of solve_q step if possible
def solve_q_step(a, b, mid, U_sorted, target_prob):
  ell = tfp.math.betaincinv(a, b, mid)
  event_occurred = jnp.any(U_sorted < ell, axis=1)
  prob = jnp.mean(event_occurred)
  return prob < target_prob


def solve_q(n, gamma, num_simulations=20000, max_steps=40, seed=42):
  """Solves for scalar value q s.t. P(M_n^+ < q) <= gamma / 2.

  Args:
    n: Sample size.
    gamma: Failure probability.
    num_simulations: Number of Monte Carlo simulations.
    max_steps: Maximum binary search steps.
    seed: Random seed.

  Returns:
    Scalar value q.
  """
  key = jax.random.PRNGKey(seed)

  # 1. Sample and sort Uniform(0, 1) once
  U = jax.random.uniform(
      key, shape=(num_simulations, n), minval=0.0, maxval=1.0
  )
  U_sorted = jnp.sort(U, axis=1)

  i = jnp.arange(1, n + 1, dtype=jnp.float64)
  a = i
  b = n - i + 1
  target_prob = gamma / 2.0

  low = 1e-15
  high = 1.0 - 1e-15

  # Binary search loop (kept in Python for simplicity, JITing step)
  for _ in range(max_steps):
    mid = 0.5 * (low + high)
    # Fixed steps Python loop is fine if inner part is fast.
    ell = tfp.math.betaincinv(a, b, mid)
    event_occurred = jnp.any(U_sorted < ell, axis=1)
    prob = jnp.mean(event_occurred)

    # Manual boolean indexing/assignment since we are in standard python loop
    if prob < target_prob:
      low = mid
    else:
      high = mid

  return float(low)


def compute_ell_at_k(n, q, k):
  """Computes ell_i = B_i^{-1}(q) for 1-based index i=k (0 if k=0)."""
  # k can be an array of indices.
  i_safe = jnp.maximum(1, k).astype(jnp.float64)
  a = i_safe
  b = n - i_safe + 1
  ell_val = tfp.math.betaincinv(a, b, q)
  return jnp.where(k == 0, 0.0, ell_val).astype(jnp.float64)


def verify_approx_gdp_algorithm5(
    key,
    mu_input,
    n,
    gamma,
    tau,
    mu_grid,
    q,
    max_log_points=1000,
):
  """Implements Algorithm 5 (JAX version) - Memory Optimized."""
  del gamma
  key_p, key_q = jax.random.split(key)

  # 2. Generate sorted privacy losses efficiently
  L_sorted = generate_L_sorted(key_p, mu_input, n)
  L_prime_sorted = generate_L_prime_sorted(key_q, mu_input, n)

  print(f"DEBUG: n={n}")
  print(f"DEBUG: L_sorted shape={L_sorted.shape}, dtype={L_sorted.dtype}")
  print(
      f"DEBUG: L_prime_sorted shape={L_prime_sorted.shape},"
      f" dtype={L_prime_sorted.dtype}"
  )

  num_devices = jax.local_device_count()
  # CHUNK SIZE TUNING KNOB: Smaller chunk_size uses less memory per batch step.
  chunk_size = 100000
  batch_step = num_devices * chunk_size

  print(f"DEBUG: n={n}")
  print(f"DEBUG: num_devices={num_devices}")
  print(f"DEBUG: chunk_size={chunk_size}")
  print(f"DEBUG: batch_step={batch_step}")
  print(f"DEBUG: L_sorted shape={L_sorted.shape}, dtype={L_sorted.dtype}")
  print(
      f"DEBUG: L_prime_sorted shape={L_prime_sorted.shape},"
      f" dtype={L_prime_sorted.dtype}"
  )

  # Pre-allocate JAX array for passed status
  passed_array = jnp.ones(len(mu_grid), dtype=jnp.bool_)

  # Stride for subsampling
  stride = max(1, n // (max_log_points // 2))

  subsampled_alpha = []
  subsampled_beta = []

  @jax.jit
  def check_failures_for_chunk_vmapped(beta_LB, alpha_LB, mu_grid):
    def check_single_mu(mu):
      delta = compute_delta(tau, mu)
      G_vals = G_mu(alpha_LB, mu)
      failures = beta_LB < (G_vals - delta)
      return ~jnp.any(failures)

    return jax.vmap(check_single_mu)(mu_grid)

  @functools.partial(
      jax.pmap, axis_name="devices", in_axes=(0, 0, None, None, None)
  )
  def process_and_verify_chunk(t_chunk, mask_chunk, L_s, L_p_s, mu_g):
    k_beta = jnp.searchsorted(L_s, t_chunk, side="right")
    beta_LB = compute_ell_at_k(n, q, k_beta)

    k_alpha = jnp.searchsorted(L_p_s, -t_chunk, side="right")
    alpha_LB = compute_ell_at_k(n, q, k_alpha)

    def check_single_mu(mu):
      delta = compute_delta(tau, mu)
      G_vals = G_mu(alpha_LB, mu)
      failures = beta_LB < (G_vals - delta)
      valid_failures = jnp.logical_and(failures, mask_chunk)
      return ~jnp.any(valid_failures)

    passed_chunk = jax.vmap(check_single_mu)(mu_g)

    return alpha_LB, beta_LB, passed_chunk

  # Helper to process thresholds
  def process_thresholds(thresholds_source, is_L_prime=False):
    del is_L_prime
    nonlocal passed_array
    for start_idx in range(0, n, batch_step):
      end_idx = min(start_idx + batch_step, n)
      valid_len = end_idx - start_idx

      t_batch_flat = thresholds_source[start_idx:end_idx]

      # Pad if remainder
      if valid_len < batch_step:
        pad_len = batch_step - valid_len
        t_batch_flat = jnp.pad(t_batch_flat, (0, pad_len), constant_values=0.0)

      t_batch = t_batch_flat.reshape((num_devices, chunk_size))

      # Create mask
      mask_flat = jnp.arange(batch_step) < valid_len
      mask_batch = mask_flat.reshape((num_devices, chunk_size))

      alpha_chunk, beta_chunk, passed_chunk_batch = process_and_verify_chunk(
          t_batch, mask_batch, L_sorted, L_prime_sorted, mu_grid
      )

      # Update passed array
      passed_batch_reduced = jnp.all(passed_chunk_batch, axis=0)
      passed_array = jnp.logical_and(passed_array, passed_batch_reduced)

      # Collect subsampled points
      alpha_flat = alpha_chunk.reshape(-1)
      beta_flat = beta_chunk.reshape(-1)

      valid_alpha = alpha_flat[:valid_len]
      valid_beta = beta_flat[:valid_len]

      indices = jnp.arange(
          0, valid_len, stride
      )  # This stride is for valid part of batch
      subsampled_alpha.append(valid_alpha[indices])
      subsampled_beta.append(valid_beta[indices])

      if not jnp.any(passed_array):
        print("All mu failed, exiting early.")
        return True  # Signal early exit
    return False

  print("Processing L as thresholds...")
  if not process_thresholds(L_sorted, is_L_prime=False):
    print("Processing L_prime as thresholds...")
    # Cannot delete L_sorted yet because process_thresholds for L_prime
    # still uses L_sorted for k_beta.
    process_thresholds(L_prime_sorted, is_L_prime=True)

  # Concatenate subsampled points
  alpha_LB_sub = jnp.concatenate(subsampled_alpha)
  beta_LB_sub = jnp.concatenate(subsampled_beta)

  # Sort them by alpha for a smooth curve
  sort_idx = jnp.argsort(alpha_LB_sub, stable=False)
  alpha_LB_sub = alpha_LB_sub[sort_idx]
  beta_LB_sub = beta_LB_sub[sort_idx]

  del L_sorted, L_prime_sorted

  return passed_array, alpha_LB_sub, beta_LB_sub


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  mu_input = jnp.float64(_MU_INPUT.value)
  n = _N.value
  gamma = jnp.float64(_GAMMA.value)
  tau = jnp.float64(_TAU.value)
  num_simulations = _NUM_SIMULATIONS.value
  seed = _SEED.value
  use_solved_q = _USE_SOLVED_Q.value
  mu_max = jnp.float64(_MU_MAX.value)
  mu_step = jnp.float64(_MU_STEP.value)

  print("JAX running on:", jax.devices()[0].device_kind)

  t0 = time.time()
  q_union_bound = gamma / (2.0 * n)
  if use_solved_q:
    print("Solving for q (JAX version)...")
    q = solve_q(n, gamma, num_simulations=num_simulations, seed=seed)
  else:
    q = float(q_union_bound)
  t1 = time.time()

  delta_input = compute_delta(tau, mu_input)

  print(f"q: {q:.6e} (took {t1 - t0:.2f}s)")
  print(f"Union Bound q: {q_union_bound:.6e}")
  print(f"delta: {float(delta_input):.2e}")

  mu_grid = jnp.arange(
      mu_input,
      mu_max + mu_step / 2.0,
      mu_step,
      dtype=jnp.float64,
  )

  key = jax.random.PRNGKey(seed)
  t2 = time.time()
  passed_grid, alpha_LB, beta_LB = verify_approx_gdp_algorithm5(
      key, mu_input, n, gamma, tau, mu_grid, q=q
  )
  # Block until computation is done to measure time correctly with JAX
  passed_grid.block_until_ready()
  t3 = time.time()

  print(f"Verification took {t3 - t2:.2f}s")

  num_points = len(alpha_LB)
  max_log_points = 10000
  if num_points > max_log_points:
    indices = np.linspace(0, num_points - 1, max_log_points, dtype=int)
    alpha_sub = alpha_LB[indices]
    beta_sub = beta_LB[indices]
    step_indices = indices
  else:
    alpha_sub = alpha_LB
    beta_sub = beta_LB
    step_indices = range(num_points)

  for i, (alpha, beta) in zip(step_indices, zip(alpha_sub, beta_sub)):
    # Should write instead of print in the future.
    print({
        "step": int(i),
        "alpha_LB": float(alpha),
        "beta_LB": float(beta),
    })

  valid_mu = mu_grid[passed_grid]
  smallest_passed_mu = float(valid_mu[0]) if len(valid_mu) > 0 else None

  # Should write instead of print in the future.
  print({
      "n": int(n),
      "mu_input": float(mu_input),
      "smallest_passed_mu": (
          float(smallest_passed_mu) if smallest_passed_mu is not None else -1.0
      ),
      "passed_any": bool(smallest_passed_mu is not None),
      "gamma": float(gamma),
      "tau": float(tau),
      "q": float(q),
      "time_seconds": float(t3 - t2),
  })

  for step_idx, (mu_target, passed) in enumerate(zip(mu_grid, passed_grid)):
    delta = compute_delta(tau, mu_target)
    # Should write instead of print in the future.
    print({
        "step": int(step_idx),
        "n": int(n),
        "mu_input": float(mu_input),
        "mu_target": float(mu_target),
        "passed": bool(passed),
        "delta": float(delta),
        "smallest_passed_mu": (
            float(smallest_passed_mu)
            if smallest_passed_mu is not None
            else -1.0
        ),
        "gamma": float(gamma),
        "tau": float(tau),
    })

  print("=" * 65)
  for mu_target, passed in zip(mu_grid, passed_grid):
    status = "PASSED" if passed else "FAILED"
    print(f"Target mu = {mu_target:.3f} | {status}")
  print("-" * 65)
  if smallest_passed_mu is not None:
    print(f"Smallest target mu that passed: {smallest_passed_mu:.3f}")
  else:
    print("None passed.")
  print("=" * 65)


if __name__ == "__main__":
  app.run(main)
