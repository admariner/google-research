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

"""Algorithm 4.9: Hybrid (eps, delta) and Clopper-Pearson ROC mu-GDP Verification.

JAX/GPU version.
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
from tensorflow_probability.substrates import jax as tfp

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)


_MU_INPUT = flags.DEFINE_float(
    "mu_input", 1.0, "Actual mechanism privacy parameter"
)
_N = flags.DEFINE_integer("n", 100000, "Sample size for verification")
_GAMMA = flags.DEFINE_float(
    "gamma", 1e-5, "Target failure probability of the statistical bound"
)
_DELTA_TARGET = flags.DEFINE_float(
    "delta_target",
    1e-5,
    "Target delta for approximate GDP and (eps, delta) accounting",
)
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")
_MU_MAX = flags.DEFINE_float("mu_max", 2.0, "Maximum mu to search in grid")
_MU_STEP = flags.DEFINE_float("mu_step", 0.05, "Step size for mu grid")


def G_mu(alpha, mu):
  """GDP trade-off function for Gaussian: G_mu(alpha)."""
  alpha = jnp.clip(alpha, 1e-15, 1.0 - 1e-15)
  val = -jax_stats.norm.ppf(alpha) - mu
  return jax_stats.norm.cdf(val)


def delta_gdp(mu, eps):
  """Theoretical (eps, delta)-DP guarantee for Gaussian mechanism mu-GDP."""
  t1 = jax_stats.norm.cdf(-eps / mu + mu / 2.0)
  log_t2 = eps + jax_stats.norm.logcdf(-eps / mu - mu / 2.0)
  return t1 - jnp.exp(log_t2)


def compute_eps_from_delta_jax(mu, delta):
  """Computes eps* s.t. delta_gdp(mu, eps*) = delta using Newton-Raphson."""
  z = jax_stats.norm.ppf(delta)
  eps_init = jnp.maximum(0.0, mu * (mu / 2.0 - z))

  def step(i, eps):
    del i
    d_val = delta_gdp(mu, eps)
    diff = d_val - delta
    log_deriv = eps + jax_stats.norm.logcdf(-eps / mu - mu / 2.0)
    deriv = -jnp.exp(log_deriv)
    eps_new = eps - diff / deriv
    return jnp.maximum(0.0, eps_new)

  return jax.lax.fori_loop(0, 15, step, eps_init)


def compute_alpha_star_from_delta_jax(mu, delta):
  """Computes divide point alpha* and tangent eps* corresponding to delta."""
  eps_star = compute_eps_from_delta_jax(mu, delta)
  alpha_star = jax_stats.norm.cdf(-eps_star / mu - mu / 2.0)
  return alpha_star, eps_star


def generate_L_sorted(seed, mu_input, n):
  mu_cpu = float(mu_input)
  rng = np.random.default_rng(seed)
  y = rng.normal(loc=mu_cpu, scale=1.0, size=(n,))
  L = y * mu_cpu - 0.5 * (mu_cpu**2)
  L.sort()
  return L


def generate_L_prime_sorted(seed, mu_input, n):
  mu_cpu = float(mu_input)
  rng = np.random.default_rng(seed)
  y_prime = rng.normal(loc=0.0, scale=1.0, size=(n,))
  L_prime = y_prime * mu_cpu - 0.5 * (mu_cpu**2)
  L_prime.sort()
  return L_prime


def compute_cp_at_k(n, epsilon, k):
  """Computes CP(k) = B_k^{-1}(epsilon) for 1-based index i=k (0 if k=0)."""
  i_safe = jnp.maximum(1, k).astype(jnp.float64)
  a = i_safe
  b = n - i_safe + 1
  ell_val = tfp.math.betaincinv(a, b, epsilon)
  return jnp.where(k == 0, 0.0, ell_val).astype(jnp.float64)


def verify_tail_eps_delta_batch(
    L, L_prime, mu_grid, eps_grid, gamma_tail, delta_target, n
):
  """Evaluates tail (eps, delta) upper bounds across mu_grid."""
  passed_tail_list = []
  for mu, eps_star in zip(mu_grid, eps_grid):
    del mu
    # Positive hockey loss E_P[(1 - exp(eps_star - L))_+]
    loss_P = np.maximum(0.0, 1.0 - np.exp(np.minimum(0.0, float(eps_star) - L)))
    mean_P = np.mean(loss_P)
    var_P = np.var(loss_P, ddof=1)

    # Negative hockey loss E_Q[(1 - exp(eps_star + L_prime))_+]
    loss_Q = np.maximum(
        0.0, 1.0 - np.exp(np.minimum(0.0, float(eps_star) + L_prime))
    )
    mean_Q = np.mean(loss_Q)
    var_Q = np.var(loss_Q, ddof=1)

    log_term = np.log(4.0 / float(gamma_tail))
    ub_P = mean_P + np.sqrt(2.0 * var_P * log_term / n) + 3.0 * log_term / n
    ub_Q = mean_Q + np.sqrt(2.0 * var_Q * log_term / n) + 3.0 * log_term / n

    allowed = float(delta_target) + float(delta_target)
    passed = (ub_P <= allowed) and (ub_Q <= allowed)
    passed_tail_list.append(passed)

  return jnp.array(passed_tail_list, dtype=jnp.bool_)


def verify_approx_gdp_algorithm4p9(
    seed,
    mu_input,
    n,
    gamma,
    delta_target,
    mu_grid,
    epsilon_cp,
    max_log_points=1000,
):
  """Implements Algorithm 4.9 (JAX version) - Reverse Delta-to-Alpha* Hybrid GDP Verification."""
  seed_p = seed
  seed_q = seed + 12345

  L_sorted = generate_L_sorted(seed_p, mu_input, n)
  L_prime_sorted = generate_L_prime_sorted(seed_q, mu_input, n)

  # Compute alpha* and eps* dynamically for each target mu in mu_grid
  alpha_grid, eps_grid = jax.vmap(
      compute_alpha_star_from_delta_jax, in_axes=(0, None)
  )(mu_grid, delta_target)

  # 1. Tail Verification via (eps, delta) accounting
  gamma_tail = gamma / 2.0
  passed_tail = verify_tail_eps_delta_batch(
      L_sorted, L_prime_sorted, mu_grid, eps_grid, gamma_tail, delta_target, n
  )

  num_devices = jax.local_device_count()
  chunk_size = 100000
  batch_step = num_devices * chunk_size

  print(f"DEBUG: n={n}")
  print(f"DEBUG: num_devices={num_devices}")
  print(f"DEBUG: chunk_size={chunk_size}")
  print(f"DEBUG: batch_step={batch_step}")

  passed_array = jnp.copy(passed_tail)

  stride = max(1, n // (max_log_points // 2))
  subsampled_alpha = []
  subsampled_beta = []

  @functools.partial(
      jax.pmap, axis_name="devices", in_axes=(0, 0, 0, None, None)
  )
  def process_and_verify_chunk(
      k_beta_chunk, k_alpha_chunk, mask_chunk, mu_g, alpha_g
  ):
    beta_LB = compute_cp_at_k(n, epsilon_cp, k_beta_chunk)
    alpha_LB = compute_cp_at_k(n, epsilon_cp, k_alpha_chunk)

    def check_single_mu(mu, alpha_star):
      G_vals = G_mu(alpha_LB, mu)
      # Restrict check to moderate range:
      # alpha_LB in [alpha_star, 1 - alpha_star]
      in_moderate = (alpha_LB >= alpha_star) & (alpha_LB <= 1.0 - alpha_star)
      failures = (beta_LB < (G_vals - delta_target)) & in_moderate
      valid_failures = jnp.logical_and(failures, mask_chunk)
      return ~jnp.any(valid_failures)

    passed_chunk = jax.vmap(check_single_mu)(mu_g, alpha_g)

    return alpha_LB, beta_LB, passed_chunk

  # Helper to process thresholds
  def process_thresholds(thresholds_source):
    nonlocal passed_array
    for start_idx in range(0, n, batch_step):
      end_idx = min(start_idx + batch_step, n)
      valid_len = end_idx - start_idx

      t_batch_flat = thresholds_source[start_idx:end_idx]

      k_beta_flat = np.searchsorted(L_sorted, t_batch_flat, side="right")
      k_alpha_flat = n - np.searchsorted(
          L_prime_sorted, t_batch_flat, side="right"
      )

      if valid_len < batch_step:
        pad_len = batch_step - valid_len
        k_beta_flat = np.pad(k_beta_flat, (0, pad_len), constant_values=0)
        k_alpha_flat = np.pad(k_alpha_flat, (0, pad_len), constant_values=0)

      k_beta_batch = k_beta_flat.reshape((num_devices, chunk_size))
      k_alpha_batch = k_alpha_flat.reshape((num_devices, chunk_size))

      mask_flat = np.arange(batch_step) < valid_len
      mask_batch = mask_flat.reshape((num_devices, chunk_size))

      alpha_chunk, beta_chunk, passed_chunk_batch = process_and_verify_chunk(
          k_beta_batch, k_alpha_batch, mask_batch, mu_grid, alpha_grid
      )

      passed_batch_reduced = jnp.all(passed_chunk_batch, axis=0)
      passed_array = jnp.logical_and(passed_array, passed_batch_reduced)

      alpha_flat = alpha_chunk.reshape(-1)
      beta_flat = beta_chunk.reshape(-1)

      valid_alpha = alpha_flat[:valid_len]
      valid_beta = beta_flat[:valid_len]

      indices = jnp.arange(0, valid_len, stride)
      subsampled_alpha.append(valid_alpha[indices])
      subsampled_beta.append(valid_beta[indices])

      if not jnp.any(passed_array):
        print("All mu failed, exiting early.")
        return True
    return False

  print("Processing L as thresholds...")
  if not process_thresholds(L_sorted):
    print("Processing L_prime as thresholds...")
    process_thresholds(L_prime_sorted)

  alpha_LB_sub = np.concatenate(subsampled_alpha)
  beta_LB_sub = np.concatenate(subsampled_beta)

  sort_idx = np.argsort(alpha_LB_sub)
  alpha_LB_sub = alpha_LB_sub[sort_idx]
  beta_LB_sub = beta_LB_sub[sort_idx]

  del L_sorted, L_prime_sorted

  return (
      passed_array,
      jnp.array(alpha_LB_sub),
      jnp.array(beta_LB_sub),
      alpha_grid,
      eps_grid,
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  mu_input = jnp.float64(_MU_INPUT.value)
  n = _N.value
  gamma = jnp.float64(_GAMMA.value)
  delta_target = jnp.float64(_DELTA_TARGET.value)
  seed = _SEED.value
  mu_max = jnp.float64(_MU_MAX.value)
  mu_step = jnp.float64(_MU_STEP.value)

  print("JAX running on:", jax.devices()[0].device_kind)

  gamma_roc = gamma / 2.0
  epsilon_cp = gamma_roc / (2.0 * n)

  print(f"epsilon_cp  : {float(epsilon_cp):.6e}")
  print(f"delta_target: {float(delta_target):.2e}")

  mu_grid = jnp.arange(
      mu_input,
      mu_max + mu_step / 2.0,
      mu_step,
      dtype=jnp.float64,
  )

  t2 = time.time()
  passed_grid, alpha_LB, beta_LB, alpha_grid, eps_grid = (
      verify_approx_gdp_algorithm4p9(
          seed, mu_input, n, gamma, delta_target, mu_grid, epsilon_cp=epsilon_cp
      )
  )
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
        "n": int(n),
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
      "delta_target": float(delta_target),
      "epsilon_cp": float(epsilon_cp),
      "time_seconds": float(t3 - t2),
  })

  for step_idx, (mu_target, passed, a_star, e_star) in enumerate(
      zip(mu_grid, passed_grid, alpha_grid, eps_grid)
  ):
    # Should write instead of print in the future.
    print({
        "step": int(step_idx),
        "n": int(n),
        "mu_input": float(mu_input),
        "mu_target": float(mu_target),
        "passed": bool(passed),
        "delta_target": float(delta_target),
        "alpha_star": float(a_star),
        "eps_star": float(e_star),
        "smallest_passed_mu": (
            float(smallest_passed_mu)
            if smallest_passed_mu is not None
            else -1.0
        ),
        "gamma": float(gamma),
    })

  print("=" * 65)
  for mu_target, passed, a_star, e_star in zip(
      mu_grid, passed_grid, alpha_grid, eps_grid
  ):
    status = "PASSED" if passed else "FAILED"
    print(
        f"Target mu = {float(mu_target):.3f} |"
        f" eps* = {float(e_star):6.3f} |"
        f" alpha* = {float(a_star):10.2e} |"
        f" {status}"
    )
  print("-" * 65)
  if smallest_passed_mu is not None:
    print(
        f"Smallest target mu that passed (Alg 4.9): {smallest_passed_mu:.3f}"
    )
  else:
    print("None passed.")
  print("=" * 65)


if __name__ == "__main__":
  app.run(main)
