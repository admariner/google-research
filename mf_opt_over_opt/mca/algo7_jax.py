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

"""Algorithm 7: Approximate mu-GDP Verification.

Joint Certification of (alpha, beta) (JAX/GPU version).
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
import scipy.special as sc_special
import scipy.stats as sc_stats

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)


_MU_INPUT = flags.DEFINE_float(
    "mu_input", 1.0, "Actual mechanism privacy parameter"
)
_N = flags.DEFINE_integer("n", 100000, "Sample size for verification")
_GAMMA = flags.DEFINE_float(
    "gamma", 1e-5, "Target failure probability of the statistical bound"
)
_TAU = flags.DEFINE_float("tau", 1e-9, "Tail probability for delta relaxation")
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")
_MU_MAX = flags.DEFINE_float("mu_max", 2.0, "Maximum mu to search in grid")
_MU_STEP = flags.DEFINE_float("mu_step", 0.05, "Step size for mu grid")
_MU_RATIO = flags.DEFINE_float(
    "mu_ratio",
    0.0,
    "Geometric ratio for mu grid (e.g. 1.01). If > 1.0, mu_step is ignored.",
)
_NUM_THRESHOLDS = flags.DEFINE_integer(
    "num_thresholds", 100000, "Number of ordered evaluation thresholds"
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    5000000,
    "Sample batch size per GPU chunk for streaming count",
)


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


def bernoulli_kl(p_hat, p):
  """Calculates KL divergence between two Bernoulli distributions.

  Args:
    p_hat: Empirical probability.
    p: True probability.

  Returns:
    KL divergence kl(p_hat || p).
  """
  eps = 1e-15
  p_hat = jnp.clip(p_hat, eps, 1.0 - eps)
  p = jnp.clip(p, eps, 1.0 - eps)

  t1 = jnp.where(
      p_hat > 0.0,
      p_hat * (jnp.log(p_hat) - jnp.log(p)),
      0.0,
  )
  t2 = jnp.where(
      p_hat < 1.0,
      (1.0 - p_hat) * (jnp.log(1.0 - p_hat) - jnp.log(1.0 - p)),
      0.0,
  )
  return jnp.maximum(0.0, t1 + t2)


def compute_Ic_element(alpha_hat, beta_hat, mu, delta):
  """Computes I_c for a single (alpha_hat, beta_hat) pair via JAX Newton."""
  eps = 1e-15
  alpha_hat_safe = jnp.clip(alpha_hat, eps, 1.0 - eps)
  beta_hat_safe = jnp.clip(beta_hat, eps, 1.0 - eps)

  G_alpha_hat = G_mu(alpha_hat_safe, mu)
  f_hat = jnp.maximum(0.0, G_alpha_hat - delta)

  in_tail_relaxation = jnp.logical_or(
      jnp.logical_or(G_alpha_hat <= delta, alpha_hat_safe <= delta),
      1.0 - beta_hat_safe <= delta,
  )
  is_outside = jnp.logical_or(beta_hat_safe > f_hat, in_tail_relaxation)

  target_G = jnp.clip(beta_hat_safe + delta, eps, 1.0 - eps)
  alpha_0 = G_mu(target_G, mu)
  alpha_0 = jnp.clip(alpha_0, eps, alpha_hat_safe - eps)

  mu_sq_half = 0.5 * (mu**2)
  sqrt_2pi_inv = 1.0 / jnp.sqrt(2.0 * jnp.pi)

  def newton_step(i, state):
    del i
    low, high, cur_a = state
    a = jnp.clip(cur_a, eps, 1.0 - eps)
    z_a = jax_stats.norm.ppf(a)
    f_a = jnp.clip(jax_stats.norm.cdf(-z_a - mu) - delta, eps, 1.0 - eps)
    f_prime_a = -jnp.exp(-mu * z_a - mu_sq_half)
    phi_z = jnp.exp(-0.5 * (z_a**2)) * sqrt_2pi_inv

    # g'(a)
    t1 = (a - alpha_hat_safe) / (a * (1.0 - a))
    u_f = (f_a - beta_hat_safe) / (f_a * (1.0 - f_a))
    t2 = u_f * f_prime_a
    df = t1 + t2

    # Update bracket
    go_low = df < 0.0
    new_low = jnp.where(go_low, a, low)
    new_high = jnp.where(~go_low, a, high)

    # g''(a) > 0
    d2_t1 = (alpha_hat_safe / (a**2)) + (
        (1.0 - alpha_hat_safe) / ((1.0 - a) ** 2)
    )
    u_prime_f = (beta_hat_safe / (f_a**2)) + (
        (1.0 - beta_hat_safe) / ((1.0 - f_a) ** 2)
    )
    f_double_prime_a = -mu * f_prime_a / jnp.maximum(phi_z, eps)
    d2_t2 = (u_prime_f * (f_prime_a**2)) + (u_f * f_double_prime_a)
    d2f = d2_t1 + d2_t2

    # Newton step with bracketing safeguard
    step = df / jnp.maximum(d2f, 1e-12)
    a_cand = a - step
    is_safe = jnp.logical_and(a_cand > new_low + eps, a_cand < new_high - eps)
    next_a = jnp.where(is_safe, a_cand, 0.5 * (new_low + new_high))

    return new_low, new_high, next_a

  init_a = 0.5 * (alpha_0 + alpha_hat_safe)
  _, _, alpha_final = jax.lax.fori_loop(
      0, 4, newton_step, (alpha_0, alpha_hat_safe, init_a)
  )

  alpha_opt = jnp.clip(alpha_final, eps, 1.0 - eps)
  z_opt = jax_stats.norm.ppf(alpha_opt)
  f_opt = jnp.clip(jax_stats.norm.cdf(-z_opt - mu) - delta, eps, 1.0 - eps)

  kl_alpha = bernoulli_kl(alpha_hat_safe, alpha_opt)
  kl_beta = bernoulli_kl(beta_hat_safe, f_opt)
  Ic_interior = kl_alpha + kl_beta

  Ic_val = jnp.where(
      in_tail_relaxation,
      1e10,
      jnp.where(is_outside, Ic_interior, 0.0),
  )
  return Ic_val, alpha_opt


def compute_exact_pc_single(n, mu, delta, Ic_obs, alpha_opt):
  """Computes Equation (7): exact discrete constrained-binomial p-value."""
  if np.isinf(Ic_obs) or Ic_obs >= 1e9:
    return float(0.0)
  if Ic_obs <= 0.0:
    return float(1.0)

  eps = np.float64(1e-15)
  alpha_star = np.clip(alpha_opt, eps, np.float64(1.0) - eps)
  beta_star = np.clip(
      sc_special.ndtr(-sc_special.ndtri(alpha_star) - mu) - delta,
      eps,
      np.float64(1.0) - eps,
  )

  k_alpha_arr = np.arange(0, n + 1, dtype=np.float64)
  alpha_hat_arr = k_alpha_arr / np.float64(n)

  # Fully vectorized binary search for critical k_beta*(k_alpha)
  low_kb = np.zeros(len(k_alpha_arr), dtype=np.float64)
  high_kb = np.full(len(k_alpha_arr), float(n), dtype=np.float64)
  ans_kb = np.full(len(k_alpha_arr), float(n + 1), dtype=np.float64)

  # Vectorized Newton evaluation of Ic in NumPy
  def compute_Ic_np(a_hat, b_hat, mu_val, delta_val, num_steps=4):
    clipped_a = np.clip(a_hat, eps, 1.0 - eps)
    G_a = np.clip(
        sc_special.ndtr(-sc_special.ndtri(clipped_a) - mu_val),
        eps,
        1.0 - eps,
    )
    f_h = np.maximum(np.float64(0.0), G_a - delta_val)
    in_tail = (
        (G_a <= delta_val)
        | (a_hat <= delta_val)
        | (np.float64(1.0) - b_hat <= delta_val)
    )
    act_mask = (b_hat > f_h) & (~in_tail)

    Ic_out = np.zeros_like(a_hat)
    Ic_out[in_tail] = 1e10
    if not np.any(act_mask):
      return Ic_out

    a_act = a_hat[act_mask]
    b_act = b_hat[act_mask]

    tgt_G = np.clip(b_act + delta_val, eps, np.float64(1.0) - eps)
    a_0 = np.clip(
        sc_special.ndtr(-sc_special.ndtri(tgt_G) - mu_val),
        eps,
        a_act - eps,
    )
    lo = np.copy(a_0)
    hi = np.clip(a_act, eps, np.float64(1.0) - eps)
    cur_a = (lo + hi) * np.float64(0.5)

    mu_sq_half = np.float64(0.5) * (mu_val**2)
    sqrt_2pi_inv = np.float64(1.0 / np.sqrt(2.0 * np.pi))

    for _ in range(num_steps):
      a = np.clip(cur_a, eps, np.float64(1.0) - eps)
      z_a = sc_special.ndtri(a)
      f_a = np.clip(
          sc_special.ndtr(-z_a - mu_val) - delta_val,
          eps,
          np.float64(1.0) - eps,
      )
      f_prime_a = -np.exp(-mu_val * z_a - mu_sq_half)
      phi_z = np.exp(np.float64(-0.5) * (z_a**2)) * sqrt_2pi_inv

      t1 = (a - a_act) / (a * (np.float64(1.0) - a))
      u_f = (f_a - b_act) / (f_a * (np.float64(1.0) - f_a))
      t2 = u_f * f_prime_a
      df = t1 + t2

      go_low = df < 0.0
      new_lo = np.where(go_low, a, lo)
      new_hi = np.where(~go_low, a, hi)

      d2_t1 = (a_act / (a**2)) + (
          (np.float64(1.0) - a_act) / ((np.float64(1.0) - a) ** 2)
      )
      u_prime_f = (b_act / (f_a**2)) + (
          (np.float64(1.0) - b_act) / ((np.float64(1.0) - f_a) ** 2)
      )
      f_double_prime_a = -mu_val * f_prime_a / np.maximum(phi_z, eps)
      d2_t2 = (u_prime_f * (f_prime_a**2)) + (u_f * f_double_prime_a)
      d2f = d2_t1 + d2_t2

      step = df / np.maximum(d2f, np.float64(1e-12))
      a_cand = a - step
      is_safe = (a_cand > new_lo + eps) & (a_cand < new_hi - eps)
      cur_a = np.where(is_safe, a_cand, (new_lo + new_hi) * np.float64(0.5))
      lo, hi = new_lo, new_hi

    a_opt_act = np.clip(cur_a, eps, np.float64(1.0) - eps)
    f_opt_act = np.clip(
        sc_special.ndtr(-sc_special.ndtri(a_opt_act) - mu_val) - delta_val,
        eps,
        np.float64(1.0) - eps,
    )
    kl_a = a_act * np.log(np.clip(a_act / a_opt_act, eps, 1e10)) + (
        np.float64(1.0) - a_act
    ) * np.log(
        np.clip(
            (np.float64(1.0) - a_act) / (np.float64(1.0) - a_opt_act),
            eps,
            1e10,
        )
    )
    kl_b = b_act * np.log(np.clip(b_act / f_opt_act, eps, 1e10)) + (
        np.float64(1.0) - b_act
    ) * np.log(
        np.clip(
            (np.float64(1.0) - b_act) / (np.float64(1.0) - f_opt_act),
            eps,
            1e10,
        )
    )
    Ic_out[act_mask] = kl_a + kl_b
    return Ic_out

  num_bs_steps = int(np.ceil(np.log2(n + 2))) + 1
  for _ in range(num_bs_steps):
    mid_kb = np.floor(0.5 * (low_kb + high_kb))
    beta_hat_arr = mid_kb / np.float64(n)
    Ic_arr = compute_Ic_np(alpha_hat_arr, beta_hat_arr, mu, delta)
    is_ge = Ic_arr >= Ic_obs
    ans_kb = np.where(is_ge, mid_kb, ans_kb)
    high_kb = np.where(is_ge, mid_kb - 1.0, high_kb)
    low_kb = np.where(is_ge, low_kb, mid_kb + 1.0)

  prob_k_alpha = sc_stats.binom.pmf(k_alpha_arr, n, alpha_star)
  prob_k_beta_ge = sc_special.bdtrc(ans_kb - 1.0, n, beta_star)

  p_c = np.sum(prob_k_alpha * prob_k_beta_ge)
  return float(p_c)


def generate_threshold_grid(mu_input, num_thresholds=100000, sigma_span=7.5):
  """Generates ordered 1D grid of thresholds covering non-relaxed spectrum."""
  mu_f = float(mu_input)
  t_min = -0.5 * (mu_f**2) - sigma_span * mu_f
  t_max = 0.5 * (mu_f**2) + sigma_span * mu_f
  return jnp.linspace(t_min, t_max, num_thresholds, dtype=jnp.float64)


def _stream_count_device(
    key_p, key_q, mu_input, n_device, T, batch_size, num_batches
):
  """Counts empirical CDFs for P and Q via GPU streaming searchsorted."""
  M = T.shape[0]
  mu_f = jnp.float64(mu_input)

  # P stream (counts L <= t)
  def body_p(i, state):
    k, count_acc = state
    k, subkey = jax.random.split(k)
    cur_start = i * batch_size
    cur_n = jnp.clip(n_device - cur_start, 0, batch_size)

    y = jax.random.normal(subkey, shape=(batch_size,), dtype=jnp.float64) + mu_f
    L = y * mu_f - 0.5 * (mu_f**2)

    bins = jnp.searchsorted(T, L, side="right")
    valid_mask = jnp.arange(batch_size) < cur_n
    counts = jnp.bincount(
        bins, weights=valid_mask.astype(jnp.int64), length=M + 1
    )
    return (k, count_acc + counts)

  init_p = (key_p, jnp.zeros(M + 1, dtype=jnp.int64))
  _, counts_P = jax.lax.fori_loop(0, num_batches, body_p, init_p)

  # Q stream (counts L' <= t)
  def body_q(i, state):
    k, count_acc = state
    k, subkey = jax.random.split(k)
    cur_start = i * batch_size
    cur_n = jnp.clip(n_device - cur_start, 0, batch_size)

    y_prime = jax.random.normal(subkey, shape=(batch_size,), dtype=jnp.float64)
    L_prime = y_prime * mu_f - 0.5 * (mu_f**2)

    bins_q = jnp.searchsorted(T, L_prime, side="right")
    valid_mask = jnp.arange(batch_size) < cur_n
    counts_q = jnp.bincount(
        bins_q, weights=valid_mask.astype(jnp.int64), length=M + 1
    )
    return (k, count_acc + counts_q)

  init_q = (key_q, jnp.zeros(M + 1, dtype=jnp.int64))
  _, counts_Q = jax.lax.fori_loop(0, num_batches, body_q, init_q)

  return counts_P, counts_Q


def verify_approx_gdp_algorithm7(
    seed,
    mu_input,
    n,
    gamma,
    tau,
    mu_grid,
    epsilon,
    thresholds=None,
    num_thresholds=100000,
    batch_size=5000000,
):
  """Implements Algorithm 7 with Streaming GPU Bin-Counting."""
  del gamma
  if thresholds is None:
    T = generate_threshold_grid(mu_input, num_thresholds=num_thresholds)
  else:
    T = jnp.sort(jnp.asarray(thresholds, dtype=jnp.float64))

  M = T.shape[0]
  num_devices = jax.local_device_count()

  print(f"DEBUG: n={n:,}")
  print(f"DEBUG: num_devices={num_devices}")
  print(f"DEBUG: num_thresholds (M)={M:,}")
  print(f"DEBUG: batch_size={batch_size:,}")

  # Distribute sample count across devices
  base_n = n // num_devices
  rem = n % num_devices
  n_per_device = np.array(
      [base_n + (1 if i < rem else 0) for i in range(num_devices)],
      dtype=np.int64,
  )
  max_n_device = int(np.max(n_per_device))
  eff_batch_size = min(batch_size, max_n_device)
  num_batches = (max_n_device + eff_batch_size - 1) // eff_batch_size

  # Generate PRNG keys for all devices
  root_key = jax.random.PRNGKey(seed)
  root_key_p, root_key_q = jax.random.split(root_key)
  keys_p = jax.random.split(root_key_p, num_devices)
  keys_q = jax.random.split(root_key_q, num_devices)

  @functools.partial(
      jax.pmap,
      axis_name="devices",
      in_axes=(0, 0, None, 0, None),
  )
  def _count_pmapped(k_p, k_q, mu_in, n_dev_arr, T_arr):
    cP, cQ = _stream_count_device(
        k_p,
        k_q,
        mu_in,
        n_dev_arr,
        T_arr,
        batch_size=eff_batch_size,
        num_batches=num_batches,
    )
    tot_P = jax.lax.psum(cP, axis_name="devices")
    tot_Q = jax.lax.psum(cQ, axis_name="devices")
    return tot_P, tot_Q

  print("Streaming sample generation and counting k_beta / k_alpha on GPU...")
  t_count_start = time.time()
  total_P_sharded, total_Q_sharded = _count_pmapped(
      keys_p, keys_q, float(mu_input), jnp.array(n_per_device), T
  )
  total_P = total_P_sharded[0].block_until_ready()
  total_Q = total_Q_sharded[0].block_until_ready()
  t_count_end = time.time()
  print(
      f"GPU streaming counting finished in {t_count_end - t_count_start:.2f}s"
  )

  # Compute empirical k_beta and k_alpha
  k_beta = jnp.cumsum(total_P[:M])
  k_alpha = n - jnp.cumsum(total_Q[:M])

  beta_hat = k_beta.astype(jnp.float64) / float(n)
  alpha_hat = k_alpha.astype(jnp.float64) / float(n)

  # Pre-allocate JAX array for passed status
  passed_array = jnp.ones(len(mu_grid), dtype=jnp.bool_)
  z_crit = float(sc_special.ndtri(1.0 - epsilon))
  I_asymp_crit = (z_crit**2) / (2.0 * float(n))

  @jax.jit
  def evaluate_all_Ic(a_hat, b_hat, mu_g):
    def compute_for_single_mu(mu):
      delta = compute_delta(tau, mu)
      Ic_vals, alpha_opts = jax.vmap(
          compute_Ic_element, in_axes=(0, 0, None, None)
      )(a_hat, b_hat, mu, delta)
      return Ic_vals, alpha_opts

    return jax.vmap(compute_for_single_mu)(mu_g)

  print("Evaluating Ic across all thresholds and target mu grid...")
  Ic_grid, alpha_opt_grid = evaluate_all_Ic(alpha_hat, beta_hat, mu_grid)
  Ic_grid_np = np.array(Ic_grid.block_until_ready())
  alpha_opt_grid_np = np.array(alpha_opt_grid.block_until_ready())

  for mu_idx, mu_val in enumerate(mu_grid):
    mu_float = float(mu_val)
    delta_float = float(compute_delta(tau, mu_val))
    Ic_for_mu = Ic_grid_np[mu_idx]
    alpha_opt_for_mu = alpha_opt_grid_np[mu_idx]

    # Immediate rejection if any threshold falls on or below privacy curve
    if np.any(Ic_for_mu <= 0.0):
      passed_array = passed_array.at[mu_idx].set(False)
      continue

    candidate_mask = (Ic_for_mu < 3.0 * I_asymp_crit) & (Ic_for_mu > 0.0)
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) > 0:
      worst_indices = candidate_indices[
          np.argsort(Ic_for_mu[candidate_indices])
      ]
      for w_idx in worst_indices:
        pc_val = compute_exact_pc_single(
            n,
            mu_float,
            delta_float,
            float(Ic_for_mu[w_idx]),
            float(alpha_opt_for_mu[w_idx]),
        )
        if pc_val > epsilon:
          passed_array = passed_array.at[mu_idx].set(False)
          break

  # Sort subsampled points by alpha for smooth trade-off curve
  alpha_hat_np = np.array(alpha_hat)
  beta_hat_np = np.array(beta_hat)
  sort_idx = np.argsort(alpha_hat_np)
  alpha_hat_sorted = alpha_hat_np[sort_idx]
  beta_hat_sorted = beta_hat_np[sort_idx]

  return passed_array, jnp.array(alpha_hat_sorted), jnp.array(beta_hat_sorted)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  mu_input = jnp.float64(_MU_INPUT.value)
  n = _N.value
  gamma = jnp.float64(_GAMMA.value)
  tau = jnp.float64(_TAU.value)
  seed = _SEED.value
  mu_max = jnp.float64(_MU_MAX.value)
  mu_step = jnp.float64(_MU_STEP.value)
  num_thresholds = _NUM_THRESHOLDS.value
  batch_size = _BATCH_SIZE.value

  print("JAX running on:", jax.devices()[0].device_kind)

  # epsilon = float(gamma) / float(num_thresholds)
  epsilon = float(gamma)
  delta_input = compute_delta(tau, mu_input)

  print("=" * 65)
  print("ALGORITHM 7 JAX (EXACT P-VALUE) PARAMETER CALCULATIONS:")
  print("=" * 65)
  print("Given Inputs:")
  print(f"  • Mechanism parameter mu : {mu_input}")
  print(f"  • Sample size (n)        : {n:,}")
  print(f"  • Failure prob (gamma)   : {gamma}")
  print(f"  • Tail prob cutoff (tau) : {tau}")
  print(f"  • Num thresholds (M)     : {num_thresholds:,}")
  print(f"  • Batch size             : {batch_size:,}")
  print("-" * 65)
  print("Calculated Quantities:")
  print(f"  • Individual epsilon     : {epsilon:.6e}")
  print(f"  • Relaxation (delta)     : {float(delta_input):.2e}")
  print("=" * 65)

  if _MU_RATIO.value > 1.0:
    mu_list = []
    curr = float(mu_input)
    while curr <= float(mu_max) * 1.0001:
      mu_list.append(curr)
      curr *= _MU_RATIO.value
    mu_grid = jnp.array(mu_list, dtype=jnp.float64)
  else:
    mu_grid = jnp.arange(
        mu_input,
        mu_max + mu_step / 2.0,
        mu_step,
        dtype=jnp.float64,
    )

  t2 = time.time()
  passed_grid, alpha_hat, beta_hat = verify_approx_gdp_algorithm7(
      seed,
      mu_input,
      n,
      gamma,
      tau,
      mu_grid,
      epsilon=epsilon,
      num_thresholds=num_thresholds,
      batch_size=batch_size,
  )
  passed_grid.block_until_ready()
  t3 = time.time()

  print(f"Verification took {t3 - t2:.2f}s")

  num_points = len(alpha_hat)
  max_log_points = 10000
  if num_points > max_log_points:
    indices = np.linspace(0, num_points - 1, max_log_points, dtype=int)
    alpha_sub = alpha_hat[indices]
    beta_sub = beta_hat[indices]
    step_indices = indices
  else:
    alpha_sub = alpha_hat
    beta_sub = beta_hat
    step_indices = range(num_points)

  for i, (a_val, b_val) in zip(step_indices, zip(alpha_sub, beta_sub)):
    # Should write instead of print in the future.
    print({
        "step": int(i),
        "n": int(n),
        "alpha_hat": float(a_val),
        "beta_hat": float(b_val),
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
      "epsilon": float(epsilon),
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
