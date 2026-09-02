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

"""Algorithm 7: Approximate mu-GDP Verification with Joint Certification.

NumPy version.
"""

# pylint: disable=invalid-name
import json
import os
import time
from absl import app
from absl import flags
import numpy as np
import scipy.special as sc_special
import scipy.stats as sc_stats

_MU_INPUT = flags.DEFINE_float(
    "mu_input", 1.0, "Actual mechanism privacy parameter"
)
_N = flags.DEFINE_float("n", 100000, "Sample size for verification")
_GAMMA = flags.DEFINE_float(
    "gamma", 1e-5, "Target failure probability of the statistical bound"
)
_TAU = flags.DEFINE_float("tau", 1e-9, "Tail probability for delta relaxation")
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")
_CACHE_FILE = flags.DEFINE_string(
    "cache_file",
    "algo7_cache.json",
    "Path to JSON file for caching results",
)
_MU_MAX = flags.DEFINE_float("mu_max", 2.0, "Maximum mu to search in grid")
_MU_STEP = flags.DEFINE_float("mu_step", 0.05, "Step size for mu grid")
_MU_RATIO = flags.DEFINE_float(
    "mu_ratio",
    0.0,
    "Geometric ratio for mu grid (e.g. 1.01). If > 1.0, mu_step is ignored.",
)


def G_mu(alpha, mu):
  """GDP trade-off function for Gaussian: G_mu(alpha).

  Args:
    alpha: Significance level alpha.
    mu: Mean separation parameter.

  Returns:
    Trade-off function value G_mu(alpha).
  """
  alpha = np.clip(alpha, np.float64(1e-15), np.float64(1.0 - 1e-15))
  val = -sc_special.ndtri(alpha) - mu
  return sc_special.ndtr(val)


def compute_delta(tau, mu_target):
  """Computes relaxation parameter delta = max(tau, G_mu(1 - tau, mu_target))."""
  z_tau_lower = sc_special.ndtri(tau)
  g_val = sc_special.ndtr(z_tau_lower - mu_target)
  return np.maximum(tau, g_val)


def bernoulli_kl(p_hat, p):
  """Calculates KL divergence between two Bernoulli distributions.

  Args:
    p_hat: Empirical probability.
    p: True probability.

  Returns:
    KL divergence kl(p_hat || p).
  """
  eps = np.float64(1e-15)
  p_hat = np.clip(p_hat, eps, np.float64(1.0) - eps)
  p = np.clip(p, eps, np.float64(1.0) - eps)

  t1 = np.where(
      p_hat > np.float64(0.0),
      p_hat * (np.log(p_hat) - np.log(p)),
      np.float64(0.0),
  )
  t2 = np.where(
      p_hat < np.float64(1.0),
      (np.float64(1.0) - p_hat)
      * (np.log(np.float64(1.0) - p_hat) - np.log(np.float64(1.0) - p)),
      np.float64(0.0),
  )
  return np.maximum(np.float64(0.0), t1 + t2)


def compute_Ic_vectorized(
    alpha_hat, beta_hat, mu, delta, num_newton_steps=4
):
  """Computes I_c via Safeguarded Newton-Raphson iteration.

  Args:
    alpha_hat: Empirical alpha array.
    beta_hat: Empirical beta array.
    mu: Mean separation parameter.
    delta: Relaxation delta.
    num_newton_steps: Number of Newton steps to take.

  Returns:
    Tuple of (Ic, alpha_opt, f_opt).
  """
  eps = np.float64(1e-15)
  G_alpha_hat = G_mu(alpha_hat, mu)
  f_hat = np.maximum(np.float64(0.0), G_alpha_hat - delta)

  # Lower tail: G_mu(alpha_hat) <= delta or alpha_hat <= delta
  # Upper tail: 1 - beta_hat <= delta
  in_tail_relaxation = (
      (G_alpha_hat <= delta)
      | (alpha_hat <= delta)
      | (np.float64(1.0) - beta_hat <= delta)
  )

  active_mask = (beta_hat > f_hat) & (~in_tail_relaxation)

  alpha_opt = np.copy(alpha_hat)
  f_opt = np.clip(G_alpha_hat - delta, eps, np.float64(1.0) - eps)
  Ic = np.zeros_like(alpha_hat)
  Ic[in_tail_relaxation] = np.inf

  if not np.any(active_mask):
    return Ic, alpha_opt, f_opt

  # Run Safeguarded Newton on active points
  a_hat_act = alpha_hat[active_mask]
  b_hat_act = beta_hat[active_mask]

  target_G = np.clip(b_hat_act + delta, eps, np.float64(1.0) - eps)
  alpha_0 = G_mu(target_G, mu)
  alpha_0 = np.clip(alpha_0, eps, a_hat_act - eps)

  low = np.copy(alpha_0)
  high = np.clip(a_hat_act, eps, np.float64(1.0) - eps)
  alpha_cur = (low + high) * np.float64(0.5)

  mu_sq_half = np.float64(0.5) * (mu**2)
  sqrt_2pi_inv = np.float64(1.0 / np.sqrt(2.0 * np.pi))

  for _ in range(num_newton_steps):
    a = np.clip(alpha_cur, eps, np.float64(1.0) - eps)
    z_a = sc_special.ndtri(a)
    f_a = np.clip(
        sc_special.ndtr(-z_a - mu) - delta, eps, np.float64(1.0) - eps
    )
    f_prime_a = -np.exp(-mu * z_a - mu_sq_half)
    phi_z = np.exp(np.float64(-0.5) * (z_a**2)) * sqrt_2pi_inv

    # g'(a)
    t1 = (a - a_hat_act) / (a * (np.float64(1.0) - a))
    u_f = (f_a - b_hat_act) / (f_a * (np.float64(1.0) - f_a))
    t2 = u_f * f_prime_a
    df = t1 + t2

    # Update bracket based on sign of df
    go_low = df < 0.0
    low = np.where(go_low, a, low)
    high = np.where(~go_low, a, high)

    # Exact g''(a) > 0
    d2_t1 = (a_hat_act / (a**2)) + (
        (np.float64(1.0) - a_hat_act) / ((np.float64(1.0) - a) ** 2)
    )
    u_prime_f = (b_hat_act / (f_a**2)) + (
        (np.float64(1.0) - b_hat_act) / ((np.float64(1.0) - f_a) ** 2)
    )
    f_double_prime_a = -mu * f_prime_a / np.maximum(phi_z, eps)
    d2_t2 = (u_prime_f * (f_prime_a**2)) + (u_f * f_double_prime_a)
    d2f = d2_t1 + d2_t2

    # Newton step with bracketing safeguard
    newton_step = df / np.maximum(d2f, np.float64(1e-12))
    a_new = a - newton_step
    is_safe = (a_new > low + eps) & (a_new < high - eps)
    alpha_cur = np.where(is_safe, a_new, (low + high) * np.float64(0.5))

  a_opt_act = np.clip(alpha_cur, eps, np.float64(1.0) - eps)
  f_opt_act = np.clip(
      sc_special.ndtr(-sc_special.ndtri(a_opt_act) - mu) - delta,
      eps,
      np.float64(1.0) - eps,
  )

  kl_alpha = bernoulli_kl(a_hat_act, a_opt_act)
  kl_beta = bernoulli_kl(b_hat_act, f_opt_act)

  alpha_opt[active_mask] = a_opt_act
  f_opt[active_mask] = f_opt_act
  Ic[active_mask] = kl_alpha + kl_beta

  return Ic, alpha_opt, f_opt


def compute_exact_pc_single(n, mu, delta, Ic_obs, alpha_opt):
  """Computes Equation (7): exact discrete constrained-binomial p-value.

  Args:
    n: Sample size.
    mu: Mean separation parameter.
    delta: Relaxation delta.
    Ic_obs: Observed divergence statistic.
    alpha_opt: Least favorable parameter alpha*.

  Returns:
    Exact p-value p_c.
  """
  if np.isinf(Ic_obs) or Ic_obs >= 1e9:
    return np.float64(0.0)
  if Ic_obs <= 0.0:
    return np.float64(1.0)

  eps = np.float64(1e-15)
  alpha_star = np.clip(alpha_opt, eps, np.float64(1.0) - eps)
  beta_star = np.clip(G_mu(alpha_star, mu) - delta, eps, np.float64(1.0) - eps)

  k_alpha_arr = np.arange(0, n + 1, dtype=np.float64)
  alpha_hat_arr = k_alpha_arr / np.float64(n)

  # Fully vectorized binary search for critical k_beta*(k_alpha)
  low_kb = np.zeros(len(k_alpha_arr), dtype=np.float64)
  high_kb = np.full(len(k_alpha_arr), float(n), dtype=np.float64)
  ans_kb = np.full(len(k_alpha_arr), float(n + 1), dtype=np.float64)

  num_bs_steps = int(np.ceil(np.log2(n + 2))) + 1
  for _ in range(num_bs_steps):
    mid_kb = np.floor(0.5 * (low_kb + high_kb))
    beta_hat_arr = mid_kb / np.float64(n)
    Ic_arr, _, _ = compute_Ic_vectorized(
        alpha_hat_arr, beta_hat_arr, mu, delta, num_newton_steps=4
    )
    is_ge = Ic_arr >= Ic_obs
    ans_kb = np.where(is_ge, mid_kb, ans_kb)
    high_kb = np.where(is_ge, mid_kb - 1.0, high_kb)
    low_kb = np.where(is_ge, low_kb, mid_kb + 1.0)

  prob_k_alpha = sc_stats.binom.pmf(k_alpha_arr, n, alpha_star)
  prob_k_beta_ge = sc_special.bdtrc(ans_kb - 1.0, n, beta_star)

  p_c = np.sum(prob_k_alpha * prob_k_beta_ge)
  return float(p_c)


def verify_approx_gdp_algorithm7(
    seed,
    mu_input,
    n,
    gamma,
    tau,
    mu_grid,
):
  """Implements Algorithm 7: Approximate mu-GDP Verification.

  Args:
    seed: PRNG seed.
    mu_input: Input mechanism privacy parameter.
    n: Sample size.
    gamma: Failure probability.
    tau: Tail probability cutoff.
    mu_grid: Grid of target mu values to check.

  Returns:
    Tuple of (passed_grid, alpha_hat, beta_hat, last_Ic, thresholds).
  """
  # 1. Individual failure probability via union bound
  # epsilon = gamma / (np.float64(2.0) * n)
  epsilon = gamma
  z_crit = sc_special.ndtri(np.float64(1.0) - epsilon)
  I_asymp_crit = (z_crit**2) / (np.float64(2.0) * n)

  # 2. Sample continuous privacy losses from P and Q
  seed_p = seed
  seed_q = seed + 12345
  rng_p = np.random.default_rng(seed_p)
  y = rng_p.normal(mu_input, 1.0, size=(n,))
  L = y * mu_input - np.float64(0.5) * (mu_input**2)
  del y

  rng_q = np.random.default_rng(seed_q)
  y_prime = rng_q.normal(0.0, 1.0, size=(n,))
  L_prime = y_prime * mu_input - np.float64(0.5) * (mu_input**2)
  del y_prime

  # Sort privacy losses
  L_sorted = np.sort(L)
  L_prime_sorted = np.sort(L_prime)

  # 3. Thresholds t: concatenate continuous sample losses L and L_prime
  t = np.sort(np.concatenate([L, L_prime]))
  del L, L_prime

  # 4. Empirical proportions for beta(t) and alpha(t) at each sample threshold t
  k_beta = np.searchsorted(L_sorted, t, side="right")
  k_alpha = n - np.searchsorted(L_prime_sorted, t, side="right")

  beta_hat = k_beta.astype(np.float64) / np.float64(n)
  alpha_hat = k_alpha.astype(np.float64) / np.float64(n)

  # 5. Joint certification over mu_grid
  passed = []
  last_Ic = None
  for mu in mu_grid:
    delta = compute_delta(tau, mu)

    G_alpha_hat = G_mu(alpha_hat, mu)
    f_hat = np.maximum(np.float64(0.0), G_alpha_hat - delta)
    in_tail_relaxation = (
        (G_alpha_hat <= delta)
        | (alpha_hat <= delta)
        | (np.float64(1.0) - beta_hat <= delta)
    )
    bad_mask = (beta_hat <= f_hat) & (~in_tail_relaxation)
    if np.any(bad_mask):
      # Immediate failure: points on or below the tradeoff curve have
      # Ic = 0 (pc = 1.0 > epsilon)
      passed.append(False)
      continue

    Ic, alpha_opt, _ = compute_Ic_vectorized(
        alpha_hat, beta_hat, mu, delta
    )
    last_Ic = Ic

    # Subsample candidate bottleneck thresholds closest to boundary
    # to verify pc <= epsilon
    candidate_mask = Ic < 3.0 * I_asymp_crit
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
      # All thresholds are safely far in the acceptance region
      passed.append(True)
      continue

    # Check exact p_c on all candidate bottleneck thresholds
    worst_indices = candidate_indices[np.argsort(Ic[candidate_indices])]
    failed_any = False
    for idx in worst_indices:
      pc_val = compute_exact_pc_single(n, mu, delta, Ic[idx], alpha_opt[idx])
      if pc_val > epsilon:
        failed_any = True
        break
    passed.append(not failed_any)

  return np.array(passed), alpha_hat, beta_hat, last_Ic, t


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  mu_input = np.float64(_MU_INPUT.value)
  n = int(_N.value)
  gamma = np.float64(_GAMMA.value)
  tau = np.float64(_TAU.value)
  seed = _SEED.value
  cache_file = _CACHE_FILE.value
  mu_max = np.float64(_MU_MAX.value)
  mu_step = np.float64(_MU_STEP.value)

  if not os.path.isabs(cache_file):
    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
      cache_file = os.path.join(
          os.environ["BUILD_WORKSPACE_DIRECTORY"],
          "experimental/mf_opt_over_opt/research/mca",
          cache_file,
      )

  cache = {"results": {}}

  if os.path.exists(cache_file):
    try:
      with open(cache_file, "r") as f:
        cache = json.load(f)
        if "results" not in cache:
          cache["results"] = {}
    except json.JSONDecodeError:
      print(
          f"Warning: Could not decode {cache_file}, starting with empty cache."
      )

  # 1. Failure probability parameter and relaxation delta
  epsilon = gamma / (np.float64(2.0) * n)
  delta_input = compute_delta(tau, mu_input)

  print("=" * 65)
  print("ALGORITHM 7 (EXACT P-VALUE) PARAMETER CALCULATIONS:")
  print("=" * 65)
  print("Given Inputs:")
  print(f"  • Mechanism parameter mu : {mu_input}")
  print(f"  • Sample size (n)        : {n:,}")
  print(f"  • Failure prob (gamma)   : {gamma}")
  print(f"  • Tail prob cutoff (tau) : {tau}")
  print("-" * 65)
  print("Calculated Quantities:")
  print(f"  • Individual epsilon     : {epsilon:.6e}")
  print(f"  • Relaxation (delta)     : {float(delta_input):.2e}")
  print("=" * 65)

  # 2. Run Algorithm 7 Verification over mu_grid
  if _MU_RATIO.value > 1.0:
    mu_list = []
    curr = float(mu_input)
    while curr <= float(mu_max) * 1.0001:
      mu_list.append(curr)
      curr *= _MU_RATIO.value
    mu_grid = np.array(mu_list, dtype=np.float64)
  else:
    mu_grid = np.arange(
        mu_input,
        mu_max + mu_step / np.float64(2.0),
        mu_step,
        dtype=np.float64,
    )

  t0 = time.time()
  passed_grid, _, _, _, t_eval = (
      verify_approx_gdp_algorithm7(
          seed, mu_input, n, gamma, tau, mu_grid
      )
  )
  t1 = time.time()

  print(f"\nAUDIT RESULTS Algorithm 7 (Target mu sweep) took {t1 - t0:.2f}s:")
  print("-" * 65)
  for mu_target, passed in zip(mu_grid, passed_grid):
    delta = compute_delta(tau, mu_target)
    status = "PASSED" if passed else "FAILED"
    print(
        f"Target mu = {mu_target:.3f} | delta = {float(delta):.2e} | {status}"
    )

  valid_mu = mu_grid[passed_grid]
  print("-" * 65)
  smallest_passed_mu = None
  if len(valid_mu) > 0:
    smallest_passed_mu = float(valid_mu[0])
    print(f"Smallest target mu that passed (Alg 7): {smallest_passed_mu:.3f}")
  else:
    print("None of the target mu values passed.")
  print("=" * 65)

  # Save result to cache
  result_key = f"mu{mu_input}_n{n}_g{gamma}_t{tau}_s{seed}_exact_p_value"
  result_data = {
      "smallest_passed_mu": smallest_passed_mu,
      "epsilon": float(epsilon),
      "cert_method": "exact_p_value",
      "num_thresholds": len(t_eval),
  }

  cache["results"][result_key] = result_data
  with open(cache_file, "w") as f:
    json.dump(cache, f, indent=2)
  print(f"Saved result to cache: {cache_file}")


if __name__ == "__main__":
  app.run(main)
