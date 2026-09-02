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

"""Algorithm 4.9: Hybrid (eps, delta) and Clopper-Pearson ROC Approximate mu-GDP.

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
_DELTA_TARGET = flags.DEFINE_float(
    "delta_target",
    1e-5,
    "Target delta for approximate GDP and (eps, delta) accounting",
)
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")
_CACHE_FILE = flags.DEFINE_string(
    "cache_file",
    "algo4p9_cache.json",
    "Path to JSON file for caching results",
)
_MU_MAX = flags.DEFINE_float("mu_max", 2.0, "Maximum mu to search in grid")
_MU_STEP = flags.DEFINE_float("mu_step", 0.05, "Step size for mu grid")


def G_mu(alpha, mu):
  """GDP trade-off function for Gaussian: G_mu(alpha).

  Args:
    alpha: Significance level alpha.
    mu: Mean separation parameter.

  Returns:
    Trade-off function value G_mu(alpha).
  """
  alpha = np.clip(alpha, np.float64(1e-15), np.float64(1.0 - 1e-15))
  val = -sc_stats.norm.ppf(alpha) - mu
  return sc_stats.norm.cdf(val)


def delta_gdp(mu, eps):
  """Theoretical (eps, delta)-DP guarantee for Gaussian mechanism mu-GDP."""
  t1 = sc_stats.norm.cdf(-eps / mu + mu / 2.0)
  log_t2 = eps + sc_stats.norm.logcdf(-eps / mu - mu / 2.0)
  return t1 - np.exp(log_t2)


def compute_eps_from_delta(mu, delta, max_iter=25, tol=1e-12):
  """Computes eps* such that delta_gdp(mu, eps*) = delta using Newton-Raphson."""
  z = sc_stats.norm.ppf(delta)
  eps = max(0.0, float(mu * (mu / 2.0 - z)))

  for _ in range(max_iter):
    d_val = delta_gdp(mu, eps)
    diff = d_val - delta
    if abs(diff) < tol:
      break
    log_deriv = eps + sc_stats.norm.logcdf(-eps / mu - mu / 2.0)
    deriv = -np.exp(log_deriv)
    eps = max(0.0, eps - diff / deriv)

  return eps


def compute_alpha_star_from_delta(mu, delta):
  """Computes divide point alpha* and tangent eps* corresponding to delta."""
  eps_star = compute_eps_from_delta(mu, delta)
  alpha_star = float(sc_stats.norm.cdf(-eps_star / mu - mu / 2.0))
  return alpha_star, eps_star


def compute_clopper_pearson_table(n, epsilon):
  """Precomputes Clopper-Pearson lower bounds for all k in {0, ..., n}."""
  cp_table = np.zeros(n + 1, dtype=np.float64)
  k = np.arange(1, n + 1, dtype=np.float64)
  a = k
  b = n - k + 1
  cp_table[1:] = sc_special.betaincinv(a, b, epsilon)
  return cp_table


def verify_tail_eps_delta(
    L, L_prime, mu_target, eps_star, gamma_tail, delta_target
):
  """Verifies the tail regions alpha < alpha* and 1 - alpha < alpha*.

  Args:
    L: Samples under P.
    L_prime: Samples under Q.
    mu_target: Target privacy parameter.
    eps_star: Tangent point epsilon.
    gamma_tail: Error budget for tail verification.
    delta_target: Target delta.

  Returns:
    Tuple of (passed, ub_P, ub_Q, d_target).
  """
  n = len(L)
  d_target = delta_gdp(mu_target, eps_star)

  # Positive case (P samples L): E_P[(1 - exp(eps_star - L))_+]
  hockey_loss_P = np.maximum(0.0, 1.0 - np.exp(np.minimum(0.0, eps_star - L)))
  mean_P = np.mean(hockey_loss_P)
  var_P = np.var(hockey_loss_P, ddof=1)

  # Negative case (Q samples L_prime): E_Q[(1 - exp(eps_star + L_prime))_+]
  hockey_loss_Q = np.maximum(
      0.0, 1.0 - np.exp(np.minimum(0.0, eps_star + L_prime))
  )
  mean_Q = np.mean(hockey_loss_Q)
  var_Q = np.var(hockey_loss_Q, ddof=1)

  # Empirical Bernstein upper bound with failure probability gamma_tail / 2 each
  log_term = np.log(4.0 / gamma_tail)
  ub_P = mean_P + np.sqrt(2.0 * var_P * log_term / n) + 3.0 * log_term / n
  ub_Q = mean_Q + np.sqrt(2.0 * var_Q * log_term / n) + 3.0 * log_term / n

  # Tail condition: upper bounds must be <= theoretical delta_target +
  # relaxation
  allowed_delta = delta_target + delta_target
  passed_P = ub_P <= allowed_delta
  passed_Q = ub_Q <= allowed_delta

  return passed_P and passed_Q, ub_P, ub_Q, d_target


def verify_approx_gdp_algorithm4p9(
    seed,
    mu_input,
    n,
    gamma,
    delta_target,
    mu_grid,
    cp_table=None,
):
  """Implements Algorithm 4.9: Reverse Delta-to-Alpha* Hybrid GDP Verification."""
  # 1. Split error budget
  gamma_roc = gamma / np.float64(2.0)
  gamma_tail = gamma / np.float64(2.0)
  epsilon_cp = gamma_roc / (np.float64(2.0) * n)

  # 2. Clopper-Pearson table
  if cp_table is None:
    cp_table = compute_clopper_pearson_table(n, epsilon_cp)

  # 3. Sample continuous privacy losses from P and Q
  rng = np.random.default_rng(seed)
  y = rng.normal(mu_input, 1.0, size=(n,))
  L = y * mu_input - np.float64(0.5) * (mu_input**2)
  del y

  y_prime = rng.normal(0.0, 1.0, size=(n,))
  L_prime = y_prime * mu_input - np.float64(0.5) * (mu_input**2)
  del y_prime

  # Sort privacy losses
  L_sorted = np.sort(L)
  L_prime_sorted = np.sort(L_prime)

  # 4. Thresholds t
  t = np.sort(np.concatenate([L, L_prime]))

  # 5. Lower bounds for beta(t) and alpha(t) at each sample threshold t
  k_beta = np.searchsorted(L_sorted, t, side="right")
  k_alpha = n - np.searchsorted(L_prime_sorted, t, side="right")

  # 6. Clopper-Pearson Lower Bounds
  beta_LB = cp_table[k_beta]
  alpha_LB = cp_table[k_alpha]

  passed = []
  alpha_star_dict = {}
  eps_star_dict = {}

  for mu in mu_grid:
    alpha_star, eps_star = compute_alpha_star_from_delta(mu, delta_target)
    alpha_star_dict[float(mu)] = alpha_star
    eps_star_dict[float(mu)] = eps_star

    # Moderate range check (ROC CP bound):
    # alpha_LB in [alpha_star, 1 - alpha_star]
    moderate_mask = (alpha_LB >= alpha_star) & (alpha_LB <= 1.0 - alpha_star)
    if np.any(moderate_mask):
      G_vals = G_mu(alpha_LB[moderate_mask], mu)
      failures_roc = beta_LB[moderate_mask] < (G_vals - delta_target)
      passed_roc = not np.any(failures_roc)
    else:
      passed_roc = True

    # Tail range check ((eps, delta) MC accounting)
    passed_tail, _, _, _ = verify_tail_eps_delta(
        L, L_prime, mu, eps_star, gamma_tail, delta_target
    )

    passed.append(passed_roc and passed_tail)

  del L, L_prime, L_sorted, L_prime_sorted

  return np.array(passed), alpha_LB, beta_LB, t, alpha_star_dict, eps_star_dict


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  mu_input = np.float64(_MU_INPUT.value)
  n = int(_N.value)
  gamma = np.float64(_GAMMA.value)
  delta_target = np.float64(_DELTA_TARGET.value)
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

  # 1. Failure probability parameter
  gamma_roc = gamma / np.float64(2.0)
  epsilon_cp = gamma_roc / (np.float64(2.0) * n)
  alpha_star_in, eps_star_in = compute_alpha_star_from_delta(
      mu_input, delta_target
  )

  print("=" * 70)
  print("ALGORITHM 4.9 (DELTA -> ALPHA* DIVIDE POINT) CALCULATIONS:")
  print("=" * 70)
  print("Given Inputs:")
  print(f"  • Mechanism parameter mu : {mu_input}")
  print(f"  • Sample size (n)        : {n:,}")
  print(f"  • Failure prob (gamma)   : {gamma}")
  print(f"  • Target delta           : {delta_target:.2e}")
  print("-" * 70)
  print("Calculated Divide Point (at mu_input):")
  print(f"  • Tangent epsilon (eps*) : {eps_star_in:.4f}")
  print(f"  • Divide alpha (alpha*)  : {alpha_star_in:.6e}")
  print(f"  • CP individual epsilon  : {epsilon_cp:.6e}")
  print("=" * 70)

  t0 = time.time()
  cp_table = compute_clopper_pearson_table(n, epsilon_cp)
  t1 = time.time()
  print(f"Computed Clopper-Pearson table (size {n+1:,}) in {t1 - t0:.3f}s")
  print(f"  • CP(1; n, epsilon)   : {float(cp_table[1]):.6e}")
  print(f"  • CP(n; n, epsilon)   : {float(cp_table[-1]):.6e}")
  print("-" * 70)

  # 2. Run Algorithm 4.9 Verification over mu_grid
  mu_grid = np.arange(
      mu_input,
      mu_max + mu_step / np.float64(2.0),
      mu_step,
      dtype=np.float64,
  )

  t2 = time.time()
  (
      passed_grid,
      alpha_LB,
      beta_LB,
      t_eval,
      alpha_star_map,
      eps_star_map,
  ) = verify_approx_gdp_algorithm4p9(
      seed, mu_input, n, gamma, delta_target, mu_grid, cp_table=cp_table
  )
  t3 = time.time()

  print(f"\nAUDIT RESULTS Algorithm 4.9 (Target mu sweep) took {t3 - t2:.2f}s:")
  print("-" * 70)
  for mu_target, passed in zip(mu_grid, passed_grid):
    a_star = alpha_star_map[float(mu_target)]
    e_star = eps_star_map[float(mu_target)]
    status = "PASSED" if passed else "FAILED"
    print(
        f"Target mu = {mu_target:.3f} |"
        f" eps* = {e_star:6.3f} |"
        f" alpha* = {a_star:10.2e} |"
        f" {status}"
    )

  valid_mu = mu_grid[passed_grid]
  print("-" * 70)
  smallest_passed_mu = None
  if len(valid_mu) > 0:
    smallest_passed_mu = float(valid_mu[0])
    print(f"Smallest target mu that passed (Alg 4.9): {smallest_passed_mu:.3f}")
  else:
    print("None of the target mu values passed.")
  print("=" * 70)

  # Save result to cache
  result_key = f"mu{mu_input}_n{n}_g{gamma}_d{delta_target}_s{seed}"
  result_data = {
      "smallest_passed_mu": smallest_passed_mu,
      "epsilon_cp": float(epsilon_cp),
      "num_thresholds": len(t_eval),
      "delta_target": float(delta_target),
  }

  if smallest_passed_mu is not None:
    num_rep = 20
    if len(alpha_LB) > num_rep:
      indices = np.linspace(0, len(alpha_LB) - 1, num=num_rep, dtype=int)
      alpha_rep = alpha_LB[indices].tolist()
      beta_rep = beta_LB[indices].tolist()
    else:
      alpha_rep = alpha_LB.tolist()
      beta_rep = beta_LB.tolist()

    result_data["alpha_LB"] = alpha_rep
    result_data["beta_LB"] = beta_rep
    result_data["alpha_star"] = alpha_star_map[smallest_passed_mu]
    result_data["eps_star"] = eps_star_map[smallest_passed_mu]

  cache["results"][result_key] = result_data
  with open(cache_file, "w") as f:
    json.dump(cache, f, indent=2)
  print(f"Saved result to cache: {cache_file}")


if __name__ == "__main__":
  app.run(main)
