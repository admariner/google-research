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

"""Algorithm 4.5: Approximate mu-GDP Verification with Clopper-Pearson.

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
    "algo4p5_cache.json",
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


def compute_delta(tau, mu_target):
  """Computes relaxation parameter delta = max(tau, G_mu(1 - tau, mu_target)).

  Args:
    tau: Tail probability cutoff.
    mu_target: Target privacy parameter.

  Returns:
    Relaxation delta.
  """
  z_tau_lower = sc_stats.norm.ppf(tau)
  g_val = sc_stats.norm.cdf(z_tau_lower - mu_target)
  return np.maximum(tau, g_val)


def compute_clopper_pearson_table(n, epsilon):
  """Precomputes Clopper-Pearson lower bounds for all k in {0, ..., n}.

  Args:
    n: Sample size.
    epsilon: Significance parameter.

  Returns:
    Precomputed Clopper-Pearson bounds table.
  """
  cp_table = np.zeros(n + 1, dtype=np.float64)
  k = np.arange(1, n + 1, dtype=np.float64)
  a = k
  b = n - k + 1
  cp_table[1:] = sc_special.betaincinv(a, b, epsilon)
  return cp_table


def verify_approx_gdp_algorithm4p5(
    seed,
    mu_input,
    n,
    gamma,
    tau,
    mu_grid,
    cp_table=None,
):
  """Implements Algorithm 4.5: Approximate mu-GDP Verification.

  Args:
    seed: PRNG seed.
    mu_input: Input mechanism privacy parameter.
    n: Sample size.
    gamma: Failure probability.
    tau: Tail probability cutoff.
    mu_grid: Grid of target mu values to check.
    cp_table: Optional precomputed Clopper-Pearson table.

  Returns:
    Tuple of (passed_grid, alpha_LB, beta_LB, thresholds).
  """
  # 1. Individual failure probability via union bound
  epsilon = gamma / (np.float64(2.0) * n)

  # 2. Clopper-Pearson table
  if cp_table is None:
    cp_table = compute_clopper_pearson_table(n, epsilon)

  # 3. Sample continuous privacy losses from P and Q
  rng = np.random.default_rng(seed)
  y = rng.normal(mu_input, 1.0, size=(n,))
  L = y * mu_input - np.float64(0.5) * (mu_input**2)
  del y

  y_prime = rng.normal(0.0, 1.0, size=(n,))
  # Standard privacy loss L' from Q: ln(P(y') / Q(y'))
  L_prime = y_prime * mu_input - np.float64(0.5) * (mu_input**2)
  del y_prime

  # Sort privacy losses
  L_sorted = np.sort(L)
  L_prime_sorted = np.sort(L_prime)

  # 4. Thresholds t: concatenate continuous sample losses L and L_prime
  t = np.sort(np.concatenate([L, L_prime]))
  del L, L_prime

  # 5. Lower bounds for beta(t) and alpha(t) at each sample threshold t
  # k_beta = count of L_i <= t
  k_beta = np.searchsorted(L_sorted, t, side="right")
  # k_alpha = count of L'_i > t (samples from Q where privacy loss > t)
  k_alpha = n - np.searchsorted(L_prime_sorted, t, side="right")

  # 6. Clopper-Pearson Lower Bounds
  beta_LB = cp_table[k_beta]
  alpha_LB = cp_table[k_alpha]

  # 7. Verify trade-off condition with delta relaxation:
  # beta_LB(t) >= G_mu(alpha_LB(t)) - delta
  passed = []
  for mu in mu_grid:
    delta = compute_delta(tau, mu)
    G_vals = G_mu(alpha_LB, mu)
    failures = beta_LB < (G_vals - delta)
    failed_any = np.any(failures)
    passed.append(~failed_any)

  return np.array(passed), alpha_LB, beta_LB, t


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
  print("ALGORITHM 4.5 PARAMETER CALCULATIONS:")
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

  t0 = time.time()
  cp_table = compute_clopper_pearson_table(n, epsilon)
  t1 = time.time()
  print(f"Computed Clopper-Pearson table (size {n+1:,}) in {t1 - t0:.3f}s")
  print(f"  • CP(1; n, epsilon)   : {float(cp_table[1]):.6e}")
  print(f"  • CP(n; n, epsilon)   : {float(cp_table[-1]):.6e}")
  print("-" * 65)

  # 2. Run Algorithm 4.5 Verification over mu_grid
  mu_grid = np.arange(
      mu_input,
      mu_max + mu_step / np.float64(2.0),
      mu_step,
      dtype=np.float64,
  )

  t2 = time.time()
  passed_grid, alpha_LB, beta_LB, t_eval = verify_approx_gdp_algorithm4p5(
      seed, mu_input, n, gamma, tau, mu_grid, cp_table=cp_table
  )
  t3 = time.time()

  print(f"\nAUDIT RESULTS Algorithm 4.5 (Target mu sweep) took {t3 - t2:.2f}s:")
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
    print(f"Smallest target mu that passed (Alg 4.5): {smallest_passed_mu:.3f}")
  else:
    print("None of the target mu values passed.")
  print("=" * 65)

  # Save result to cache
  result_key = f"mu{mu_input}_n{n}_g{gamma}_t{tau}_s{seed}"
  result_data = {
      "smallest_passed_mu": smallest_passed_mu,
      "epsilon": float(epsilon),
      "num_thresholds": len(t_eval),
  }

  if smallest_passed_mu is not None:
    delta_val = float(compute_delta(tau, smallest_passed_mu))
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
    result_data["delta"] = delta_val

  cache["results"][result_key] = result_data
  with open(cache_file, "w") as f:
    json.dump(cache, f, indent=2)
  print(f"Saved result to cache: {cache_file}")


if __name__ == "__main__":
  app.run(main)
