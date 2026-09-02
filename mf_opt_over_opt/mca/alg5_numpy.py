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

"""Algorithm 5: Approximate mu-GDP Verification with Order Statistics.

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
_NUM_SIMULATIONS = flags.DEFINE_integer(
    "num_simulations", 20000, "Number of MC runs for binary search solving q"
)
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")
_USE_SOLVED_Q = flags.DEFINE_boolean(
    "use_solved_q",
    False,
    "Whether to use solved q (True) or union bound q (False)",
)
_CACHE_FILE = flags.DEFINE_string(
    "cache_file",
    "alg5_cache.json",
    "Path to JSON file for caching q and results",
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


def solve_q(n, gamma, num_simulations=20000, max_steps=40, seed=42):
  """Solves for scalar value q s.t. P(M_n^+ < q) <= gamma / 2.

  Args:
    n: Sample size.
    gamma: Failure probability.
    num_simulations: Number of Monte Carlo runs.
    max_steps: Maximum binary search steps.
    seed: PRNG seed.

  Returns:
    Solved scalar value q.
  """
  rng = np.random.default_rng(seed)
  # 1. Sample and sort Uniform(0, 1) once to obtain order statistics U_{(i)}
  U = rng.uniform(0.0, 1.0, size=(num_simulations, n))
  U_sorted = np.sort(U, axis=1)  # shape: (num_simulations, n)

  i = np.arange(1, n + 1, dtype=np.float64)
  a = i
  b = n - i + 1
  target_prob = gamma / np.float64(2.0)

  # 2. Binary search / root finding on q in (0, 1)
  low = np.float64(1e-15)
  high = np.float64(1.0 - 1e-15)

  for _ in range(max_steps):
    mid = np.float64(0.5) * (low + high)
    # Apply B_i^{-1} to candidate q: ell_i = B_i^{-1}(mid) of shape (n,)
    ell = sc_special.betaincinv(a, b, mid)
    # Check condition: exists i s.t. U_{(i)} < ell_i
    event_occurred = np.any(U_sorted < ell, axis=1)  # shape: (num_simulations,)
    prob = np.mean(event_occurred)

    if prob < target_prob:
      low = mid
    else:
      high = mid

  q = float(low)
  return q


def compute_ell(n, q):
  """Computes ell_i = B_i^{-1}(q) for i = 1, ..., n.

  Args:
    n: Sample size.
    q: Quantile parameter.

  Returns:
    Array of lower bounds for each order statistic.
  """
  i = np.arange(1, n + 1, dtype=np.float64)
  a = i
  b = n - i + 1
  # Inverse incomplete beta function: B_i^{-1}(q)
  ell = sc_special.betaincinv(a, b, q)
  return np.array(ell, dtype=np.float64)


def verify_approx_gdp_algorithm5(
    seed,
    mu_input,
    n,
    gamma,
    tau,
    mu_grid,
    num_simulations=20000,
    q=None,
    ell=None,
):
  """Implements Algorithm 5: Approximate mu-GDP Verification.

  Args:
    seed: PRNG seed.
    mu_input: Input mechanism privacy parameter.
    n: Sample size.
    gamma: Failure probability.
    tau: Tail probability cutoff.
    mu_grid: Grid of target mu values to check.
    num_simulations: Number of Monte Carlo simulations for q.
    q: Optional pre-solved scalar value q.
    ell: Optional precomputed ell bounds array.

  Returns:
    Tuple of (passed_grid, alpha_LB, beta_LB).
  """
  # 1. Solve for scalar q via binary search and compute ell_i = B_i^{-1}(q)
  if ell is None:
    if q is None:
      q = solve_q(n, gamma, num_simulations=num_simulations, seed=seed)
    ell = compute_ell(n, q)

  # ell_extended has 0.0 at index 0 to enforce the convention that
  # the lower bound is 0 when no sample is <= query threshold
  ell_extended = np.concatenate([np.array([0.0], dtype=np.float64), ell])

  # 2. Sample from P and Q in float64
  rng = np.random.default_rng(seed)
  y = rng.normal(mu_input, 1.0, size=(n,))

  # 3. Privacy loss
  # L from P: ln(P(y) / Q(y))
  L = y * mu_input - np.float64(0.5) * (mu_input**2)
  del y

  y_prime = rng.normal(0.0, 1.0, size=(n,))
  # L_prime from Q: ln(Q(y') / P(y')) (with negative sign added to L_prime)
  L_prime = -(y_prime * mu_input - np.float64(0.5) * (mu_input**2))
  del y_prime

  # Sort privacy losses
  L_sorted = np.sort(L)
  L_prime_sorted = np.sort(L_prime)

  # 4. Thresholds t: concatenate L and L_prime, then sort them
  t = np.sort(np.concatenate([L, L_prime]))
  del L, L_prime

  # 5. Lower bounds for beta(t) and alpha(t) at each threshold t
  # beta_LB(t) = max of ell_i s.t. L_i <= t (0 if none)
  k_beta = np.searchsorted(L_sorted, t, side="right")
  beta_LB = ell_extended[k_beta]

  # alpha_LB(t) follows the exact same routine as beta, feeding in -t
  # alpha_LB(t) = max of ell_i s.t. L'_i <= -t (0 if none)
  k_alpha = np.searchsorted(L_prime_sorted, -t, side="right")
  alpha_LB = ell_extended[k_alpha]

  # 6. Verify trade-off condition with delta relaxation:
  # beta_LB(t) >= G_mu(alpha_LB(t)) - delta
  passed = []
  for mu in mu_grid:
    delta = compute_delta(tau, mu)
    G_vals = G_mu(alpha_LB, mu)
    failures = beta_LB < (G_vals - delta)
    failed_any = np.any(failures)
    passed.append(~failed_any)

  return np.array(passed), alpha_LB, beta_LB


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  mu_input = np.float64(_MU_INPUT.value)
  n = int(_N.value)
  gamma = np.float64(_GAMMA.value)
  tau = np.float64(_TAU.value)
  num_simulations = _NUM_SIMULATIONS.value
  seed = _SEED.value
  use_solved_q = _USE_SOLVED_Q.value
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

  cache = {"q_cache": {}, "results": {}}

  if os.path.exists(cache_file):
    try:
      with open(cache_file, "r") as f:
        cache = json.load(f)
        if "q_cache" not in cache:
          cache["q_cache"] = {}
        if "results" not in cache:
          cache["results"] = {}
    except json.JSONDecodeError:
      print(
          f"Warning: Could not decode {cache_file}, starting with empty cache."
      )

  print("=" * 65)
  print("ALGORITHM 5 PARAMETER CALCULATIONS:")
  print("=" * 65)
  print("Given Inputs:")
  print(f"  • Mechanism parameter mu : {mu_input}")
  print(f"  • Sample size (n)        : {n:,}")
  print(f"  • Failure prob (gamma)   : {gamma}")
  print(f"  • Tail prob cutoff (tau) : {tau}")
  print(f"  • MC simulations for q   : {num_simulations:,}")
  print(f"  • Use solved q           : {use_solved_q}")
  print("-" * 65)

  t0 = time.time()
  # 1. Solve for scalar q via binary search s.t. P(M_n^+ < q) <= gamma / 2
  q_union_bound = gamma / (np.float64(2.0) * n)
  if use_solved_q:
    q_key = f"n{n}_g{gamma}_ns{num_simulations}_s{seed}"
    if q_key in cache["q_cache"]:
      q = cache["q_cache"][q_key]
      print(f"Loaded q from cache: {q:.6e}")
    else:
      print("Solving for q (this may take a while)...")
      q = solve_q(n, gamma, num_simulations=num_simulations, seed=seed)
      cache["q_cache"][q_key] = q
      with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
  else:
    q = q_union_bound
  t1 = time.time()

  # 2. Compute ell_i = B_i^{-1}(q)
  ell = compute_ell(n, q)

  # 3. Calculate relaxation parameter delta for the mechanism parameter mu_input
  delta_input = compute_delta(tau, mu_input)

  print("Calculated Quantities:")
  print(f"  • Scalar root q (gamma/2): {q:.6e} (took {t1 - t0:.2f}s)")
  print(f"  • Union Bound q          : {q_union_bound:.6e}")
  print(f"  • Ratio (q / q_union)    : {q / q_union_bound:.4f}")
  print(f"  • ell_1 (min lower bound): {float(ell[0]):.6e}")
  print(f"  • ell_n (max lower bound): {float(ell[-1]):.6e}")
  print(f"  • Relaxation (delta)     : {float(delta_input):.2e}")
  print("=" * 65)

  # 4. Run Algorithm 5 Verification over mu_grid
  mu_grid = np.arange(
      mu_input,
      mu_max + mu_step / np.float64(2.0),
      mu_step,
      dtype=np.float64,
  )

  t2 = time.time()
  passed_grid, alpha_LB, beta_LB = verify_approx_gdp_algorithm5(
      seed, mu_input, n, gamma, tau, mu_grid, q=q, ell=ell
  )
  t3 = time.time()

  print(f"\nAUDIT RESULTS Algorithm 5 (Target mu sweep) took {t3 - t2:.2f}s:")
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
    print(f"Smallest target mu that passed (Alg 5): {smallest_passed_mu:.3f}")
  else:
    print("None of the target mu values passed.")
  print("=" * 65)

  # Save result to cache
  result_key = f"mu{mu_input}_n{n}_g{gamma}_t{tau}_sq{use_solved_q}_ns{num_simulations}_s{seed}"
  result_data = {
      "smallest_passed_mu": smallest_passed_mu,
      "q": q,
      "q_union_bound": q_union_bound,
      "ratio_q_qunion": q / q_union_bound,
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
