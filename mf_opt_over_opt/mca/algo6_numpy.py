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

"""Algorithm 6: Support-function Monte Carlo verifier for aGDP.

NumPy version.
"""

# pylint: disable=invalid-name
import time
from absl import app
from absl import flags
import numpy as np
import scipy.stats as sc_stats

_MU_INPUT = flags.DEFINE_float(
    "mu_input",
    1.0,
    "Actual mechanism privacy parameter (start of search grid)",
)
_MU_MAX = flags.DEFINE_float("mu_max", 2.0, "Maximum mu to search in grid")
_MU_STEP = flags.DEFINE_float("mu_step", 0.05, "Step size for mu grid")
_DELTA_TARGET = flags.DEFINE_float(
    "delta_target", 1e-5, "Target privacy parameter delta"
)
_N = flags.DEFINE_integer("n", 100000, "Sample size for verification")
_GAMMA = flags.DEFINE_float("gamma", 1e-5, "Monte Carlo failure probability")
_R = flags.DEFINE_integer("R", 100, "Refinement budget")
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")


def H_mu(lam, mu):
  """Equation 7: H_mu(lambda)."""
  log_lam = np.log(lam)
  val1 = sc_stats.norm.cdf(log_lam / mu - mu / np.float64(2.0))
  # Use logcdf to avoid precision issues with large lam
  log_cdf_val2 = sc_stats.norm.logcdf(-log_lam / mu - mu / np.float64(2.0))
  val2 = np.exp(log_lam + log_cdf_val2)
  return val1 + val2


def h_mu_delta(lam, mu, delta):
  """Equation 6: support function of target privacy curve."""
  z_delta = sc_stats.norm.ppf(delta)
  alpha_delta = sc_stats.norm.cdf(-z_delta - mu)
  lam_delta = np.exp(mu * z_delta + (mu ** np.float64(2.0)) / np.float64(2.0))

  if np.isscalar(lam):
    if lam <= lam_delta:
      return lam * alpha_delta
    else:
      return H_mu(lam, mu) - delta
  else:
    res = np.zeros_like(lam, dtype=np.float64)
    mask = lam <= lam_delta
    res[mask] = lam[mask] * alpha_delta
    res[~mask] = H_mu(lam[~mask], mu) - delta
    return res


def compute_psi_and_theta(X, m):
  """Equation 10: Maximize sum log(1 + theta(X_i - m))."""
  # Guard against m too close to 0 or 1
  m = np.clip(m, np.float64(1e-10), np.float64(1.0 - 1e-10))

  # Valid domain for theta is (-1/(X_max - m), 1/(m - X_min))
  X_max = np.max(X)
  X_min = np.min(X)

  # We want to optimize over A_m = [-1/(1-m), 1/m]
  theta_L = np.float64(-1.0) / (np.float64(1.0) - m)
  theta_R = np.float64(1.0) / m

  # Trim slightly to avoid division by zero if X_max=1 or X_min=0
  if X_max > m:
    theta_L = np.maximum(
        theta_L, np.float64(-1.0) / (X_max - m) + np.float64(1e-9)
    )
  if X_min < m:
    theta_R = np.minimum(
        theta_R, np.float64(1.0) / (m - X_min) - np.float64(1e-9)
    )

  def get_derivatives(theta):
    u = (X - m) / (np.float64(1.0) + theta * (X - m))
    return np.sum(u), -np.sum(u**2)

  val_df_L, _ = get_derivatives(theta_L)
  if val_df_L <= 0:
    opt_theta = theta_L
    psi = np.sum(np.log(np.float64(1.0) + opt_theta * (X - m)))
    return psi, opt_theta

  val_df_R, _ = get_derivatives(theta_R)
  if val_df_R >= 0:
    opt_theta = theta_R
    psi = np.sum(np.log(np.float64(1.0) + opt_theta * (X - m)))
    return psi, opt_theta

  # Maintain a bracket for safety
  low = theta_L
  high = theta_R

  # Start at midpoint
  theta = (low + high) / np.float64(2.0)

  for _ in range(20):  # Newton converges fast
    val_df, val_d2f = get_derivatives(theta)

    if np.abs(val_df) < np.float64(1e-8):
      break

    # Update bracket
    if val_df > 0:
      low = theta
    else:
      high = theta

    # Newton step
    delta = -val_df / val_d2f
    new_theta = theta + delta

    # If Newton step is within bracket, use it; otherwise bisect
    if new_theta > low and new_theta < high:
      theta = new_theta
    else:
      theta = (low + high) / np.float64(2.0)

  opt_theta = theta
  psi = np.sum(np.log(np.float64(1.0) + opt_theta * (X - m)))
  return psi, opt_theta


def compute_psi_inv(X, C):
  """Equation 11: Inf { m in [0,1] : psi(X, m) <= C }."""
  mean_X = np.mean(X)
  if mean_X <= np.float64(1e-10):
    return np.float64(0.0)

  # Check psi at m close to 0
  psi_at_0, _ = compute_psi_and_theta(X, np.float64(1e-8))
  if psi_at_0 <= C:
    return np.float64(0.0)

  low = np.float64(0.0)
  high = mean_X

  # Start slightly below mean_X to avoid theta=0 division if mean_X is chosen
  m = mean_X * 0.9
  N = X.shape[0]

  for _ in range(20):
    psi, theta_opt = compute_psi_and_theta(X, m)

    if np.abs(psi - C) < np.float64(1e-8):
      break

    # Update bracket
    if psi <= C:
      high = m
    else:
      low = m

    # Newton step: d(psi)/dm = -N * theta
    if theta_opt <= np.float64(1e-12):
      m = (low + high) / np.float64(2.0)
      continue

    delta = -(psi - C) / (-N * theta_opt)
    new_m = m + delta

    if new_m > low and new_m < high:
      m = new_m
    else:
      m = (low + high) / np.float64(2.0)

  return m


def compute_lower_band(s_0, L, gamma, n, mode="P"):
  """Computes F_P(s_0) or F_Q(r_0)."""

  def obj(u):
    if u <= 0 or u >= s_0:
      return np.float64(0.0)
    lam_u = u / (np.float64(1.0) - u)
    if mode == "P":
      X = np.minimum(np.float64(1.0), lam_u * np.exp(-L))
    else:  # mode == 'Q'
      X = np.minimum(np.float64(1.0), lam_u * np.exp(L))

    C = np.log(np.float64(2.0) * np.sqrt(n) / ((s_0 - u) * gamma))
    return compute_psi_inv(X, C)

  # Maximize obj(u) over (0, s_0)
  u_grid = np.linspace(np.float64(1e-4) * s_0, np.float64(0.99) * s_0, 5)
  m_vals = [obj(u) for u in u_grid]
  return np.max(m_vals)


def verify_algo6_grid(
    seed, mu_input, mu_grid, delta_target, n, gamma, R, initial_knots=None
):
  """Verifies approximate GDP across target mu grid using Algorithm 6."""
  rng = np.random.default_rng(seed)

  # 1. Sample and compute privacy losses
  y = rng.normal(mu_input, np.float64(1.0), size=(n,))
  L_P = y * mu_input - np.float64(0.5) * (mu_input ** np.float64(2.0))
  del y

  y_prime = rng.normal(np.float64(0.0), np.float64(1.0), size=(n,))
  L_Q = y_prime * mu_input - np.float64(0.5) * (mu_input ** np.float64(2.0))
  del y_prime

  # 2. Confidence band budget
  gamma_P = gamma / np.float64(2.0)
  gamma_Q = gamma / np.float64(2.0)

  # Global cache for H_bar evaluations
  H_cache = {np.float64(0.0): np.float64(0.0)}

  # Helper to compute H_bar(lambda)
  def H_bar(lam):
    if lam in H_cache:
      return H_cache[lam]

    if lam <= np.float64(0.0):
      return np.float64(0.0)
    s = lam / (np.float64(1.0) + lam)
    r = np.float64(1.0) / (np.float64(1.0) + lam)

    H_P = compute_lower_band(s, L_P, gamma_P, n, mode="P")
    H_Q_bar = compute_lower_band(r, L_Q, gamma_Q, n, mode="Q")
    H_Q = lam * H_Q_bar

    val = np.maximum(H_P, H_Q)
    H_cache[lam] = val
    return val

  passed_grid = []

  for mu_target in mu_grid:
    print(f"Verifying target mu = {mu_target:.3f}")

    # 3. Knots
    if initial_knots is None:
      # Choose knots in log space
      knots = np.logspace(
          np.float64(-2.0), np.float64(2.0), 10, dtype=np.float64
      )
    else:
      knots = np.copy(initial_knots)

    # Evaluate H at knots
    print("Evaluating H_bar at initial knots...")
    ell = np.array([H_bar(lam) for lam in knots])

    # Prepend (0,0)
    knots = np.concatenate([[np.float64(0.0)], knots])
    ell = np.concatenate([[np.float64(0.0)], ell])

    # 4. Refinement loop
    passed = False
    for r in range(R):
      # Vectorized refinement
      lam_j = knots[:-1]
      lam_j1 = knots[1:]
      ell_j = ell[:-1]
      ell_j1 = ell[1:]

      S = (ell_j1 - ell_j) / (lam_j1 - lam_j)

      # Calculate alpha_delta for current mu_target
      z_delta = sc_stats.norm.ppf(delta_target)
      alpha_delta = sc_stats.norm.cdf(-z_delta - mu_target)

      min_lams = np.zeros_like(S)

      # Case 1: S <= 0
      mask_le_0 = S <= 0
      min_lams[mask_le_0] = lam_j1[mask_le_0]

      # Case 2: S >= alpha_delta
      mask_ge_alpha = S >= alpha_delta
      min_lams[mask_ge_alpha] = lam_j[mask_ge_alpha]

      # Case 3: 0 < S < alpha_delta
      mask_mid = ~mask_le_0 & ~mask_ge_alpha
      if np.any(mask_mid):
        z_S = sc_stats.norm.ppf(S[mask_mid])
        lam_star = np.exp(
            -mu_target * z_S
            - (mu_target ** np.float64(2.0)) / np.float64(2.0)
        )
        min_lams[mask_mid] = np.clip(
            lam_star, lam_j[mask_mid], lam_j1[mask_mid]
        )

      # Compute chord values at min_lams
      chord_vals = ell_j + S * (min_lams - lam_j)
      min_vals = chord_vals - h_mu_delta(min_lams, mu_target, delta_target)

      # Identify uncertified intervals
      uncertified_mask = min_vals < np.float64(-1e-8)

      if not np.any(uncertified_mask):
        # Check tail condition
        if ell[-1] >= np.float64(1.0) - delta_target:
          passed = True
          break
        else:
          print(
              f"Warning: Tail condition failed: ell_K = {ell[-1]:.4f} < 1 -"
              f" delta = {1.0 - delta_target:.4f}"
          )

      if not np.any(uncertified_mask):
        # If we are here, it means g_j >= 0 but tail failed.
        # We can try to extend the knot range.
        new_lam = knots[-1] * np.float64(2.0)
        print(f"Extending knot range to {new_lam}")
        new_ell = H_bar(new_lam)
        knots = np.concatenate([knots, [new_lam]])
        ell = np.concatenate([ell, [new_ell]])
        continue

      # If there are uncertified intervals, try to add the worst one that is
      # not a duplicate
      added_knot = False

      uncertified_indices = np.where(uncertified_mask)[0]
      sorted_order = np.argsort(min_vals[uncertified_mask])
      sorted_uncertified_indices = uncertified_indices[sorted_order]

      for idx in sorted_uncertified_indices:
        new_lam = min_lams[idx]
        # Check if duplicate
        if np.min(np.abs(knots - new_lam)) > np.float64(1e-6):
          new_ell = H_bar(new_lam)
          insert_idx = np.searchsorted(knots, new_lam)
          knots = np.insert(knots, insert_idx, new_lam)
          ell = np.insert(ell, insert_idx, new_ell)
          print(
              f"Refinement step {r+1}: Added knot {new_lam:.4f}, H_bar ="
              f" {new_ell:.4f}, min_val = {min_vals[idx]:.4f}"
          )
          added_knot = True
          break

      if not added_knot:
        print(
            "Could not add any new knot (duplicates/too close). Failing"
            " refinement for this mu."
        )
        break

    passed_grid.append(passed)

  return np.array(passed_grid)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  mu_input = np.float64(_MU_INPUT.value)
  mu_max = np.float64(_MU_MAX.value)
  mu_step = np.float64(_MU_STEP.value)
  delta_target = np.float64(_DELTA_TARGET.value)
  n = int(_N.value)
  gamma = np.float64(_GAMMA.value)
  R = int(_R.value)
  seed = int(_SEED.value)

  mu_grid = np.arange(
      mu_input,
      mu_max + mu_step / np.float64(2.0),
      mu_step,
      dtype=np.float64,
  )

  print("=" * 65)
  print("ALGORITHM 6 PARAMETER CALCULATIONS:")
  print("=" * 65)
  print("Given Inputs:")
  print(f"  • Mechanism parameter mu_in : {mu_input}")
  print(f"  • Target parameter mu start : {mu_input}")
  print(f"  • Target parameter mu max   : {mu_max}")
  print(f"  • Target parameter mu step  : {mu_step}")
  print(f"  • Target parameter delta    : {delta_target}")
  print(f"  • Sample size (n)           : {n:,}")
  print(f"  • Failure prob (gamma)      : {gamma}")
  print(f"  • Refinement budget (R)     : {R}")
  print("-" * 65)

  t0 = time.time()
  passed_grid = verify_algo6_grid(
      seed, mu_input, mu_grid, delta_target, n, gamma, R
  )
  t1 = time.time()

  print(f"\nAUDIT RESULTS Algorithm 6 took {t1 - t0:.2f}s:")
  print("-" * 65)
  for mu_target, passed in zip(mu_grid, passed_grid):
    print(
        f"Target mu = {mu_target:.3f} | delta = {delta_target:.2e} |"
        f" {'PASSED' if passed else 'FAILED'}"
    )
  print("-" * 65)

  valid_mu = mu_grid[passed_grid]
  if len(valid_mu) > 0:
    print(f"Smallest target mu that passed: {valid_mu[0]:.3f}")
  else:
    print("None of the target mu values passed.")
  print("=" * 65)


if __name__ == "__main__":
  app.run(main)
