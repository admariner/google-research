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

JAX/Multi-GPU version.
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
import scipy.stats as sc_stats

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)


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
  log_lam = jnp.log(lam)
  val1 = jax_stats.norm.cdf(log_lam / mu - mu / 2.0)
  # Use logcdf to avoid precision issues with large lam
  log_cdf_val2 = jax_stats.norm.logcdf(-log_lam / mu - mu / 2.0)
  val2 = jnp.exp(log_lam + log_cdf_val2)
  return val1 + val2


def h_mu_delta(lam, mu, delta):
  """Equation 6: support function of target privacy curve."""
  z_delta = jax_stats.norm.ppf(delta)
  alpha_delta = jax_stats.norm.cdf(-z_delta - mu)
  lam_delta = jnp.exp(mu * z_delta + (mu**2.0) / 2.0)

  return jnp.where(lam <= lam_delta, lam * alpha_delta, H_mu(lam, mu) - delta)


def compute_psi_and_theta_local(X_shard, m, axis_name="data"):
  """Calculates psi and optimal theta for a shard of X."""

  # Guard against m too close to 0 or 1
  m = jnp.clip(m, 1e-10, 1.0 - 1e-10)

  local_min = jnp.min(X_shard)
  local_max = jnp.max(X_shard)
  global_min = jax.lax.pmin(local_min, axis_name=axis_name)
  global_max = jax.lax.pmax(local_max, axis_name=axis_name)

  # Valid domain for theta is (-1/(X_max - m), 1/(m - X_min))
  theta_L = -1.0 / (1.0 - m)
  theta_R = 1.0 / m

  # Trim slightly to avoid division by zero if X_max=1 or X_min=0
  theta_L = jnp.where(
      global_max > m,
      jnp.maximum(theta_L, -1.0 / (global_max - m) + 1e-9),
      theta_L,
  )
  theta_R = jnp.where(
      global_min < m,
      jnp.minimum(theta_R, 1.0 / (m - global_min) - 1e-9),
      theta_R,
  )

  def get_derivatives(theta):
    u = (X_shard - m) / (1.0 + theta * (X_shard - m))
    local_df = jnp.sum(u)
    local_d2f = -jnp.sum(u**2)
    global_df = jax.lax.psum(local_df, axis_name=axis_name)
    global_d2f = jax.lax.psum(local_d2f, axis_name=axis_name)
    return global_df, global_d2f

  val_df_L, _ = get_derivatives(theta_L)
  val_df_R, _ = get_derivatives(theta_R)

  def optimize_theta(_):
    # Maintain a bracket for safety
    def body_fun(i, state):
      del i
      theta, low, high = state
      val_df, val_d2f = get_derivatives(theta)

      # Update bracket
      low = jnp.where(val_df > 0, theta, low)
      high = jnp.where(val_df <= 0, theta, high)

      # Newton step
      delta = -val_df / val_d2f
      new_theta = theta + delta

      # If Newton step is within bracket, use it; otherwise bisect
      use_new = (new_theta > low) & (new_theta < high)
      theta = jnp.where(use_new, new_theta, (low + high) / 2.0)

      return theta, low, high

    theta_init = (theta_L + theta_R) / 2.0
    state = (theta_init, theta_L, theta_R)
    # 20 steps is usually enough for Newton
    final_state = jax.lax.fori_loop(0, 20, body_fun, state)
    return final_state[0]

  # Conditional execution based on boundary derivatives
  opt_theta = jnp.where(
      val_df_L <= 0,
      theta_L,
      jnp.where(val_df_R >= 0, theta_R, optimize_theta(None)),
  )

  psi_local = jnp.sum(jnp.log(1.0 + opt_theta * (X_shard - m)))
  psi_global = jax.lax.psum(psi_local, axis_name=axis_name)

  return psi_global, opt_theta


def compute_psi_inv_local(X_shard, C, mean_X, axis_name="data"):
  """Solves for m such that psi(X, m) <= C."""

  N_local = X_shard.shape[0]
  N_global = jax.lax.psum(N_local, axis_name=axis_name)

  # Initial bracket
  low = 0.0
  high = mean_X

  # Start slightly below mean_X
  m = mean_X * 0.9

  def body_fun(i, state):
    del i
    m, low, high = state
    psi, theta_opt = compute_psi_and_theta_local(
        X_shard, m, axis_name=axis_name
    )

    # Update bracket
    is_less = psi <= C
    high = jnp.where(is_less, m, high)
    low = jnp.where(is_less, low, m)

    # Newton step: d(psi)/dm = -N * theta
    use_bisect = theta_opt <= 1e-12

    delta = -(psi - C) / (-N_global * theta_opt)
    new_m = m + delta

    use_new = (new_m > low) & (new_m < high) & (~use_bisect)

    m = jnp.where(use_new, new_m, (low + high) / 2.0)

    return m, low, high

  state = (m, low, high)
  # 20 steps is usually enough for Newton
  final_state = jax.lax.fori_loop(0, 20, body_fun, state)
  return final_state[0]


@functools.partial(
    jax.pmap,
    axis_name="data",
    in_axes=(None, 0, None, None, None, None),
    static_broadcasted_argnums=(3, 4, 5),
)
def eval_obj_shard(u, L_shard, s_0, gamma, n, mode):
  """Evaluates the objective function on a single device/shard."""
  lam_u = u / (1.0 - u)
  if mode == "P":
    X_shard = jnp.minimum(1.0, lam_u * jnp.exp(-L_shard))
  else:
    X_shard = jnp.minimum(1.0, lam_u * jnp.exp(L_shard))

  C = jnp.log(2.0 * jnp.sqrt(n) / ((s_0 - u) * gamma))

  local_sum = jnp.sum(X_shard)
  global_sum = jax.lax.psum(local_sum, axis_name="data")
  N_local = X_shard.shape[0]
  N_global = jax.lax.psum(N_local, axis_name="data")
  mean_X = global_sum / N_global

  # Handle case where mean_X is too close to 0
  # In JAX we avoid early returns, but we can compute and return m_opt
  m_opt = compute_psi_inv_local(X_shard, C, mean_X)

  return jnp.where(mean_X <= 1e-10, 0.0, m_opt)


@functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def generate_L_P_worker(key, mu_input, shard_size):
  y = jax.random.normal(key, shape=(shard_size,)) + mu_input
  L = y * mu_input - 0.5 * (mu_input**2.0)
  return L


@functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def generate_L_Q_worker(key, mu_input, shard_size):
  y = jax.random.normal(key, shape=(shard_size,))
  L = y * mu_input - 0.5 * (mu_input**2.0)
  return L


def compute_lower_band(s_0, L_sharded, gamma, n, mode, num_devices):
  """Computes F_P(s_0) or F_Q(r_0) handling sharded data."""
  del num_devices

  def obj(u):
    if u <= 0 or u >= s_0:
      return 0.0

    # Broadcast u to all devices? pmap handles scalar u easily
    m_opt_sharded = eval_obj_shard(u, L_sharded, s_0, gamma, n, mode)
    # All devices return the same value, take first
    return float(m_opt_sharded[0])

  # Maximize obj(u) over (0, s_0)
  u_grid = np.linspace(1e-4 * s_0, 0.99 * s_0, 5)
  m_vals = [obj(u) for u in u_grid]
  return np.max(m_vals)


def verify_algo6_grid(
    seed, mu_input, mu_grid, delta_target, n, gamma, R, initial_knots=None
):
  """Verifies approximate GDP across target mu grid using Algorithm 6."""
  num_devices = jax.local_device_count()
  if n % num_devices != 0:
    raise ValueError(
        f"Sample size n ({n}) must be divisible by number of devices"
        f" ({num_devices})"
    )

  shard_size = n // num_devices

  key = jax.random.PRNGKey(seed)
  key_P, key_Q = jax.random.split(key)

  # 1. Sample and compute privacy losses (Data Parallel Generation)
  keys_P = jax.random.split(key_P, num_devices)
  L_P_sharded = generate_L_P_worker(keys_P, mu_input, shard_size)

  keys_Q = jax.random.split(key_Q, num_devices)
  L_Q_sharded = generate_L_Q_worker(keys_Q, mu_input, shard_size)

  # 2. Confidence band budget
  gamma_P = gamma / 2.0
  gamma_Q = gamma / 2.0

  # Global cache for H_bar evaluations
  H_cache = {0.0: 0.0}

  # Helper to compute H_bar(lambda)
  def H_bar(lam):
    if lam in H_cache:
      return H_cache[lam]

    if lam <= 0.0:
      return 0.0
    s = lam / (1.0 + lam)
    r = 1.0 / (1.0 + lam)

    H_P = compute_lower_band(
        s, L_P_sharded, gamma_P, n, mode="P", num_devices=num_devices
    )
    H_Q_bar = compute_lower_band(
        r, L_Q_sharded, gamma_Q, n, mode="Q", num_devices=num_devices
    )
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
      knots = np.logspace(-2.0, 2.0, 10)
    else:
      knots = np.copy(initial_knots)

    # Evaluate H at knots
    print("Evaluating H_bar at initial knots...")
    ell = np.array([H_bar(lam) for lam in knots])

    # Prepend (0,0)
    knots = np.concatenate([[0.0], knots])
    ell = np.concatenate([[0.0], ell])

    # 4. Refinement loop
    passed = False
    for r in range(R):
      # Vectorized refinement (keeps using NumPy/analytical solution as it's
      # fast and operates on few knots)
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
        lam_star = np.exp(-mu_target * z_S - (mu_target**2.0) / 2.0)
        min_lams[mask_mid] = np.clip(
            lam_star, lam_j[mask_mid], lam_j1[mask_mid]
        )

      # Compute chord values at min_lams
      chord_vals = ell_j + S * (min_lams - lam_j)
      min_vals = chord_vals - h_mu_delta(min_lams, mu_target, delta_target)

      # Identify uncertified intervals
      uncertified_mask = min_vals < -1e-8

      if not np.any(uncertified_mask):
        # Check tail condition
        if ell[-1] >= 1.0 - delta_target:
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
        new_lam = knots[-1] * 2.0
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
        if np.min(np.abs(knots - new_lam)) > 1e-6:
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

  mu_input = _MU_INPUT.value
  mu_max = _MU_MAX.value
  mu_step = _MU_STEP.value
  delta_target = _DELTA_TARGET.value
  n = _N.value
  gamma = _GAMMA.value
  R = _R.value
  seed = _SEED.value

  mu_grid = np.arange(
      mu_input,
      mu_max + mu_step / 2.0,
      mu_step,
  )

  print("=" * 65)
  print("ALGORITHM 6 JAX PARAMETER CALCULATIONS:")
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
  print(f"JAX Devices: {jax.local_device_count()}")
  print("=" * 65)

  t0 = time.time()
  passed_grid = verify_algo6_grid(
      seed, mu_input, mu_grid, delta_target, n, gamma, R
  )
  t1 = time.time()
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
      "R": int(R),
      "time_seconds": float(t1 - t0),
  })

  for step_idx, (mu_target, passed) in enumerate(zip(mu_grid, passed_grid)):
    # Should write instead of print in the future.
    print({
        "step": int(step_idx),
        "n": int(n),
        "mu_input": float(mu_input),
        "mu_target": float(mu_target),
        "passed": bool(passed),
        "delta_target": float(delta_target),
        "smallest_passed_mu": (
            float(smallest_passed_mu)
            if smallest_passed_mu is not None
            else -1.0
        ),
        "gamma": float(gamma),
    })

  print(f"\nAUDIT RESULTS Algorithm 6 JAX took {t1 - t0:.2f}s:")
  print("-" * 65)
  for mu_target, passed in zip(mu_grid, passed_grid):
    print(
        f"Target mu = {mu_target:.3f} | delta = {delta_target:.2e} |"
        f" {'PASSED' if passed else 'FAILED'}"
    )
  print("-" * 65)

  if smallest_passed_mu is not None:
    print(f"Smallest target mu that passed: {smallest_passed_mu:.3f}")
  else:
    print("None of the target mu values passed.")
  print("=" * 65)


if __name__ == "__main__":
  app.run(main)
