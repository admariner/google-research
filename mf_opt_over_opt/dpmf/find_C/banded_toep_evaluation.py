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

"""Stable banded Toeplitz matrix construction and inverse evaluation."""

# pylint: disable=invalid-name
import torch
import torch.nn.functional as F


def get_stable_banded_toeplitz(v, T, alpha=0.99):
  """Builds a stable banded Toeplitz matrix and its inverse.

  Args:
    v: Free unconstrained parameters of shape (b-1,) optimized by Adam.
    T: Total dimension of the matrix.
    alpha: Safety margin for stability.

  Returns:
    A tuple of (C_final, C_final_inv).
  """
  b_minus_1 = v.shape[0]
  b = b_minus_1 + 1

  # 1. Reparameterization to Stable Roots (Levinson-Durbin)
  k = alpha * torch.tanh(v)

  a = torch.ones(1, device=v.device)
  for i in range(b_minus_1):
    a_padded = F.pad(a, (0, 1))
    a_rev_padded = F.pad(torch.flip(a, dims=[0]), (1, 0))
    a = a_padded + k[i] * a_rev_padded

  # 2. Strict Norm-1 Enforcement (For the Bulk of the Matrix)
  c = a / torch.norm(a, p=2)

  # 3. Vectorized Column Norms (For the Truncated Boundary)
  c_sq_cumsum = torch.cumsum(c**2, dim=0)
  # Number of elements present in each column: min(b, T - j)
  lens = torch.clamp(T - torch.arange(T, device=v.device), max=b)
  col_norms = torch.sqrt(c_sq_cumsum[lens - 1])

  # 4. Construct Unscaled Banded Toeplitz Matrix
  C_toep = torch.zeros(T, T, device=v.device)
  i, j = torch.tril_indices(T, T)

  band_mask = (i - j) < b
  i_band, j_band = i[band_mask], j[band_mask]
  C_toep[i_band, j_band] = c[i_band - j_band]

  # 5. Build Final Matrices (With Fast Inverse Trick)
  # C = C_toep * D^{-1} (Divides boundary columns by their truncated norms)
  C_final = C_toep / col_norms.unsqueeze(0)

  # C^{-1} = D * C_toep^{-1}
  # Solve for the 1st column of C_toep^{-1}
  e1 = torch.zeros(T, 1, device=v.device)
  e1[0] = 1.0
  c_inv_col = torch.linalg.solve_triangular(
      C_toep, e1, upper=False
  ).squeeze(-1)

  # Populate the full dense Toeplitz inverse instantly
  C_inv_toep = torch.zeros(T, T, device=v.device)
  C_inv_toep[i, j] = c_inv_col[i - j]

  # Scale ROWS of the inverse by D (shrinking the boundary edges safely)
  C_final_inv = col_norms.unsqueeze(1) * C_inv_toep

  return C_final, C_final_inv
