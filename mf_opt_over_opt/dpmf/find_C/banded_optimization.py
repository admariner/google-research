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

"""Stable banded matrix construction for PyTorch."""

# pylint: disable=invalid-name
import torch


def construct_stable_banded_C(V, b, alpha=0.99):
  """Maps unconstrained parameters to a stable lower triangular banded matrix C.

  Args:
    V: Free unconstrained parameters of shape (T, b-1).
    b: Band width.
    alpha: Strict contraction factor in (0, 1).

  Returns:
    The assembled matrix C of shape (T, T).
  """
  T = V.shape[0]

  # 1. Mask out unused parameters at the bottom right of the matrix
  # (e.g., the very last column has 0 sub-diagonals below it)
  col_indices = torch.arange(T).unsqueeze(1)
  sub_indices = torch.arange(1, b).unsqueeze(0)
  mask = (col_indices + sub_indices) < T
  V = V * mask.to(V.device).float()

  # 2. Calculate the Custom Norm N(V) for each column
  L1 = torch.norm(V, p=1, dim=1)
  L2 = torch.norm(V, p=2, dim=1)
  N = torch.sqrt(L1**2 + L2**2)

  # 3. Safe Squash to Region (torch.where handles the N=0 limit seamlessly)
  scale = alpha * torch.where(N < 1e-6, torch.ones_like(N), torch.tanh(N) / N)
  U = V * scale.unsqueeze(1)  # U represents our bounded off-diagonals x_j

  # 4. Main Diagonal (Mathematically forces exact Column Norm = 1)
  c_diag = torch.sqrt(1.0 - torch.norm(U, p=2, dim=1) ** 2)

  # 5. Assemble the final Matrix C
  C = torch.diag(c_diag)
  for i in range(1, b):
    # Place the bounded variables onto the i-th sub-diagonal
    sub_diag_elements = U[: T - i, i - 1]
    C += torch.diag(sub_diag_elements, diagonal=-i)

  return C
