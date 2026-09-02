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

"""Matrix strategies for DP-MF."""

# pylint: disable=invalid-name
import abc
from typing import Any

import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import banded
from jax_privacy.matrix_factorization import streaming_matrix
from jax_privacy.matrix_factorization import toeplitz


class MatrixStrategy(abc.ABC):
  """Abstract base class for a tunable DP noising matrix strategy."""

  @abc.abstractmethod
  def init_params(self):
    """Returns the initial guess for the parameters to optimize."""
    raise NotImplementedError

  @abc.abstractmethod
  def build_noising_matrix(
      self, params
  ):
    """Constructs the DP scaled C^{-1} matrix given the tunable parameters."""
    raise NotImplementedError

  @abc.abstractmethod
  def build_normalized_noising_matrix(
      self, params
  ):
    """Constructs the DP scaled C^{-1} matrix with sensitivity=1."""
    raise NotImplementedError

  def project_params(self, params):
    """Optionally restricts/projects parameters to a safe zone.

    Args:
      params: Strategy parameters.

    Returns:
      Projected parameters.
    """
    return params


class SingleParamToeplitz(MatrixStrategy):
  """A fully correlated Toeplitz strategy using a single multiplicative param c."""

  def __init__(self, T, bands, init_c = 0.5):  # pylint: disable=invalid-name
    self.T = T  # pylint: disable=invalid-name
    self.bands = bands
    self.init_c = init_c

  def init_params(self):
    return jnp.array(self.init_c)

  def build_noising_matrix(
      self, params
  ):
    coef_c_bands = jnp.power(params, jnp.arange(self.bands))
    coef_c = toeplitz.pad_coefs_to_n(coef_c_bands, self.T)
    a = jnp.sqrt(toeplitz.sensitivity_squared(coef_c, n=self.T))

    C_inv = toeplitz.inverse_as_streaming_matrix(coef_c_bands)
    return streaming_matrix.scale_rows_and_columns(
        C_inv, col_scale=jnp.ones(self.T) * a
    )

  def build_normalized_noising_matrix(
      self, params
  ):
    coef_c_bands = jnp.power(params, jnp.arange(self.bands))
    return toeplitz.inverse_as_streaming_matrix(coef_c_bands, self.T)

  def project_params(self, params):
    c_0 = 1.
    rest = jnp.power(params, jnp.arange(1, self.bands))
    sum_abs = jnp.sum(jnp.abs(rest))
    limit = 0.99 * c_0
    scale = jnp.where(sum_abs > limit, limit / (sum_abs + 1e-12), 1.0)
    new_rest = rest * scale
    return new_rest[0]


class MultiParamToeplitz(MatrixStrategy):
  """A multi-band Toeplitz strategy using optimal initialization."""

  def __init__(self, T, bands):  # pylint: disable=invalid-name
    self.T = T  # pylint: disable=invalid-name
    self.bands = bands

  def init_params(self):
    return jnp.array([0.5] + [0.0] * (self.bands - 1))

  @staticmethod
  def multiparamtoep_coef(params):
    """Transforms raw parameters to strategy coefficients (locks c_0 positive)."""
    c_0 = jnp.log(1.0 + jnp.exp(params[0]))
    return jnp.concatenate([jnp.array([c_0]), params[1:]])

  def build_noising_matrix(
      self, params
  ):
    # Lock sign of c_0 using softplus transform
    coefs = self.multiparamtoep_coef(params)
    coef_c = toeplitz.pad_coefs_to_n(coefs, self.T)
    a = jnp.sqrt(toeplitz.sensitivity_squared(coef_c, n=self.T))

    # We use inverse as streaming matrix and scale columns automatically
    C_inv = toeplitz.inverse_as_streaming_matrix(coefs)  # pylint: disable=invalid-name
    return streaming_matrix.scale_rows_and_columns(
        C_inv, col_scale=jnp.ones(self.T) * a
    )

  def build_normalized_noising_matrix(
      self, params
  ):
    coefs = self.multiparamtoep_coef(params)
    return toeplitz.inverse_as_streaming_matrix(coefs, self.T)

  def project_params(self, params):
    c_0 = jnp.log(1.0 + jnp.exp(params[0]))
    rest = params[1:]
    sum_abs = jnp.sum(jnp.abs(rest))
    limit = 0.99 * c_0
    scale = jnp.where(sum_abs > limit, limit / (sum_abs + 1e-12), 1.0)
    new_rest = rest * scale
    return jnp.concatenate([jnp.array([params[0]]), new_rest])


class BandedStrategy(MatrixStrategy):
  """An entirely multi-band parameter layout tracking all bounds."""

  def __init__(self, T, bands):  # pylint: disable=invalid-name
    self.T = T  # pylint: disable=invalid-name
    self.bands = bands

  def init_params(self):
    return banded.ColumnNormalizedBanded.default(
        n=self.T, bands=self.bands
    ).params

  def build_noising_matrix(
      self, params
  ):
    # Notice a ColumnNormalizedBanded strategy
    # naturally implements sensitivity=1
    strategy = banded.ColumnNormalizedBanded(params=params)
    return strategy.inverse_as_streaming_matrix()

  def build_normalized_noising_matrix(
      self, params
  ):
    # Notice a ColumnNormalizedBanded strategy
    # naturally implements sensitivity=1
    strategy = banded.ColumnNormalizedBanded(params=params)
    return strategy.inverse_as_streaming_matrix()


def banded_toeplitz_as_streaming_matrix(
    coef,
):
  """Creates a StreamingMatrix for banded lower triangular Toeplitz matrix."""
  bands = coef.shape[0]

  def init(abstract_yi):
    dtype = jnp.promote_types(abstract_yi.dtype, coef.dtype)
    zero = jnp.zeros_like(abstract_yi, dtype=dtype)
    if bands <= 1:
      return jnp.zeros((0,) + zero.shape, dtype=dtype)
    return jnp.broadcast_to(zero, (bands - 1,) + zero.shape)

  def _next(yi, state):
    if bands == 1:
      return yi * coef[0], state

    inner = jnp.tensordot(coef[1:], state, axes=1)
    xi = yi * coef[0] + inner

    new_state = jnp.roll(state, 1, axis=0).at[0].set(yi)
    return xi, new_state

  return streaming_matrix.StreamingMatrix.from_array_implementation(init, _next)


class TwoParamCinvToeplitz(MatrixStrategy):
  """A strategy directly defining C^{-1} as a banded Toeplitz matrix."""

  def __init__(
      self, T, init_a = 0.5, init_b = 0.5
  ):
    self.T = T
    self.init_a = init_a
    self.init_b = init_b

  def init_params(self):
    return jnp.array([self.init_a, self.init_b])

  def _get_Cinv_coefs(self, params):
    a = params[0]
    b = params[1]
    return jnp.array([1.0, -a, -b])

  def build_noising_matrix(
      self, params
  ):
    return self.build_normalized_noising_matrix(params)

  def build_normalized_noising_matrix(
      self, params
  ):
    Cinv_coefs = self._get_Cinv_coefs(params)

    # Compute coefficients of C
    C_coefs = toeplitz.inverse_coef(Cinv_coefs, n=self.T)

    # Compute column norms of C
    col_norms = jnp.sqrt(jnp.cumsum(C_coefs**2))[::-1]

    # Build Cinv as StreamingMatrix
    Cinv = banded_toeplitz_as_streaming_matrix(Cinv_coefs)

    # Normalize: scale ROWS of Cinv by col_norms of C
    return streaming_matrix.scale_rows_and_columns(Cinv, row_scale=col_norms)

  def project_params(self, params):
    return jnp.clip(params, 0.0, 1.0)


def arma_cinv_streaming_matrix(
    a, b
):
  """Applies C^{-1} using ARMA recurrence.

  Args:
    a: Numerator parameter.
    b: Denominator parameter.

  Returns:
    A StreamingMatrix implementing C^{-1}.
  """

  def init(abstract_yi):
    dtype = jnp.promote_types(abstract_yi.dtype, a.dtype)
    return jnp.zeros_like(abstract_yi, dtype=dtype)

  def _next(yi, state):
    xi = yi + (b - a) * state
    new_state = b * state + yi
    return xi, new_state

  return streaming_matrix.StreamingMatrix.from_array_implementation(init, _next)


class TwoParamDenseToeplitz(MatrixStrategy):
  """A strategy defining C as a dense lower triangular Toeplitz matrix."""

  def __init__(self, T, init_a = 0.5, init_b = 0.5):
    self.T = T
    self.init_a = init_a
    self.init_b = init_b

  def init_params(self):
    return jnp.array([self.init_a, self.init_b])

  def _get_C_coefs(self, params):
    a = params[0]
    b = params[1]
    if self.T <= 0:
      return jnp.array([])
    if self.T == 1:
      return jnp.array([1.0])

    k = jnp.arange(self.T - 1)
    decay = (a - b) * (a**k)
    return jnp.concatenate([jnp.array([1.0]), decay])

  def build_noising_matrix(
      self, params
  ):
    return self.build_normalized_noising_matrix(params)

  def build_normalized_noising_matrix(
      self, params
  ):
    c_coefs = self._get_C_coefs(params)

    # Compute column norms of C
    col_norms = jnp.sqrt(jnp.cumsum(c_coefs**2))[::-1]

    # Build Cinv as StreamingMatrix
    Cinv = arma_cinv_streaming_matrix(params[0], params[1])

    # Normalize: scale ROWS of Cinv by col_norms of C
    return streaming_matrix.scale_rows_and_columns(Cinv, row_scale=col_norms)

  def project_params(self, params):
    return jnp.clip(params, 0.0, 1.0)


def blt_cinv_streaming_matrix(
    buf_decay, output_scale
):
  """Applies C^{-1} for a BLT matrix using Algorithm 3.

  Args:
    buf_decay: Buffer decay parameters of C.
    output_scale: Output scale parameters of C.

  Returns:
    A StreamingMatrix implementing C^{-1}.
  """

  def init(abstract_yi):
    num_buffers = buf_decay.shape[0]
    zero = jnp.zeros_like(abstract_yi, dtype=buf_decay.dtype)
    return jnp.broadcast_to(zero, (num_buffers,) + zero.shape)

  def _read(state):
    return jnp.tensordot(output_scale, state, axes=((0,), (0,)))

  def _update(state, xi):
    decay_expanded = jnp.expand_dims(
        buf_decay, axis=tuple(range(1, state.ndim))
    )
    state_decayed = state * decay_expanded
    return state_decayed + xi

  def next_fn(yi, state):
    xi = yi - _read(state)
    state = _update(state, xi)
    return xi, state

  return streaming_matrix.StreamingMatrix.from_array_implementation(
      init, next_fn
  )


class BLTStrategy(MatrixStrategy):
  """A strategy using Buffered Linear Toeplitz (BLT) matrices."""

  def __init__(self, T, num_buffers):
    self.T = T
    self.num_buffers = num_buffers

  def init_params(self):
    return jnp.zeros(self.num_buffers * 2)

  def build_noising_matrix(
      self, params
  ):
    return self.build_normalized_noising_matrix(params)

  def build_normalized_noising_matrix(
      self, params
  ):
    # params is [buf_decay_0, ..., buf_decay_{K-1},
    #            output_scale_0, ..., output_scale_{K-1}]
    buf_decay = params[: self.num_buffers]
    output_scale = params[self.num_buffers :]

    # Compute C coefficients for normalization
    powers = jnp.arange(self.T - 1)
    tmp = buf_decay**powers[:, None] * output_scale
    c_coefs = jnp.concatenate([
        jnp.array([1.0], dtype=buf_decay.dtype),
        jnp.sum(tmp, axis=1),
    ])

    col_norms = jnp.sqrt(jnp.cumsum(c_coefs**2))[::-1]

    cinv_streaming = blt_cinv_streaming_matrix(buf_decay, output_scale)
    return streaming_matrix.scale_rows_and_columns(
        cinv_streaming, row_scale=col_norms
    )

  def project_params(self, params):
    return jnp.clip(params, 0.001, 0.999)

