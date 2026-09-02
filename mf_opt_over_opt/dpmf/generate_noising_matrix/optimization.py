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

# Copyright 2026 DeepMind Technologies Limited.
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

"""Simple wrapper around optax to be used for strategy optimization."""

from collections.abc import Callable
import dataclasses
from typing import Any, TypeAlias, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
import optax

ParamT = TypeVar('ParamT', bound=chex.ArrayTree)

DEFAULT_OPTIMIZER = optax.lbfgs(
    memory_size=1, linesearch=optax.scale_by_backtracking_linesearch(128)
)


def sgd_with_line_search(
    learning_rate = 1.0,
    max_backtracking_steps = 128,
):
  """Returns an SGD optimizer with backtracking line search."""
  return optax.chain(
      optax.sgd(learning_rate=learning_rate),
      optax.scale_by_backtracking_linesearch(
          max_backtracking_steps=max_backtracking_steps
      ),
  )


def adamw_with_line_search(
    learning_rate = 1.0,
    max_backtracking_steps = 128,
):
  """Returns an AdamW optimizer with backtracking line search."""
  return optax.chain(
      optax.adamw(learning_rate=learning_rate),
      optax.scale_by_backtracking_linesearch(
          max_backtracking_steps=max_backtracking_steps
      ),
  )


@dataclasses.dataclass
class CallbackArgs:
  """Information passed to the callback function on each optimization step.

  Properties:
    step: The current optimization step.
    loss: The loss value at the current step.
    grad: The gradient at the current step.
    params: The current parameters.
    state: The current optimizer state.
  """

  step: int
  loss: jnp.ndarray
  grad: chex.ArrayTree | None
  params: chex.ArrayTree
  state: Any


CallbackFnType: TypeAlias = Callable[[CallbackArgs], None | bool]


def jax_enable_x64(fn):
  """Decorator to enable x64 precision for a function."""

  def wrapped_fn(*args, **kwargs):
    with jax.enable_x64():
      return fn(*args, **kwargs)

  return wrapped_fn


@jax_enable_x64
def optimize(
    loss_and_grad,
    params,
    *,
    max_optimizer_steps = 250,
    value_fn = None,
    callback = lambda _: None,
    optimizer = DEFAULT_OPTIMIZER,
    projection_fn = None,
):
  """Optimize a differentiable loss function.

  This is a simple wrapper around optax.  It automatically enables x64 precision
  and JIT-compiles the objective function, gradient, and update rule.

  Args:
    loss_and_grad: A function returning the loss and gradient.
    params: Initial parameters.  These will be cast to float64 internally.
    max_optimizer_steps: The (maximum) number of optimization steps.
    value_fn: Optional value-only evaluation function to be used inside the
      backtracking linesearch.
    callback: Optional callback function to call after each optimization step.
      The callback will be called after each iteration with a `CallbackArgs`
      dataclass.  Early stopping can be achieved by having the callback return a
      truthy value.
    optimizer: An optax.GradientTransformation to use as the underlying
      optimizer.
    projection_fn: Optional function to project parameters to a safe/constrained
      region at each step.

  Returns:
    The parameters that approximately locally minimize the given loss_fun,
    casted back to the same types as the original `params`.
  """
  if projection_fn is not None and value_fn is not None:
    user_value_fn = value_fn
    value_fn = lambda p: user_value_fn(projection_fn(p))

  @jax.jit
  def single_step(params, opt_state):
    value, grad = loss_and_grad(params)
    updates, opt_state = optimizer.update(
        grad, opt_state, params, value=value, grad=grad, value_fn=value_fn
    )
    new_params = cast(ParamT, optax.apply_updates(params, updates))
    if projection_fn is not None:
      new_params = projection_fn(new_params)
    return value, grad, new_params, opt_state

  # MF-style strategy optimization problems are numerically sensitive to
  # precision, so we use f64 internally.
  original_dtypes = jax.tree.map(lambda x: x.dtype, params)

  params = jax.tree.map(jnp.float64, params)
  if projection_fn is not None:
    params = projection_fn(params)
  state = optimizer.init(params)
  for i in range(max_optimizer_steps):
    loss, grad, params, state = single_step(params, state)
    if callback(
        CallbackArgs(step=i, loss=loss, grad=grad, params=params, state=state)
    ):
      break

  return jax.tree.map(jnp.astype, params, original_dtypes)

