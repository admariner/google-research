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

"""Extension of JaxBERT and utilities for training."""

from typing import Any

import haiku as hk
import jax
from jax import numpy as jnp

TF_LAYERNORM_EPSILON = 1e-12
BERT_VOCAB_SIZE = 30522
FIRST_WORD = 999
MASK = 103

# Model definition.


class BERT(hk.Module):
  """Same as /learning/deepmind/research/language/jaxbert/model.py, but also outputs `log_probs`."""

  def __init__(
      self,
      vocab_size,
      hidden_size,
      num_hidden_layers,
      num_attention_heads,
      intermediate_size,
      hidden_dropout_prob,
      attention_probs_dropout_prob,
      max_position_embeddings,
      type_vocab_size,
      initializer_range,
      name = 'BERT',
  ):
    super().__init__(name=name)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.size_per_head = hidden_size // num_attention_heads

  def _bert_layer(
      self,
      layer_input,
      layer_index,
      input_mask,
      is_training,
  ):
    """Forward pass of a single layer."""

    *batch_dims, seq_length, hidden_size = layer_input.shape

    queries = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='query_%d' % layer_index,
    )(layer_input)
    keys = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='keys_%d' % layer_index,
    )(layer_input)
    values = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='values_%d' % layer_index,
    )(layer_input)

    btnh = (
        *batch_dims,
        seq_length,
        self.num_attention_heads,
        self.size_per_head,
    )
    queries = jnp.reshape(queries, btnh)
    keys = jnp.reshape(keys, btnh)
    values = jnp.reshape(values, btnh)

    # Attention scores.
    attention_scores = jnp.einsum('...tnh,...fnh->...nft', keys, queries)
    attention_scores *= self.size_per_head ** (-0.5)

    # attention_scores shape: [..., num_heads, num_attending, num_attended_over]
    # Broadcast the input mask along heads and query dimension.
    # If a key/value location is pad, do not attend over it.
    # Do that by plunging the attention logit to negative infinity.
    bcast_shape = list(input_mask.shape[:-1]) + [1, 1, input_mask.shape[-1]]
    input_mask_broadcasted = jnp.reshape(input_mask, bcast_shape)
    attention_mask = -1.0 * 1e30 * (1.0 - input_mask_broadcasted)
    attention_scores += attention_mask

    attention_probs = jax.nn.softmax(attention_scores)
    if is_training:
      attention_probs = hk.dropout(
          hk.next_rng_key(), self.attention_probs_dropout_prob, attention_probs
      )

    # Weighted sum.
    attention_output = jnp.einsum(
        '...nft,...tnh->...fnh', attention_probs, values
    )
    attention_output = jnp.reshape(
        attention_output, (*batch_dims, seq_length, hidden_size)
    )

    # Projection to hidden size.
    attention_output = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='attention_output_dense_%d' % layer_index,
    )(attention_output)
    if is_training:
      attention_output = hk.dropout(
          hk.next_rng_key(), self.hidden_dropout_prob, attention_output
      )
    attention_output = hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        eps=TF_LAYERNORM_EPSILON,
        name='attention_output_ln_%d' % layer_index,
    )(attention_output + layer_input)

    # FFW.
    intermediate_output = hk.Linear(
        self.intermediate_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='intermediate_output_%d' % layer_index,
    )(attention_output)
    intermediate_output = jax.nn.gelu(intermediate_output)

    layer_output = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='layer_output_%d' % layer_index,
    )(intermediate_output)
    if is_training:
      layer_output = hk.dropout(
          hk.next_rng_key(), self.hidden_dropout_prob, layer_output
      )
    layer_output = hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        eps=TF_LAYERNORM_EPSILON,
        name='layer_output_ln_%d' % layer_index,
    )(layer_output + attention_output)

    return layer_output

  def __call__(
      self,
      input_ids,
      token_type_ids,
      input_mask,
      is_training = True,
  ):
    """Forward pass of the BERT model."""

    # Prepare size, fill out missing inputs.
    *_, seq_length = input_ids.shape

    if input_mask is None:
      input_mask = jnp.ones(shape=input_ids.shape, dtype=jnp.int32)

    if token_type_ids is None:
      token_type_ids = jnp.zeros(shape=input_ids.shape, dtype=jnp.int32)

    position_ids = jnp.arange(seq_length)[None, :]

    # Embeddings.
    word_embedder = hk.Embed(
        vocab_size=self.vocab_size,
        embed_dim=self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='word_embeddings',
    )
    word_embeddings = word_embedder(input_ids)
    token_type_embeddings = hk.Embed(
        vocab_size=self.type_vocab_size,
        embed_dim=self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='token_type_embeddings',
    )(token_type_ids)
    position_embeddings = hk.Embed(
        vocab_size=self.max_position_embeddings,
        embed_dim=self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='position_embeddings',
    )(position_ids)
    input_embeddings = (
        word_embeddings + token_type_embeddings + position_embeddings
    )
    input_embeddings = hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        eps=TF_LAYERNORM_EPSILON,
        name='embeddings_ln',
    )(input_embeddings)
    if is_training:
      input_embeddings = hk.dropout(
          hk.next_rng_key(), self.hidden_dropout_prob, input_embeddings
      )

    # BERT layers.
    h = input_embeddings
    layer_outputs = []
    for i in range(self.num_hidden_layers):
      h = self._bert_layer(
          h, layer_index=i, input_mask=input_mask, is_training=is_training
      )
      layer_outputs.append(h)
    last_layer = h

    # Masked language modelling logprobs.
    mlm_hidden = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='mlm_dense',
    )(last_layer)
    mlm_hidden = jax.nn.gelu(mlm_hidden)
    mlm_hidden = hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        eps=TF_LAYERNORM_EPSILON,
        name='mlm_ln',
    )(mlm_hidden)
    output_weights = jnp.transpose(word_embedder.embeddings)
    logits = jnp.matmul(mlm_hidden, output_weights)
    logits = hk.Bias(bias_dims=[-1], name='mlm_bias')(logits)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Pooled output: [CLS] token.
    first_token_last_layer = last_layer[Ellipsis, 0, :]
    pooled_output = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='pooler_dense',
    )(first_token_last_layer)
    pooled_output = jnp.tanh(pooled_output)

    return {
        'layers': layer_outputs,
        'pooled_output': pooled_output,
        'log_probs': log_probs,
    }


def masked_lm_preprocess(
    input_ids, key
):
  """Preprocess input ids for masked language modelling.

  This follows the conventions of https://arxiv.org/abs/1810.04805 for the
  Masked LM task: 15% of tokens are selected for prediction. Of those 80% are
  replaced with the special mask token, 10% with a random word, and 10%
  untouched.

  This method assumes the default BERT tokenization, using hard-coded constants
  building on that tokenization.

  Args:
    input_ids: The input token ids.
    key: The PRNG key.

  Returns:
    The masked input ids, and a boolean array indicating which positions were
    selected for prediction.
  """
  key1, key2 = jax.random.split(key)
  mask_coins = jax.random.uniform(key1, shape=input_ids.shape)
  maskable = input_ids >= FIRST_WORD
  masked = jnp.logical_and(mask_coins < 0.15, maskable)
  replace_with_random = jnp.logical_and(mask_coins < 0.15 * 0.1, maskable)
  replace_with_mask = jnp.logical_and(
      mask_coins < 0.15 * 0.9, jnp.logical_not(replace_with_random)
  )
  random_words = jax.random.randint(
      key2, shape=input_ids.shape, minval=FIRST_WORD, maxval=BERT_VOCAB_SIZE - 1
  )
  masked_input_ids = jnp.where(
      replace_with_mask,
      MASK,
      jnp.where(replace_with_random, random_words, input_ids),
  )
  return masked_input_ids, masked


def cross_entropy_loss(
    log_softmax_logits,
    labels,
    weights,
    eps = 1e-9,
):
  """Returns the cross-entropy classification loss.

  Args:
    log_softmax_logits: The log of the softmax of the logits for the mini-batch,
      e.g. as output by jax.nn.log_softmax(logits).
    labels: The labels for the mini-batch.
    weights: The per-token weights for the loss function.
    eps: A small value to prevent division by zero.
  """
  num_classes = log_softmax_logits.shape[-1]
  one_hot_labels = jax.nn.one_hot(labels, num_classes)
  per_example_losses = -jnp.sum(one_hot_labels * log_softmax_logits, axis=-1)
  return jnp.sum(jnp.multiply(weights, per_example_losses)) / (
      jnp.sum(weights) + eps
  )


# Loss function


def tiny_bert_forward_fn(
    input_ids,
    token_type_ids,
    is_training = True,
):
  """Forward function for BERT."""
  bert = BERT(
      vocab_size=30522,
      hidden_size=128,
      num_hidden_layers=2,
      num_attention_heads=2,
      intermediate_size=512,
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      max_position_embeddings=512,
      type_vocab_size=2,
      initializer_range=0.02,
  )
  result = bert(
      input_ids, token_type_ids, input_mask=None, is_training=is_training
  )
  return result


tiny_bert_forward = hk.transform(tiny_bert_forward_fn)


def tiny_bert_loss_fn(
    params,
    key,
    input_ids,
    is_training = True,
):
  """Loss function for Tiny BERT."""
  token_type_ids = jnp.zeros_like(input_ids)
  key1, key2 = jax.random.split(key)
  masked_ids, masked = masked_lm_preprocess(input_ids, key1)
  result = tiny_bert_forward.apply(
      params,
      key2,
      input_ids=masked_ids,
      token_type_ids=token_type_ids,
      is_training=is_training,
  )
  log_probs = result['log_probs']
  return cross_entropy_loss(
      log_probs, input_ids, masked.astype(jnp.int32), eps=1e-9
  )
