"""Universal Transformers.

Universal Transformer is described in https://arxiv.org/abs/1807.03819.

Universal Transformer is recurrent in depth while employing self-attention
to combine information from different parts of sequences.
In contrast to the Transformer, given enough memory its recurrence in depth
makes the Universal Transformer computationally universal.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.utils import registry
from . import my_universal_transformer_util

import tensorflow as tf


@registry.register_model
class MyUniversalTransformer(universal_transformer.UniversalTransformer):
  """Universal Transformer: Depth-wise recurrent transformer model."""

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode Universal Transformer inputs.

    It is similar to "transformer.encode", but it uses
    "universal_transformer_util.universal_transformer_encoder" instead of
    "transformer.transformer_encoder".

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: Unused.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
          encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """
    del losses

    inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer.transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    (encoder_output, encoder_extra_output) = (
            my_universal_transformer_util.universal_transformer_encoder(
            encoder_input,
            self_attention_bias,
            hparams,
            nonpadding=transformer.features_to_nonpadding(features, "inputs"),
            save_weights_to=self.attention_weights))

    return encoder_output, encoder_decoder_attention_bias, encoder_extra_output

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             nonpadding=None,
             losses=None):
    """Decode Universal Transformer outputs from encoder representation.

    It is similar to "transformer.decode", but it uses
    "universal_transformer_util.universal_transformer_decoder" instead of
    "transformer.transformer_decoder".

    Args:
      decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
        hidden_dim]
      encoder_output: Encoder representation. [batch_size, input_length,
        hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
        attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
        self-attention. [batch_size, decoder_length]
      hparams: hyperparmeters for model.
      cache: Unimplemented.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]
      losses: Unused.

    Returns:
       Tuple of:
         Final decoder representation. [batch_size, decoder_length,
            hidden_dim]
         encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)

    """
    del losses
    # TODO(dehghani): enable caching.
    del cache

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    # No caching in Universal Transformers!
    (decoder_output, dec_extra_output) = (
      my_universal_transformer_util.universal_transformer_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            hparams,
            nonpadding=nonpadding,
            save_weights_to=self.attention_weights))

    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2), dec_extra_output

@registry.register_hparams
def universal_transformer_with_gru_as_transition_function():
  hparams = universal_transformer.universal_transformer_base()
  hparams.recurrence_type = "gru"
  hparams.add_step_timing_signal = False # Let gru count in time for us!
  return hparams

@registry.register_hparams
def universal_transformer_with_lstm_as_transition_function():
  hparams = universal_transformer.universal_transformer_base()
  hparams.recurrence_type = "lstm"
  hparams.add_step_timing_signal = False # Let lstm count in time for us!
  return hparams

@registry.register_hparams
def universal_transformer_basic_plus_gru():
  hparams = universal_transformer.universal_transformer_base()
  hparams.recurrence_type = "basic_plus_gru"
  # hparams.transformer_ffn_type = "fc"
  hparams.batch_size = 2048
  hparams.add_step_timing_signal = False # Let lstm count in time for us!
  return hparams

@registry.register_hparams
def universal_transformer_basic_plus_lstm():
  hparams = universal_transformer.universal_transformer_base()
  hparams.recurrence_type = "basic_plus_lstm"
  # hparams.transformer_ffn_type = "fc"
  hparams.batch_size = 2048
  hparams.add_step_timing_signal = False # Let lstm count in time for us!
  return hparams

@registry.register_hparams
def universal_transformer_all_steps_so_far():
  hparams = universal_transformer.universal_transformer_base()
  hparams.recurrence_type = "all_steps_so_far"
  return hparams

