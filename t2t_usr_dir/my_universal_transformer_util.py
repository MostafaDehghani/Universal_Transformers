"""Utilities for Universal Transformer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import copy

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils
from tensor2tensor.models.research import universal_transformer_util

import tensorflow as tf


def universal_transformer_encoder(encoder_input,
                                  encoder_self_attention_bias,
                                  hparams,
                                  name="encoder",
                                  nonpadding=None,
                                  save_weights_to=None,
                                  make_image_summary=True):
  """Universal Transformer encoder function.

  Prepares all the arguments and the inputs and passes it to a
  universal_transformer_layer to encode the encoder_input.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convoltutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    y: a Tensors as the output of the encoder
    extra_output: which can be used to pass extra information to the body
  """

  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      padding = common_attention.attention_bias_to_padding(
          encoder_self_attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
      pad_remover = expert_utils.PadRemover(padding)

    ffn_unit = functools.partial(
        universal_transformer_util.transformer_encoder_ffn_unit,
        hparams=hparams,
        nonpadding_mask=nonpadding,
        pad_remover=pad_remover)

    attention_unit = functools.partial(
        universal_transformer_util.transformer_encoder_attention_unit,
        hparams=hparams,
        encoder_self_attention_bias=encoder_self_attention_bias,
        attention_dropout_broadcast_dims=attention_dropout_broadcast_dims,
        save_weights_to=save_weights_to,
        make_image_summary=make_image_summary)

    x, extra_output = universal_transformer_layer(
        x, hparams, ffn_unit, attention_unit, pad_remover=pad_remover)

    if hparams.get("use_memory_as_last_state", False):
      x = extra_output  # which is memory
    return common_layers.layer_preprocess(x, hparams), extra_output


def universal_transformer_decoder(decoder_input,
                                  encoder_output,
                                  decoder_self_attention_bias,
                                  encoder_decoder_attention_bias,
                                  hparams,
                                  name="decoder",
                                  nonpadding=None,
                                  save_weights_to=None,
                                  make_image_summary=True):
  """Universal Transformer decoder function.

  Prepares all the arguments and the inputs and passes it to a
  core_universal_transformer_layer to decoder.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used
      to mask out padding in convoltutional layers.  We generally only
      need this mask for "packed" datasets, because for ordinary datasets,
      no padding is ever followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    y: the output Tensors
    extra_output: which can be used to pass extra information to the body
  """
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    ffn_unit = functools.partial(
        universal_transformer_util.transformer_decoder_ffn_unit,
        hparams=hparams,
        nonpadding_mask=nonpadding)

    attention_unit = functools.partial(
        universal_transformer_util.transformer_decoder_attention_unit,
        hparams=hparams,
        encoder_output=encoder_output,
        decoder_self_attention_bias=decoder_self_attention_bias,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        attention_dropout_broadcast_dims=attention_dropout_broadcast_dims,
        save_weights_to=save_weights_to,
        make_image_summary=make_image_summary)

    x, extra_output = universal_transformer_layer(
        x, hparams, ffn_unit, attention_unit)

    return common_layers.layer_preprocess(x, hparams), extra_output


def universal_transformer_layer(x,
                                hparams,
                                ffn_unit,
                                attention_unit,
                                pad_remover=None):
  """Core function applying the universal transforemr layer.

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    the output tensor,  extra output (can be memory, ponder time, etc.)

  Raises:
    ValueError: Unknown recurrence type
  """

  def add_vanilla_transformer_layer(x, num_layers):
    """Passes the input through num_layers of vanilla transformer layers.

    Args:
     x: input
     num_layers: number of layers

    Returns:
       output of vanilla_transformer_layer
    """

    if hparams.add_position_timing_signal:
      # In case of add_position_timing_signal=true, we set  hparams.pos=None
      # and add position timing signal at the beginning of each step, so for
      # the vanilla transformer, we need to add timing signal here.
      x = common_attention.add_timing_signal_1d(x)
    for layer in range(num_layers):
      with tf.variable_scope("layer_%d" % layer):
        x = ffn_unit(attention_unit(x))
    return x

  with tf.variable_scope("universal_transformer_%s" % hparams.recurrence_type):

    if hparams.mix_with_transformer == "before_ut":
      x = add_vanilla_transformer_layer(x, hparams.num_mixedin_layers)

    ut_function, initializer = get_ut_layer(x, hparams, ffn_unit,
                                            attention_unit, pad_remover)

    output, _, extra_output = tf.foldl(
        ut_function, tf.range(hparams.num_rec_steps), initializer=initializer)

    # This can be the case if we use universal_transformer_lstm layer.
    if hparams.get("use_memory_as_final_state", False):
      output = extra_output

    if hparams.mix_with_transformer == "after_ut":
      output = add_vanilla_transformer_layer(output, hparams.num_mixedin_layers)

    return output, extra_output


def get_ut_layer(x,
                 hparams,
                 ffn_unit,
                 attention_unit,
                 pad_remover=None):
  """Provides the function that is used in universal transforemr steps.

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    ut_function and the ut_initializer

  Raises:
    ValueError: Unknown recurrence type
  """

  if hparams.recurrence_type == "gru":
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_with_gru_as_transition_function,
        hparams=hparams,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "lstm":
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_with_lstm_as_transition_function,
        hparams=hparams,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "basic_plus_gru":
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_basic_plus_gru,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)


  elif hparams.recurrence_type == "basic_plus_lstm":
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_basic_plus_lstm,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "all_steps_so_far":
    #at each step, model gets a combination of the representations learned
    # from all the previouse steps

    # prepare initializer:
    # memory contains the original input + states from all steps
    memory_size = hparams.num_rec_steps + 1
    memory_empty = tf.zeros([memory_size] + common_layers.shape_list(x))
    # filling the first slot with the original input
    memory = universal_transformer_util.fill_memory_slot(memory_empty, x, 0)

    ut_initializer = (x, x, memory)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_all_steps_so_far,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit)

  else:
    raise ValueError("Unknown recurrence type: %s" % hparams.recurrence_type)

  return ut_function, ut_initializer


def universal_transformer_with_gru_as_transition_function(layer_inputs,
                               step,
                               hparams,
                               attention_unit,
                               pad_remover=None):
  """The UT layer which uses a gru as transition function.

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
      - memory: memory used in lstm.
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    layer_output:
        new_state: new state
        inputs: not uesed
        memory: not used
  """


  state, unused_inputs, unused_memory = tf.unstack(layer_inputs,
                                                   num=None, axis=0,
                                                   name="unstack")
  # NOTE:
  # state (ut_state): output of the gru in the previous step
  # inputs (ut_inputs): original input --> we don't use it here
  # memory: we don't use it here

  # Multi_head_attention:
  assert hparams.add_step_timing_signal == False  # Let gru count for us!
  mh_attention_input = universal_transformer_util.step_preprocess(state, step, hparams)
  transition_function_input = attention_unit(mh_attention_input)


  # Transition Function:
  transition_function_input = common_layers.layer_preprocess(
    transition_function_input, hparams)
  with tf.variable_scope("gru"):
    # gru update gate: z_t = sigmoid(W_z.x_t + U_z.h_{t-1})
    transition_function_update_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="update",
        bias_initializer=tf.constant_initializer(1.0),
        activation=tf.sigmoid,
        pad_remover=pad_remover)

    tf.contrib.summary.scalar("gru_update_gate",
                              tf.reduce_mean(transition_function_update_gate))

    # gru reset gate: r_t = sigmoid(W_r.x_t + U_r.h_{t-1})
    transition_function_reset_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="reset",
        bias_initializer=tf.constant_initializer(1.0),
        activation=tf.sigmoid,
        pad_remover=pad_remover)

    tf.contrib.summary.scalar("gru_reset_gate",
                              tf.reduce_mean(transition_function_reset_gate))
    reset_state = transition_function_reset_gate * state

    # gru_candidate_activation: h' = tanh(W_{x_t} + U (r_t h_{t-1})
    transition_function_candidate = _ffn_layer_multi_inputs(
        [transition_function_input, reset_state],
        hparams,
        name="candidate",
        bias_initializer=tf.zeros_initializer(),
        activation=tf.tanh,
        pad_remover=pad_remover)

    transition_function_output = ((1 - transition_function_update_gate)
                                  * transition_function_input +
                                  transition_function_update_gate
                                  * transition_function_candidate)

  transition_function_output = common_layers.layer_preprocess(
      transition_function_output, hparams)

  return transition_function_output, unused_inputs, unused_memory



def universal_transformer_with_lstm_as_transition_function(layer_inputs,
                               step,
                               hparams,
                               attention_unit,
                               pad_remover=None):
  """The UT layer which uses a lstm as transition function.

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
      - memory: memory used in lstm.
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    layer_output:
        new_state: new state
        inputs: the original embedded inputs (= inputs to the first step)
        memory: contains information of state from all the previous steps.
  """


  state, unused_inputs, memory = tf.unstack(layer_inputs, num=None, axis=0,
                                     name="unstack")
  # NOTE:
  # state (ut_state): output of the lstm in the previous step
  # inputs (ut_input): original input --> we don't use it here
  # memory: lstm memory



  # Multi_head_attention:
  assert hparams.add_step_timing_signal == False  # Let lstm count for us!
  mh_attention_input = universal_transformer_util.step_preprocess(state, step, hparams)
  transition_function_input = attention_unit(mh_attention_input)


  # Transition Function:
  transition_function_input = common_layers.layer_preprocess(
    transition_function_input, hparams)
  with tf.variable_scope("lstm"):
    # lstm input gate: i_t = sigmoid(W_i.x_t + U_i.h_{t-1})
    transition_function_input_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="input",
        bias_initializer=tf.zeros_initializer(),
        activation=tf.sigmoid,
        pad_remover=pad_remover)

    tf.contrib.summary.scalar("lstm_input_gate",
                              tf.reduce_mean(transition_function_input_gate))

    # lstm forget gate: f_t = sigmoid(W_f.x_t + U_f.h_{t-1})
    transition_function_forget_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="forget",
        bias_initializer=tf.zeros_initializer(),
        activation=None,
        pad_remover=pad_remover)
    forget_bias_tensor = tf.constant(hparams.lstm_forget_bias)
    transition_function_forget_gate = tf.sigmoid(
      transition_function_forget_gate + forget_bias_tensor)

    tf.contrib.summary.scalar("lstm_forget_gate",
                              tf.reduce_mean(transition_function_forget_gate))

    # lstm ouptut gate: o_t = sigmoid(W_o.x_t + U_o.h_{t-1})
    transition_function_output_gate = _ffn_layer_multi_inputs(
      [transition_function_input, state],
      hparams,
      name="output",
      bias_initializer=tf.zeros_initializer(),
      activation=tf.sigmoid,
      pad_remover=pad_remover)

    tf.contrib.summary.scalar("lstm_output_gate",
                              tf.reduce_mean(transition_function_output_gate))

    # lstm input modulation
    transition_function_input_modulation = _ffn_layer_multi_inputs(
      [transition_function_input, state],
      hparams,
      name="input_modulation",
      bias_initializer=tf.zeros_initializer(),
      activation=tf.tanh,
      pad_remover=pad_remover)

    transition_function_memory = (memory * transition_function_forget_gate +
              transition_function_input_gate * transition_function_input_modulation)

    transition_function_output = (
            tf.tanh(transition_function_memory) * transition_function_output_gate)



  transition_function_output = common_layers.layer_preprocess(
      transition_function_output, hparams)

  return transition_function_output, unused_inputs, transition_function_memory



def universal_transformer_basic_plus_gru(layer_inputs,
                               step,
                               hparams,
                               ffn_unit,
                               attention_unit,
                               pad_remover=None):
  """The UT layer which uses a gru as transition function.

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
      - memory: memory used in lstm.
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    layer_output:
        new_state: new state
        inputs: not uesed
        memory: not used
  """


  state, unused_inputs, unused_memory = tf.unstack(layer_inputs,
                                                   num=None, axis=0,
                                                   name="unstack")
  # NOTE:
  # state (ut_state): output of the gru in the previous step
  # inputs (ut_inputs): original input --> we don't use it here
  # memory: we don't use it here

  # Multi_head_attention:
  assert hparams.add_step_timing_signal == False  # Let gru count for us!
  mh_attention_input = universal_transformer_util.step_preprocess(state, step, hparams)
  transition_function_input = ffn_unit(attention_unit(mh_attention_input))


  # Transition Function:
  transition_function_input = common_layers.layer_preprocess(
    transition_function_input, hparams)
  with tf.variable_scope("gru"):
    # gru update gate: z_t = sigmoid(W_z.x_t + U_z.h_{t-1})
    transition_function_update_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="update",
        bias_initializer=tf.constant_initializer(1.0),
        activation=tf.sigmoid,
        pad_remover=pad_remover)

    tf.contrib.summary.scalar("gru_update_gate",
                              tf.reduce_mean(transition_function_update_gate))

    # gru reset gate: r_t = sigmoid(W_r.x_t + U_r.h_{t-1})
    transition_function_reset_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="reset",
        bias_initializer=tf.constant_initializer(1.0),
        activation=tf.sigmoid,
        pad_remover=pad_remover)

    tf.contrib.summary.scalar("gru_reset_gate",
                              tf.reduce_mean(transition_function_reset_gate))
    reset_state = transition_function_reset_gate * state

    # gru_candidate_activation: h' = tanh(W_{x_t} + U (r_t h_{t-1})
    transition_function_candidate = _ffn_layer_multi_inputs(
        [transition_function_input, reset_state],
        hparams,
        name="candidate",
        bias_initializer=tf.zeros_initializer(),
        activation=tf.tanh,
        pad_remover=pad_remover)

    transition_function_output = ((1 - transition_function_update_gate)
                                  * transition_function_input +
                                  transition_function_update_gate
                                  * transition_function_candidate)

  transition_function_output = common_layers.layer_preprocess(
      transition_function_output, hparams)

  return transition_function_output, unused_inputs, unused_memory



def universal_transformer_basic_plus_lstm(layer_inputs,
                               step,
                               hparams,
                               ffn_unit,
                               attention_unit,
                               pad_remover=None):
  """The UT layer which uses a lstm as transition function.

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
      - memory: memory used in lstm.
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    layer_output:
        new_state: new state
        inputs: the original embedded inputs (= inputs to the first step)
        memory: contains information of state from all the previous steps.
  """


  state, unused_inputs, memory = tf.unstack(layer_inputs, num=None, axis=0,
                                     name="unstack")
  # NOTE:
  # state (ut_state): output of the lstm in the previous step
  # inputs (ut_input): original input --> we don't use it here
  # memory: lstm memory



  # Multi_head_attention:
  assert hparams.add_step_timing_signal == False  # Let lstm count for us!
  mh_attention_input = universal_transformer_util.step_preprocess(state, step, hparams)
  transition_function_input = ffn_unit(attention_unit(mh_attention_input))


  # Transition Function:
  transition_function_input = common_layers.layer_preprocess(
    transition_function_input, hparams)
  with tf.variable_scope("lstm"):
    # lstm input gate: i_t = sigmoid(W_i.x_t + U_i.h_{t-1})
    transition_function_input_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="input",
        bias_initializer=tf.zeros_initializer(),
        activation=tf.sigmoid,
        pad_remover=pad_remover)

    tf.contrib.summary.scalar("lstm_input_gate",
                              tf.reduce_mean(transition_function_input_gate))

    # lstm forget gate: f_t = sigmoid(W_f.x_t + U_f.h_{t-1})
    transition_function_forget_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="forget",
        bias_initializer=tf.zeros_initializer(),
        activation=None,
        pad_remover=pad_remover)
    forget_bias_tensor = tf.constant(hparams.lstm_forget_bias)
    transition_function_forget_gate = tf.sigmoid(
      transition_function_forget_gate + forget_bias_tensor)

    tf.contrib.summary.scalar("lstm_forget_gate",
                              tf.reduce_mean(transition_function_forget_gate))

    # lstm ouptut gate: o_t = sigmoid(W_o.x_t + U_o.h_{t-1})
    transition_function_output_gate = _ffn_layer_multi_inputs(
      [transition_function_input, state],
      hparams,
      name="output",
      bias_initializer=tf.zeros_initializer(),
      activation=tf.sigmoid,
      pad_remover=pad_remover)

    tf.contrib.summary.scalar("lstm_output_gate",
                              tf.reduce_mean(transition_function_output_gate))

    # lstm input modulation
    transition_function_input_modulation = _ffn_layer_multi_inputs(
      [transition_function_input, state],
      hparams,
      name="input_modulation",
      bias_initializer=tf.zeros_initializer(),
      activation=tf.tanh,
      pad_remover=pad_remover)

    transition_function_memory = (memory * transition_function_forget_gate +
              transition_function_input_gate * transition_function_input_modulation)

    transition_function_output = (
            tf.tanh(transition_function_memory) * transition_function_output_gate)



  transition_function_output = common_layers.layer_preprocess(
      transition_function_output, hparams)

  return transition_function_output, unused_inputs, transition_function_memory


def universal_transformer_all_steps_so_far(layer_inputs,
                                            step, hparams,
                                            ffn_unit,
                                            attention_unit):
  """universal_transformer.
  It uses an attention mechanism-flipped vertically-
  over all the states from previous steps to generate the new_state.
  Args:
    layer_inputs:
      - state: state
      - memory: contains states from all the previous steps.
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
  Returns:
    layer_output:
        new_state: new state
        memory: contains states from all the previous steps.
  """
  _, inputs, memory = layer_inputs
  all_states = memory
  # get the states up to the current step (non-zero part of the memory)
  states_so_far = all_states[:step, :, :, :]

  states_so_far_weights = tf.nn.softmax(
      common_layers.dense(
          states_so_far, (hparams.hidden_size if hparams.dwa_elements else 1),
          activation=None,
          use_bias=True),
          axis=-1)

  # # get summary of the step weights
  # step_weightes = tf.unstack(states_so_far_weights, axis=0, name="step_weightes")
  # for step_i, step_w  in enumerate(step_weightes):
  #   tf.contrib.summary.scalar("step_%d_weight:"%step_i,
  #                             tf.reduce_mean(step_w))


  # prepare the state as the summary of
  state_to_be_transformed = tf.reduce_sum(
      (states_so_far * states_so_far_weights), axis=0)

  state_to_be_transformed = universal_transformer_util.step_preprocess(
                            state_to_be_transformed, step, hparams)

  new_state = ffn_unit(attention_unit(state_to_be_transformed))


  # add the new state to the memory
  memory = universal_transformer_util.fill_memory_slot(memory,
                                                       new_state, step + 1)

  return new_state, inputs, memory


def _ffn_layer_multi_inputs(inputs_list,
                            hparams,
                            ffn_layer_type="dense",
                            name="ffn",
                            kernel_initializer=None,
                            bias_initializer=None,
                            activation=None,
                            pad_remover=None,
                            preprocess=False,
                            postprocess=False):
  """Implements a Feed-forward layer with multiple inputs, pad-removing, etc.
  Args:
    inputs_list: list of input tensors
    hparams: hyper-parameters
    ffn_layer_type: dense / dense_dropconnect/ dense_relu_dense
    name: name
    kernel_initializer: kernel initializer
    bias_initializer: bias initializer
    activation: activation function
    pad_remover: pad remover
    preprocess: if preprocess the input
    postprocess: if postprocess the output
  Returns:
    a tensor
  Raises:
    ValueError: Unknown ffn_layer type.
  """

  # need at least one inputs
  num_inputs = len(inputs_list)
  assert num_inputs > 0

  if preprocess and num_inputs == 1:
    inputs_list[0] = common_layers.layer_preprocess(inputs_list[0], hparams)

  if postprocess:
    original_inputs = inputs_list[0]

  # the output size is the hidden size of the main inputs
  main_input = inputs_list[0]
  original_shape = common_layers.shape_list(main_input)
  assert hparams.hidden_size == common_layers.shape_list(main_input)[-1]

  # all the inputs are in the same shape with main inputs
  for inputs in inputs_list:
    main_input.get_shape().assert_is_compatible_with(inputs.get_shape())

  def remove_pads(x):
    original_shape = common_layers.shape_list(x)
    # Collapse `x` across examples, and remove padding positions.
    x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
    x = tf.expand_dims(pad_remover.remove(x), axis=0)
    return x

  if pad_remover:
    for i, inputs in enumerate(inputs_list):
      inputs_list[i] = remove_pads(inputs)

  ffn_inputs = (
      inputs_list[0]
      if len(inputs_list) == 1 else tf.concat(inputs_list, axis=-1))

  if ffn_layer_type == "dense":
    output = common_layers.dense(
        ffn_inputs,
        hparams.hidden_size,
        name=name,
        activation=activation,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)

  elif ffn_layer_type == "dense_dropconnect":
    output = common_layers.dense_dropconnect(
        ffn_inputs,
        hparams.hidden_size,
        name=name,
        dropconnect_dropout=hparams.dropconnect_dropout,
        output_activation=activation)
    postprocess = False  # no dropout on the output unit

  elif ffn_layer_type == "dense_relu_dense":
    output = common_layers.dense_relu_dense(
        ffn_inputs,
        hparams.filter_size,
        hparams.hidden_size,
        name=name,
        dropout=hparams.relu_dropout,
        output_activation=activation,
    )

  else:
    raise ValueError("Unknown ffn_layer type: %s" % ffn_layer_type)

  if pad_remover:
    # Restore `output` to the original shape of `x`, including padding.
    output = tf.reshape(
        pad_remover.restore(tf.squeeze(output, axis=0)), original_shape)

  if postprocess:
    if num_inputs == 1:
      output = common_layers.layer_postprocess(original_inputs, output, hparams)
    else:  # only dropout (no residual)x
      hp = copy.copy(hparams)
      hp.layer_postprocess_sequence = hp.layer_postprocess_sequence.replace(
          "a", "")
      output = common_layers.layer_postprocess(original_inputs, output, hp)

  return output

