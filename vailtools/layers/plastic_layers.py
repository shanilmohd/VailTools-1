"""
TODO: Improve documentation.
TODO: Validate that the plasticity is applied in the correct locations.
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.python.ops import array_ops

from ..utils import register_custom_objects


def hebb(lr, prev_output, curr_output, plastic_vals):
    """
    Implements a classic Hebbian plasticity update rule.

    Currently only tested where prev_output/curr_output have shape (batch_size, time_steps, features).

    Args:
        lr: float or keras.Tensor, Learning rate / step size for the plasticity update.
        prev_output: keras.Tensor, Pre-synaptic activations.
        curr_output: keras.Tensor, Post-synaptic activations.
        plastic_vals: keras.Tensor, Current plasticity values.

    Returns: keras.Tensor
        Updated plasticity values.
    """
    return (
        lr * K.dot(K.expand_dims(prev_output, 2), K.expand_dims(curr_output, 1))
        + (1 - lr) * plastic_vals
    )


def oja(lr, prev_output, curr_output, plastic_vals):
    """
    Implements Oja's plasticity update rule.

    Currently only tested where prev_output/curr_output have shape (batch_size, time_steps, features).

    Args:
        lr: float or keras.Tensor, Learning rate / step size for the plasticity update.
        prev_output: keras.Tensor, Pre-synaptic activations.
        curr_output: keras.Tensor, Post-synaptic activations.
        plastic_vals: keras.Tensor, Current plasticity values.

    Returns: keras.Tensor
        Updated plasticity values.
    """
    return plastic_vals + lr * K.expand_dims(curr_output, 1) * (
        K.expand_dims(prev_output, 2) - plastic_vals * K.expand_dims(curr_output, 1)
    )


class PlasticRNNCell(tf.keras.layers.SimpleRNNCell):
    """
    Extends the Keras SimpleRNNCell with a differentiable plasticity mechanism.

    Reference:
        https://arxiv.org/abs/1804.02464
    """

    def __init__(
        self, units, plasticity_rule="oja", **kwargs,
    ):
        super().__init__(units, **kwargs)
        self.state_size = (self.units, self.units ** 2)

        self.plastic_kernel = None
        self.plastic_lr = None
        self.plasticity_rule = plasticity_rule
        self.plastic_update = oja if plasticity_rule == "oja" else hebb

    def build(self, input_shape):
        self.plastic_kernel = self.add_weight(
            shape=(1, self.units, self.units),
            name="plastic_kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.plastic_lr = self.add_weight(
            shape=(1,),
            name="plastic_lr",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        super().build(input_shape)

    def call(self, inputs, states, training=None):
        prev_output = states[0]
        plastic_vals = tf.reshape(states[1], (-1, self.units, self.units))

        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(prev_output, training)
        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output *= rec_dp_mask

        output = h + K.dot(prev_output, self.recurrent_kernel)

        # Apply plasticity
        output = output + K.batch_dot(prev_output, self.plastic_kernel * plastic_vals)

        if self.activation is not None:
            output = self.activation(output)

        plastic_vals = self.plastic_update(
            self.plastic_lr, prev_output, output, plastic_vals
        )

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True
        return output, [output, K.batch_flatten(plastic_vals)]

    def get_config(self):
        config = {"plasticity_rule": self.plasticity_rule}
        base_config = super().get_config()
        return dict(**base_config, **config)


class PlasticRNN(tf.keras.layers.SimpleRNN):
    """
    Augments the Keras SimpleRNN with a differentiable plasticity mechanism.

    Reference:
        https://arxiv.org/abs/1804.02464
    """

    def __init__(
        self,
        units,
        activation="tanh",
        bias_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        dropout=0.0,
        kernel_constraint=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        plasticity_rule="oja",
        recurrent_constraint=None,
        recurrent_dropout=0.0,
        recurrent_initializer="orthogonal",
        recurrent_regularizer=None,
        use_bias=True,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        **kwargs,
    ):
        super().__init__(
            units,
            go_backwards=go_backwards,
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )
        self.plasticity_rule = plasticity_rule

        self.cell = PlasticRNNCell(
            units,
            activation=activation,
            bias_constraint=bias_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            dropout=dropout,
            kernel_constraint=kernel_constraint,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            plasticity_rule=plasticity_rule,
            recurrent_constraint=recurrent_constraint,
            recurrent_dropout=recurrent_dropout,
            recurrent_initializer=recurrent_initializer,
            recurrent_regularizer=recurrent_regularizer,
            use_bias=use_bias,
        )

    def get_config(self):
        config = {"plasticity_rule": self.plasticity_rule}
        base_config = super().get_config()
        return dict(**base_config, **config)


class PlasticGRUCell(tf.keras.layers.GRUCell):
    """
    Augments the Keras GRUCell with a differentiable plasticity mechanism.

    Reference:
        https://arxiv.org/abs/1804.02464
    """

    def __init__(self, units, plasticity_rule="oja", **kwargs):
        super().__init__(units, **kwargs)
        self.state_size = (self.units, self.units ** 2)

        self.plastic_kernel = None
        self.plastic_lr = None
        self.plasticity_rule = plasticity_rule
        self.plastic_update = oja if plasticity_rule == "oja" else hebb
        self.plastic_val = None

    def build(self, input_shape):
        self.plastic_kernel = self.add_weight(
            shape=(1, self.units, self.units),
            name="plastic_kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.plastic_lr = self.add_weight(
            shape=(1,),
            name="plastic_lr",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory
        plastic_vals = tf.reshape(states[1], (-1, self.units, self.units))

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = array_ops.unstack(self.bias)

        if self.implementation == 1:
            if 0.0 < self.dropout < 1.0:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = K.dot(inputs_z, self.kernel[:, : self.units])
            x_r = K.dot(inputs_r, self.kernel[:, self.units : self.units * 2])
            x_h = K.dot(inputs_h, self.kernel[:, self.units * 2 :])

            # Apply plasticity
            x_h = x_h + K.batch_dot(h_tm1, self.plastic_kernel * plastic_vals)

            if self.use_bias:
                x_z = K.bias_add(x_z, input_bias[: self.units])
                x_r = K.bias_add(x_r, input_bias[self.units : self.units * 2])
                x_h = K.bias_add(x_h, input_bias[self.units * 2 :])

            if 0.0 < self.recurrent_dropout < 1.0:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, : self.units])
            recurrent_r = K.dot(
                h_tm1_r, self.recurrent_kernel[:, self.units : self.units * 2]
            )
            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, recurrent_bias[: self.units])
                recurrent_r = K.bias_add(
                    recurrent_r, recurrent_bias[self.units : self.units * 2]
                )

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2 :])
                if self.use_bias:
                    recurrent_h = K.bias_add(
                        recurrent_h, recurrent_bias[self.units * 2 :]
                    )
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(
                    r * h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
                )

            hh = self.activation(x_h + recurrent_h)
        else:
            if 0.0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = K.dot(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = K.bias_add(matrix_x, input_bias)

            x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)

            # Apply plasticity
            x_h = x_h + K.batch_dot(h_tm1, self.plastic_kernel * plastic_vals)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, : 2 * self.units])

            recurrent_z, recurrent_r, recurrent_h = array_ops.split(
                matrix_inner, [self.units, self.units, -1], axis=-1
            )

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(
                    r * h_tm1, self.recurrent_kernel[:, 2 * self.units :]
                )

            hh = self.activation(x_h + recurrent_h)
        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        # Update plasticity
        plastic_vals = self.plastic_update(self.plastic_lr, h_tm1, h, plastic_vals)
        return h, [h, K.batch_flatten(plastic_vals)]

    def get_config(self):
        config = {"plasticity_rule": self.plasticity_rule}
        base_config = super().get_config()
        return dict(**base_config, **config)


class PlasticGRU(tf.keras.layers.GRU):
    """
    Augments the Keras GRU with a differentiable plasticity mechanism.

    Reference:
        https://arxiv.org/abs/1804.02464
    """

    def __init__(
        self,
        units,
        activation="tanh",
        bias_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        dropout=0.0,
        go_backwards=False,
        kernel_constraint=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        plasticity_rule="oja",
        recurrent_activation="hard_sigmoid",
        recurrent_constraint=None,
        recurrent_dropout=0.0,
        recurrent_initializer="orthogonal",
        recurrent_regularizer=None,
        reset_after=False,
        return_sequences=False,
        return_state=False,
        stateful=False,
        unroll=False,
        use_bias=True,
        **kwargs,
    ):
        super().__init__(
            units,
            dropout=dropout,
            go_backwards=go_backwards,
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )

        self.plasticity_rule = plasticity_rule
        self.cell = PlasticGRUCell(
            units,
            activation=activation,
            bias_constraint=bias_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            dropout=dropout,
            kernel_constraint=kernel_constraint,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            plasticity_rule=plasticity_rule,
            recurrent_activation=recurrent_activation,
            recurrent_constraint=recurrent_constraint,
            recurrent_dropout=recurrent_dropout,
            recurrent_initializer=recurrent_initializer,
            recurrent_regularizer=recurrent_regularizer,
            reset_after=reset_after,
            use_bias=use_bias,
        )

    def get_config(self):
        config = {"plasticity_rule": self.plasticity_rule}
        base_config = super().get_config()
        return dict(**base_config, **config)


class PlasticLSTMCell(tf.keras.layers.LSTMCell):
    """
    Augments the Keras LSTMCell with a differentiable plasticity mechanism.

    Reference:
        https://arxiv.org/abs/1804.02464
    """

    def __init__(
        self, units, plasticity_rule="oja", **kwargs,
    ):
        super().__init__(units, **kwargs)
        self.state_size = (self.units, self.units, self.units ** 2)

        self.plastic_kernel = None
        self.plastic_lr = None
        self.plasticity_rule = plasticity_rule
        self.plastic_update = oja if plasticity_rule == "oja" else hebb
        self.plastic_val = None

    def build(self, input_shape):
        self.plastic_kernel = self.add_weight(
            shape=(1, self.units, self.units),
            name="plastic_kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.plastic_lr = self.add_weight(
            shape=(1,),
            name="plastic_lr",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        plastic_vals = K.reshape(states[2], (-1, self.units, self.units))

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)

        if self.implementation == 1:
            if 0 < self.dropout < 1.0:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            k_i, k_f, k_c, k_o = array_ops.split(
                self.kernel, num_or_size_splits=4, axis=1
            )
            x_i = K.dot(inputs_i, k_i)
            x_f = K.dot(inputs_f, k_f)
            x_c = K.dot(inputs_c, k_c)
            x_o = K.dot(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = array_ops.split(
                    self.bias, num_or_size_splits=4, axis=0
                )
                x_i = K.bias_add(x_i, b_i)
                x_f = K.bias_add(x_f, b_f)
                x_c = K.bias_add(x_c, b_c)
                x_o = K.bias_add(x_o, b_o)

            if 0 < self.recurrent_dropout < 1.0:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
            h_tm1 = h_tm1_c

        else:
            if 0.0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0.0 < self.recurrent_dropout < 1.0:
                h_tm1 = h_tm1 * rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z = array_ops.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        # Apply plasticity
        c = c + K.batch_dot(c_tm1, self.plastic_kernel * plastic_vals)

        h = o * self.activation(c)

        # Update plastic values
        plastic_vals = self.plastic_update(self.plastic_lr, h_tm1, h, plastic_vals)
        return h, [h, c, K.batch_flatten(plastic_vals)]

    def get_config(self):
        config = {"plasticity_rule": self.plasticity_rule}
        base_config = super().get_config()
        return dict(**base_config, **config)


class PlasticLSTM(tf.keras.layers.LSTM):
    """
    Augments the Keras LSTM with a differentiable plasticity mechanism.

    Reference:
        https://arxiv.org/abs/1804.02464
    """

    def __init__(
        self,
        units,
        activation="tanh",
        activity_regularizer=None,
        bias_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        dropout=0.0,
        go_backwards=False,
        kernel_constraint=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        plasticity_rule="oja",
        recurrent_activation="hard_sigmoid",
        recurrent_constraint=None,
        recurrent_dropout=0.0,
        recurrent_initializer="orthogonal",
        recurrent_regularizer=None,
        return_sequences=False,
        return_state=False,
        stateful=False,
        unit_forget_bias=True,
        unroll=False,
        use_bias=True,
        **kwargs,
    ):
        super().__init__(
            units,
            go_backwards=go_backwards,
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )

        self.plasticity_rule = plasticity_rule
        self.cell = PlasticLSTMCell(
            units,
            activation=activation,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            dropout=dropout,
            kernel_constraint=kernel_constraint,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            plasticity_rule=plasticity_rule,
            recurrent_activation=recurrent_activation,
            recurrent_constraint=recurrent_constraint,
            recurrent_dropout=recurrent_dropout,
            recurrent_initializer=recurrent_initializer,
            recurrent_regularizer=recurrent_regularizer,
            unit_forget_bias=unit_forget_bias,
            use_bias=use_bias,
        )

    def get_config(self):
        config = {"plasticity_rule": self.plasticity_rule}
        base_config = super().get_config()
        return dict(**base_config, **config)


class NMPlasticRNNCell(tf.keras.layers.SimpleRNNCell):
    """
    Extends the Keras SimpleRNNCell with a differentiable plasticity mechanism.

    Reference:
        https://arxiv.org/abs/1804.02464
    """

    def __init__(
        self, units, plasticity_rule="oja", **kwargs,
    ):
        super().__init__(units, **kwargs)
        self.state_size = (self.units, self.units ** 2, self.units ** 2)

        self.plastic_kernel = None
        self.modulatory_kernel = None
        self.plastic_lr = None
        self.plasticity_rule = plasticity_rule
        self.plastic_update = oja if plasticity_rule == "oja" else hebb

    def build(self, input_shape):
        self.plastic_kernel = self.add_weight(
            shape=(1, self.units, self.units),
            name="plastic_kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.modulatory_kernel = self.add_weight(
            shape=(1, self.units, self.units),
            name="modulatory_kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.plastic_lr = self.add_weight(
            shape=(1,),
            name="plastic_lr",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        super().build(input_shape)

    def call(self, inputs, states, training=None):
        prev_output = states[0]
        plastic_vals = tf.reshape(states[1], (-1, self.units, self.units))
        trace = K.reshape(states[2], (-1, self.units, self.units))

        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(prev_output, training)
        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output *= rec_dp_mask

        output = h + K.dot(prev_output, self.recurrent_kernel)
        output = output + K.batch_dot(prev_output, self.plastic_kernel * plastic_vals)

        if self.activation is not None:
            output = self.activation(output)

        modulation = K.dot(prev_output, self.modulatory_kernel)
        plastic_vals = layers.Activation("tanh")(plastic_vals + modulation * trace)
        trace = self.plastic_update(self.plastic_lr, prev_output, output, trace)

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True
        return output, [output, K.batch_flatten(plastic_vals), K.batch_flatten(trace)]

    def get_config(self):
        config = {"plasticity_rule": self.plasticity_rule}
        base_config = super().get_config()
        return dict(**base_config, **config)


class NMPlasticRNN(tf.keras.layers.SimpleRNN):
    """
    Augments the Keras SimpleRNN with a differentiable plasticity mechanism.

    Reference:
        https://arxiv.org/abs/1804.02464
    """

    def __init__(
        self,
        units,
        activation="tanh",
        bias_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        dropout=0.0,
        kernel_constraint=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        plasticity_rule="oja",
        recurrent_constraint=None,
        recurrent_dropout=0.0,
        recurrent_initializer="orthogonal",
        recurrent_regularizer=None,
        use_bias=True,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        **kwargs,
    ):
        super().__init__(
            units,
            go_backwards=go_backwards,
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )
        self.plasticity_rule = plasticity_rule

        self.cell = NMPlasticRNNCell(
            units,
            activation=activation,
            bias_constraint=bias_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            dropout=dropout,
            kernel_constraint=kernel_constraint,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            plasticity_rule=plasticity_rule,
            recurrent_constraint=recurrent_constraint,
            recurrent_dropout=recurrent_dropout,
            recurrent_initializer=recurrent_initializer,
            recurrent_regularizer=recurrent_regularizer,
            use_bias=use_bias,
        )

    def get_config(self):
        config = {"plasticity_rule": self.plasticity_rule}
        base_config = super().get_config()
        return dict(**base_config, **config)


register_custom_objects(
    [
        PlasticRNNCell,
        PlasticRNN,
        PlasticGRUCell,
        PlasticGRU,
        PlasticLSTMCell,
        PlasticLSTM,
        NMPlasticRNNCell,
        NMPlasticRNN,
    ]
)
