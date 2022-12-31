import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid, tanh
LSTMStateTuple = tf.compat.v1.nn.rnn_cell.LSTMStateTuple


class MyLSTM(tf.compat.v1.nn.rnn_cell.BasicLSTMCell):
    def linear(self, input_, output_size, scope=None):
        '''
        Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
        Args:
            args: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
        Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
        Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
        '''

        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]

        # Now the computation.
        with tf.compat.v1.variable_scope(scope or "SimpleLinear"):
            matrix = tf.compat.v1.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
            bias_term = tf.compat.v1.get_variable("Bias", [output_size], dtype=input_.dtype)

        return tf.matmul(input_, tf.transpose(matrix)) + bias_term

    def __call__(self, inputs, state, scope=None):
        """LSTM as mentioned in paper."""
        with vs.variable_scope(scope or "basic_lstm_cell"):
            # Parameters of gates are concatenated into one multiply for
            # efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(
                    value=state, num_or_size_splits=2, split_dim=1)
            g = tf.compat.v1.concat(1, [inputs, h])
            concat = self.linear([g], 4 * self._num_units, scope=scope)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(
                value=concat, num_split=4, split_dim=1)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat_v2([new_c, new_h], 1)
            return new_h, new_state
