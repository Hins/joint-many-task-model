import tensorflow as tf
import numpy as np


def length(sequence):
    used = tf.compat.v1.sign(tf.compat.v1.reduce_max(tf.compat.v1.abs(sequence), reduction_indices=2))
    length = tf.compat.v1.reduce_sum(used, reduction_indices=1)
    length = tf.compat.v1.cast(length, tf.compat.v1.int32)
    return length


def cost(output, target):
    cross_entropy = target * tf.compat.v1.log(output)
    cross_entropy = -tf.compat.v1.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.compat.v1.sign(tf.compat.v1.reduce_max(tf.compat.v1.abs(target), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.compat.v1.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.compat.v1.reduce_sum(mask, reduction_indices=1)
    return tf.compat.v1.reduce_mean(cross_entropy, name="loss")


def activate(outputs, weight_shape, bias_shape, activation=tf.compat.v1.nn.softmax):
    dim_str = {3: 'ijk,kl->ijl', 2: 'ij,jk->ik'}
    weights = tf.compat.v1.get_variable(
        "weights", shape=weight_shape, initializer=tf.compat.v1.random_normal_initializer())
    biases = tf.compat.v1.get_variable("biases", shape=bias_shape,
                             initializer=tf.compat.v1.constant_initializer(0.0))
    if outputs.get_shape().ndims == 2:
        result = activation(tf.compat.v1.matmul(outputs, weights) + biases)
    else:
        result = activation(tf.compat.v1.reshape(tf.compat.v1.matmul(tf.compat.v1.reshape(outputs, [-1, weight_shape[
            0]]), weights), [-1, outputs.get_shape().as_list()[1], weight_shape[1]]) + biases)

    return result


def rmse_loss(outputs, targets):
    return tf.compat.v1.sqrt(tf.compat.v1.reduce_mean(tf.compat.v1.square(tf.compat.v1.subtract(targets, outputs))))


def pad(x, max_length, pad_constant=-1):
    x = list(x)
    for i in range(len(x)):
        x[i] += [pad_constant] * (max_length - len(x[i]))
        x[i] = np.array(x[i])
    return x


def get_batch_pos(obj, size=5):
    idx = np.random.choice(range(len(obj.sent)), size=size, replace=False)
    p = pad(obj.pos[idx], obj.max_length, -1)
    s = pad(obj.sent[idx], obj.max_length, obj.vec.shape[0] - 1)
    c = pad(obj.chun[idx], obj.max_length, -1)
    return s, p, c


def get_batch_sent(obj, size=5):
    idx = np.random.choice(range(len(obj.sent1)), size=size, replace=False)
    s1 = pad(obj.sent1[idx], obj.max_length, obj.vec.shape[0] - 1)
    s2 = pad(obj.sent2[idx], obj.max_length, obj.vec.shape[0] - 1)
    r = obj.rel[idx]
    e = obj.ent[idx]
    return s1, s2, r.values, e.values
