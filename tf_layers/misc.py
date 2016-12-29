"""Collection of different general purpose layers
"""


def prelu(data, name_or_scope=None):
    with tf.variable_scope(
            name_or_scope,
            default_name='prelu',
            values=[data]):
        alphas = tf.get_variable(shape=data.get_shape().as_list()[-1:],
                                 initializer=tf.constant_initializer(0.01),
                                 name="alphas")

        return tf.nn.relu(data) + tf.mul(alphas, (data - tf.abs(data))) * 0.5
