import tensorflow as tf


def get_num_channels(x):
    """
    returns the input's number of channels
    :param x: input tensor with shape [batch_size, ..., num_channels]
    :return: number of channels
    """
    return x.get_shape().as_list()[-1]


def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.contrib.layers.xavier_initializer(uniform=False)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initial bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def conv_2d(x, filter_size, num_filters, layer_name, stride=1, use_relu=True):
    """
    Create a 2D convolution layer
    :param x: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param stride: convolution filter stride
    :param use_relu: whether to use Relu or not
    :return: The output array
    """
    num_in_channel = get_num_channels(x)
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        weights = weight_variable(layer_name, shape=shape)
        tf.summary.histogram('W', weights)
        layer = tf.nn.conv2d(input=x,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        print('{}: {}'.format(layer_name, layer.get_shape()))
        biases = bias_variable(layer_name, [num_filters])
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
    return layer


def max_pool(x, ksize, stride, name):
    """
    Create a max pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param stride: stride of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="SAME",
                          name=name)


def flatten_layer(layer):
    """
    Flattens the output of the convolutional layer to be fed into fully-connected layer
    :param layer: input array
    :return: flattened array
    """
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def fc_layer(x, num_units, layer_name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param layer_name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        in_dim = x.get_shape()[1]
        W = weight_variable(layer_name, shape=[in_dim, num_units])
        tf.summary.histogram('weight', W)
        b = bias_variable(layer_name, shape=[num_units])
        tf.summary.histogram('bias', b)
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer


def dropout(x, rate, training):
    """Create a dropout layer."""
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

