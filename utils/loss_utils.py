import tensorflow as tf


def euclidean_distance(vec1, vec2):
    return tf.sqrt(tf.reduce_sum(tf.square(vec1 - vec2), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred, margin=1):
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))
