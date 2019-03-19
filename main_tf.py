# references:
# https://github.com/sudharsan13296/Hands-On-Meta-Learning-With-Python
# https://github.com/sudharsan13296/Hands-On-Meta-Learning-With-Python/blob/master/02.%20Face%20and%20Audio%20Recognition%20using%20Siamese%20Networks/2.4%20Face%20Recognition%20Using%20Siamese%20Network.ipynb
# data: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
# others: https://github.com/leimao/Siamese_Network_MNIST
# https://github.com/ardiya/siamesenetwork-tensorflow

from utils.data_utils import get_data, randomize, get_next_batch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils.network_utils import conv_2d, max_pool, dropout, flatten_layer, fc_layer

# data parameters
TOTAL_SAMPLE_SIZE = 10000
X, Y = get_data(TOTAL_SAMPLE_SIZE)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
H, W = x_train.shape[-2], x_train.shape[-1]

# Hyper-parameters
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DISPLAY_FREQ = 100
LOGS_PATH = "./logs"  # path to the folder that we want to save the logs for Tensorboard

# network parameters
NB_FILTERS = [6, 12]
FILTER_SIZE = 3
m = 1       # margin for the contrastive loss


def build_base_network(x, num_kernels=[6, 12], k=3, is_train=True):
    for i, n_kernel in enumerate(num_kernels):
        x = conv_2d(x, filter_size=k, num_filters=n_kernel, layer_name='conv_'+str(i+1), stride=1, use_relu=True)
        x = max_pool(x, ksize=2, stride=2, name='pool_'+str(i+1))
        x = dropout(x, rate=0.1, training=is_train)
    x = flatten_layer(x)
    x = fc_layer(x, num_units=128, layer_name='fc_1', use_relu=True)
    x = dropout(x, rate=0.1, training=is_train)
    x = fc_layer(x, num_units=50, layer_name='fc_2', use_relu=True)
    return x


def euclidean_distance(vec1, vec2):
    return tf.sqrt(tf.reduce_sum(tf.square(vec1 - vec2), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred, margin=1):
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))


with tf.variable_scope('Input'):
    x1 = tf.placeholder(tf.float32, shape=[None, H, W, 1], name='X1')
    x2 = tf.placeholder(tf.float32, shape=[None, H, W, 1], name='X2')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

with tf.variable_scope('Siamese'):
    x1_embed = build_base_network(x1)
    x2_embed = build_base_network(x2)

with tf.variable_scope('Train'):
    with tf.variable_scope('Loss'):
        d_w = euclidean_distance(x1_embed, x2_embed)
        loss = contrastive_loss(y, d_w, margin=m)
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='Adam-op').minimize(loss)
    with tf.variable_scope('Accuracy'):
        pass


# Initialize the variables
init = tf.global_variables_initializer()
# Merge all summaries
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    global_step = 0
    summary_writer = tf.summary.FileWriter(LOGS_PATH, sess.graph)
    # Number of training iterations in each epoch
    num_tr_iter = int(len(y_train) / BATCH_SIZE)
    for epoch in range(EPOCHS):
        print('Training epoch: {}'.format(epoch + 1))
        x_train, y_train = randomize(x_train, y_train)
        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * BATCH_SIZE
            end = (iteration + 1) * BATCH_SIZE
            x1_batch, x2_batch, y_batch = get_next_batch(x_train, y_train, start, end)

            # Run optimization op (backprop)
            feed_dict_batch = {x1: x1_batch, x2: x2_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)

            if iteration % DISPLAY_FREQ == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch, acc_batch, summary_tr = sess.run([loss, accuracy, merged],
                                                             feed_dict=feed_dict_batch)
                summary_writer.add_summary(summary_tr, global_step)

                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                      format(iteration, loss_batch, acc_batch))

        # Run validation after every epoch
        feed_dict_valid = {x: x_valid, y: y_valid}
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
              format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')


print()


