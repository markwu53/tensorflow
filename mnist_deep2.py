import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

def consoleLog(message): print("[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
def conv2d(x, W): return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x): return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def weight_variable(shape): return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape): return tf.Variable(tf.constant(0.1, shape=shape))
def main(_):
    mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('reshape'): x_image = tf.reshape(x, [-1, 28, 28, 1])
    with tf.name_scope('conv1'): h_conv1 = tf.nn.relu(conv2d(x_image, weight_variable([5, 5, 1, 32])) + bias_variable([32]))
    with tf.name_scope('pool1'): h_pool1 = max_pool_2x2(h_conv1)
    with tf.name_scope('conv2'): h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_variable([5, 5, 32, 64])) + bias_variable([64]))
    with tf.name_scope('pool2'): h_pool2 = max_pool_2x2(h_conv2)
    with tf.name_scope('fc1'): h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2, [-1, 7 * 7 * 64]), weight_variable([7 * 7 * 64, 1024])) + bias_variable([1024]))
    with tf.name_scope('dropout'): h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    with tf.name_scope('fc2'): y_conv = tf.matmul(h_fc1_drop, weight_variable([1024, 10])) + bias_variable([10])
    with tf.name_scope('loss'): cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    with tf.name_scope('adam_optimizer'): train_step = tf.train.AdamOptimizer(1e-4).minimize(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    with tf.name_scope('accuracy'): accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0: consoleLog('step %d, training accuracy %g' % (i, accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == "__main__": tf.app.run(main)
