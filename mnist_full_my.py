import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import time

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

data_dir = "/tmp/tensorflow/mnist/input_data" 
batch_size = 100
hidden1 = 128
hidden2 = 32
learning_rate = .01
max_steps = 2000

def main(_):
    #building nn layers
    images = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    labels = tf.placeholder(tf.int32, shape=(batch_size))
    with tf.name_scope("hidden1"):
        weights = tf.Variable( tf.truncated_normal([IMAGE_PIXELS, hidden1], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name="weights")
        biases = tf.Variable(tf.zeros([hidden1]), name="biases")
        hidden1_out = tf.nn.relu(tf.matmul(images, weights) + biases)
    with tf.name_scope("hidden2"):
        weights = tf.Variable( tf.truncated_normal([hidden1, hidden2], stddev=1.0 / math.sqrt(float(hidden1))), name="weights")
        biases = tf.Variable(tf.zeros([hidden2]), name="biases")
        hidden2_out = tf.nn.relu(tf.matmul(hidden1_out, weights) + biases)
    with tf.name_scope("softmax_linear"):
        weights = tf.Variable( tf.truncated_normal([hidden2, NUM_CLASSES], stddev=1.0 / math.sqrt(float(hidden2))), name="weights")
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name="biases")
        logits = tf.matmul(hidden2_out, weights) + biases
    with tf.name_scope("loss"):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    with tf.name_scope("training"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    with tf.name_scope("evaluation"):
        correct = tf.nn.in_top_k(logits, labels, 1)
        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

    data_sets = input_data.read_data_sets(data_dir)
    with tf.Session() as sess:
        def do_eval(data_set):
            true_count = 0
            steps_per_epoch = data_set.num_examples // batch_size
            num_examples = steps_per_epoch * batch_size
            for step in range(steps_per_epoch):
                images_feed, labels_feed = data_set.next_batch(batch_size)
                feed_dict = { images: images_feed, labels: labels_feed, }
                true_count += sess.run(eval_correct, feed_dict=feed_dict)
            precision = float(true_count) / num_examples
            print("  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f" % (num_examples, true_count, precision))
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for step in range(max_steps):
            images_feed, labels_feed = data_sets.train.next_batch(batch_size)
            feed_dict = { images: images_feed, labels: labels_feed, }
            sess.run(train_op, feed_dict=feed_dict)
            if step % 100 == 0:
                duration = time.time() - start_time
                start_time = time.time()
                loss_value = sess.run(loss, feed_dict=feed_dict)
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                print('Training Data Eval:')
                do_eval(data_sets.train)
                print('Validation Data Eval:')
                do_eval(data_sets.validation)
                print('Test Data Eval:')
                do_eval(data_sets.test)

if __name__ == "__main__": tf.app.run(main)
