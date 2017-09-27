import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import time

data_dir = "/tmp/tensorflow/mnist/input_data" 
batch_size = 100
hidden1 = 128
hidden2 = 32
learning_rate = .01
max_steps = 2000

def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = { images_pl: images_feed, labels_pl: labels_feed, }
    return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

def main(_):
    data_sets = input_data.read_data_sets(data_dir)
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    logits = mnist.inference(images_placeholder, hidden1, hidden2)
    loss = mnist.loss(logits, labels_placeholder)
    train_op = mnist.training(loss, learning_rate)
    eval_correct = mnist.evaluation(logits, labels_placeholder)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(max_steps):
        start_time = time.time()
        feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        duration = time.time() - start_time
        if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
            print('Training Data Eval:')
            do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
            print('Validation Data Eval:')
            do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
            print('Test Data Eval:')
            do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)

if __name__ == "__main__": tf.app.run(main)