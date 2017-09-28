import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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
    data_sets = input_data.read_data_sets(data_dir)
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

if __name__ == "__main__": tf.app.run(main)
