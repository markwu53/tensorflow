import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_dir = "/tmp/tensorflow/mnist/input_data" 
batch_size = 100
hidden1 = 128
hidden2 = 32
learning_rate = .01
max_steps = 2000

def main(_):
    data_sets = input_data.read_data_sets(data_dir)

if __name__ == "__main__": tf.app.run(main)
