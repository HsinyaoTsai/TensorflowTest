#coding=utf8
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import CNN_inference
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "savedModel/"
MODEL_NAME = "CNNmodel.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, CNN_inference.IMAGE_SIZE, CNN_inference.IMAGE_SIZE, CNN_inference.NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, CNN_inference.OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = CNN_inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)
    #创建滑动平均的影子变量，并把它们应用在所有可以训练的变量上
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #argmax?
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #和tf.group等效，用来同时维护train_step和更新滑动平均值两个动作
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(BATCH_SIZE, CNN_inference.IMAGE_SIZE, CNN_inference.IMAGE_SIZE, CNN_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 500 == 0:
                print("After %d training steps, loss is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv = None):
    mnist = input_data.read_data_sets("dataSets/", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()