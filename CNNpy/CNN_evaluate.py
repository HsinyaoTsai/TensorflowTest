# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import CNN_inference
import CNN_train
import numpy as np

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    x = tf.placeholder(tf.float32, [mnist.validation.num_examples, CNN_inference.IMAGE_SIZE, CNN_inference.IMAGE_SIZE, CNN_inference.NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, CNN_inference.OUTPUT_NODE], name="y-input")
    xs = mnist.validation.images
    ys = mnist.validation.labels
    reshaped_xs = np.reshape(xs,(mnist.validation.num_examples, CNN_inference.IMAGE_SIZE, CNN_inference.IMAGE_SIZE, CNN_inference.NUM_CHANNELS))
    validate_feed = {x: reshaped_xs, y_:ys}

    y = CNN_inference.inference(x, None, None)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(CNN_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    while True:
        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(CNN_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print(" validation accuracy is %g" % ( accuracy_score))
            else:
                print("NO CP FOUND!")
                return
            time.sleep(EVAL_INTERVAL_SECS)
            '''
            tf.global_variables_initializer().run()
            for i in range(5000):
                if i % 500 == 0:
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %d steps, accuracy on validation is %g" % (i, validate_acc))
            '''

def main(argv = None) :
    mnist = input_data.read_data_sets("dataSets/", one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()