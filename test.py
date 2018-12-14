import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
from train import INPUT_SIZE
from train import MOVING_AVERAGE_DECAY
from model import model
import os.path
import glob


def test():
    with tf.Graph().as_default() as g:
        raw_image = gfile.FastGFile("test_image/cat.1.jpg", mode="rb").read()
        image = tf.image.decode_jpeg(raw_image)
        image = tf.image.resize_images(image, [INPUT_SIZE, INPUT_SIZE], method=1)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        label = ["is cat", "is dog"]
        x = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 3], name="x")
        y = model(x, None,None)  # 1,2
        index_y = tf.argmax(y, 1)
        variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('C:/backup/TFR_for_catanddog/check_point')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("no ckpt")
            input_image = sess.run(image)
            plt.imshow(input_image[0])
            plt.show()
            input, index = sess.run([y, index_y], feed_dict={x: input_image})
            print(input)
            print(label[index[0]])


if __name__ == "__main__":
    test()