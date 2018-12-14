import tensorflow as tf
import cv2
import numpy as np
a = tf.get_variable(name='a', shape=[10, 10, 3], dtype=tf.uint8, initializer=tf.ones_initializer)
b = a*128
c = tf.image.convert_image_dtype(b,dtype=tf.float32)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(a))
    print(sess.run(c))