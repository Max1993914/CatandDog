import tensorflow as tf
import numpy as np
import cv2
from model import model
from data_preprocess import parser
from data_preprocess import preprocess_for_train
import os
import matplotlib.pyplot as plt

INPUT_SIZE = 209
OUTPUT_SIZE = 2
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.999
REGULARIZATION_RATE = 0.0001
TRAINING_STEP = 35000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "C:/backup/TFR_for_catanddog/check_point"
MODEL_NAME = "model.ckpt"

train_files = tf.train.match_filenames_once("C:/backup/TFR_for_catanddog/catdog_data_TFR/tfrcatdog.tfrecords*")


def train():
    # 读取TFR里的数据
    shuffle_buffer = 10000

    dataset = tf.data.TFRecordDataset(train_files)
    dataset = dataset.map(parser)  # 从对每一个example使用parser,得到decoded_image, label二元组

    # lamda表示dataset存的东西（这里是二元组）。冒号后面表示处理后返回的东西。
    #dataset = dataset.map(lambda image, label: (preprocess_for_train(image, INPUT_SIZE, INPUT_SIZE, None), label))
    # 将整个数据集进行shuffle然后组成batch
    dataset = dataset.shuffle(shuffle_buffer).batch(BATCH_SIZE)

    NUM_EPOCH = 10000
    # 将数据集中的数据复制N份，相当于间接地提供了训练轮数
    dataset = dataset.repeat(NUM_EPOCH)
    # 定义迭代器。因为使用tf.train.match_filenames_once和placeholder机制相近，所以iterator选用下面的而不是make_one_shot_iterator
    iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()

    # x的维度应该是（m，nx，ny，nc）
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 3], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = model(x, train, regularizer)

    global_step = tf.Variable(0, trainable=False)

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_average_op = variable_average.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 3670为测试的总图片数
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               3670 / BATCH_SIZE, LEARNING_RATE_DECAY)

    train_step = tf.train.AdamOptimizer(learning_rate).\
        minimize(loss, global_step=global_step)

    train_op = tf.group(train_step, variable_average_op)

    precision = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(precision, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        sess.run(iterator.initializer)
        for i in range(TRAINING_STEP):
            xs, ys = sess.run([image_batch, label_batch])
            labels,_, loss_value, step,accu = sess.run([y_,train_op, loss, global_step,accuracy], feed_dict={x: xs, y_: ys})
            # plt.imshow(la[0,:,:,:])
            # plt.show()
            if i % 10 == 0:
                print(labels)
                print("when the step is %d, the loss_value is %g" % (step, loss_value))
                print("the accuracy is %g" % (accu))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


if __name__ == "__main__":
    train()
