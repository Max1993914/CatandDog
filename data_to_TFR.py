import tensorflow as tf
import numpy as np
import glob
import os.path
import PIL
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
from train import INPUT_SIZE
import cv2

# 输入数据的地址
INPUT_DATA = 'C:/backup/catanddog'


def create_image_lists(sess):
    sub_dirs = []

    # 存放各种图片
    current_label = 1

    # 遍历图片目录
    for x in os.walk(INPUT_DATA):
        sub_dirs.append(x[0])

    file_index = 4
    # 取得除了根目录之外的其他五个包含五种花的文件目录
    for sub_dir in sub_dirs:
        print(sub_dir)
        index_of_picture = 0
        training_images = []
        training_labels = []
        if sub_dir == INPUT_DATA:
            continue
        else:
            file_list = []
            dir_name = os.path.basename(sub_dir)
            # *与jpg是连在一起的因为这就是文件名，所以要放一块
            file_glob = os.path.join(INPUT_DATA, dir_name, '*jpg')
            file_list.extend(glob.glob(file_glob))
            for file_name in file_list:
                image_raw_data = tf.gfile.FastGFile(file_name, 'rb').read()
                image_data = tf.image.decode_jpeg(image_raw_data)  # 转进来默认是uint8属性的（2的8次方嘛0到255）
                if image_data.dtype != tf.uint8:
                    image_data = tf.image.convert_image_dtype(image_data, tf.uint8)
                image_data = tf.image.resize_images(image_data, [INPUT_SIZE, INPUT_SIZE], method=1)
                image_data_value = sess.run(image_data)
                # plt.imshow(image_data_value)
                # plt.show()
                training_images.append(image_data_value)
                training_labels.append(current_label)
                print("current_label is %d" % (current_label))
                print("data is processing on number %d picture" % (index_of_picture))
                index_of_picture += 1
                if index_of_picture % 2499 == 0:
                    togo = [training_images, training_labels]
                    record_to_tfr(togo[0], togo[1], file_index)
                    training_images.clear()
                    training_labels.clear()
                    file_index += 1
        current_label += 1  # label 对应的是0，1


def record_to_tfr(images, labels, index):

    assert len(images) == len(labels)
    file_name = ('C:/backup/TFR_for_catanddog/catdog_data_TFR/tfrcatdog.tfrecords-%.5d-of-%.5d' % (index, 1))
    writer = tf.python_io.TFRecordWriter(file_name)

    for i in range(len(images)):
        print("current label is %d" % labels[i])
        print("current image shape is %d x %d" %(images[i].shape[0], images[i].shape[1]))
        print("current image channel is %d" % (images[i].shape[2]))
        s_image = images[i].tobytes()
        print("picture is saving to TFR")
        example = tf.train.Example(features=tf.train.Features(feature=
        {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[s_image])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[images[i].shape[1]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[images[i].shape[0]])),
            'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[images[i].shape[2]]))
        }))
        # 将得到的example写入tfr文件
        writer.write(example.SerializeToString())
    writer.close()


def main():
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 队列操作需要使用局部变量，必须初始化
        tf.local_variables_initializer().run()
        create_image_lists(sess)


if __name__ == '__main__':
    main()