import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channel': tf.FixedLenFeature([], tf.int64)
        }
    )
    decoded_image = tf.decode_raw(features['image'], tf.uint8)
    height, width, channel = [tf.cast(features['height'],tf.int32), tf.cast(features['width'],tf.int32), tf.cast(features['channel'],tf.int32)]
    decoded_image = tf.reshape(decoded_image, [height, width, channel])
    #decoded_image.set_shape([256, 256, 3])
    decoded_image = tf.image.convert_image_dtype(decoded_image,tf.float32)
    print(decoded_image.dtype)
    label = features['label']
    return decoded_image, label


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=0.12)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.50)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=0.12)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.50)
    # 记得clip，防止像素值大于1或小于0
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        # 转成float32格式方便处理
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # tf.shape可以用来获取tensor的shape
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox,
                                                                      min_object_covered=0.4)
    # 以boundingbox为基础切图片（具体参数由上面的函数提供）
    distored_image = tf.slice(image, bbox_begin, bbox_size)
    # 将随机截取的图像调整为神经网络输入层的大小。
    distored_image = tf.image.resize_images(distored_image, [height, width], method=0)
    # 进行变换
    distored_image = tf.image.random_flip_left_right(distored_image)
    distored_image = distort_color(distored_image, color_ordering=0)

    return distored_image