import tensorflow as tf
import numpy as np


# 卷积层
def conv(input_tensor, size, channel_num, filter_num, layer_name, train, padsize=1):
    with tf.variable_scope(layer_name):
        weight =tf.get_variable("weight", [size, size, channel_num, filter_num],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", [filter_num],
                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        out = tf.pad(input_tensor, [[0, 0], [padsize, padsize], [padsize, padsize], [0, 0]])
        out = tf.nn.conv2d(out, weight, strides=[1, 1, 1, 1], padding="VALID")
        out = tf.nn.relu(tf.nn.bias_add(out, bias))

        out = tf.layers.batch_normalization(out, axis=-1, momentum=0.9, training=True, name=layer_name+'bn')
        # else:
        #     out = tf.layers.batch_normalization(out, axis=-1, momentum=0.9, training=False, name=layer_name + 'bn')
        return out


# 池化层
def pool(input_tensor, size, stride, layer_name):
    with tf.variable_scope(layer_name):
        out = tf.nn.max_pool(input_tensor, [1, size, size, 1], [1, stride, stride, 1], padding="SAME")
        return out


# # 带passthrough的重组层
# def reorg(input_tensor, stride):
#     out = tf.space_to_depth(input_tensor,block_size=stride)
#     return out


# full-connection全连接层
def fc(input_tensor, input_nodes, fc_nodes, regularizer, layer_name, train, activation=True):
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', [input_nodes, fc_nodes],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(weight))
        if train:
            tf.nn.dropout(weight, 0.5)
        bias = tf.get_variable("bias", [fc_nodes], initializer=tf.truncated_normal_initializer(stddev=0.1))
        out = tf.matmul(input_tensor, weight)+bias
        if activation:
            out = tf.nn.relu(out)
        return out


def model(input_tensor, train, regularizer):
    net = conv(input_tensor, size=3, channel_num=3, filter_num=16, layer_name="conv1", train=train, padsize=1)  # m,209，209，16
    net = pool(net, 2, 2, "pool1")  # 104,104,16
    net = conv(net, 3, 16, 16, 'conv2', train=train, padsize=1)  # 104,104,16
    net = pool(net, 2, 2, "pool2")   # 52,52,16

    shape = net.get_shape()
    nodes = shape[1]*shape[2]*shape[3]
    net = tf.reshape(net, [-1, nodes])  # 将convolutional模式切换到fully-connection模式
    net = fc(net, nodes, 256, regularizer, layer_name="fc1", train=train, activation=True)
    net = fc(net, 256, 128, regularizer, layer_name="fc2", train=train, activation=True)
    net = fc(net, 128, 2, regularizer, layer_name="fc3", train=train, activation=False)  # m,2
    return net

# 测试用
# if __name__ == "__main__":
#     test_image = tf.random_normal([10,128,128,3])
#     out = model(test_image,None,None)
#     with tf.Session() as sess:
#         tf.global_variables_initializer().run()
#         a = sess.run(out)
#         print(a)






