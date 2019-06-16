import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import os
import data

def  DNN(input,keep_prob=0.7,is_training=True,num_outputs=7*7*30):
	net = tf.pad(input, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
        net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
        net = slim.conv2d(net, 192, 3, scope='conv_4')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
        net = slim.conv2d(net, 128, 1, scope='conv_6')
        net = slim.conv2d(net, 256, 3, scope='conv_7')
        net = slim.conv2d(net, 256, 1, scope='conv_8')
        net = slim.conv2d(net, 512, 3, scope='conv_9')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
        net = slim.conv2d(net, 256, 1, scope='conv_11')
        net = slim.conv2d(net, 512, 3, scope='conv_12')
        net = slim.conv2d(net, 256, 1, scope='conv_13')
        net = slim.conv2d(net, 512, 3, scope='conv_14')
        net = slim.conv2d(net, 256, 1, scope='conv_15')
        net = slim.conv2d(net, 512, 3, scope='conv_16')
        net = slim.conv2d(net, 256, 1, scope='conv_17')
        net = slim.conv2d(net, 512, 3, scope='conv_18')
        net = slim.conv2d(net, 512, 1, scope='conv_19')
        net = slim.conv2d(net, 1024, 3, scope='conv_20')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
        net = slim.conv2d(net, 512, 1, scope='conv_22')
        net = slim.conv2d(net, 1024, 3, scope='conv_23')
        net = slim.conv2d(net, 512, 1, scope='conv_24')
        net = slim.conv2d(net, 1024, 3, scope='conv_25')
        net = slim.conv2d(net, 1024, 3, scope='conv_26')
        net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),name='pad_27')
        net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
        net = slim.conv2d(net, 1024, 3, scope='conv_29')
        net = slim.conv2d(net, 1024, 3, scope='conv_30')
        net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
        net = slim.flatten(net, scope='flat_32')
        net = slim.fully_connected(net, 512, scope='fc_33')
        net = slim.fully_connected(net, 4096, scope='fc_34')
        net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
        net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')
        return net

if __name__=='__main__':
	a,b=data.get_batch_input()
	out=DNN(a)
	print out
        with tf.Session() as sess:
	      tf.global_variables_initializer().run()
	      print sess.run(out)
