import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import os
import data
import DNN
img_size=448
batch=1

def loss():
    pass


if __name__=='__main__':
	imgs_input,labels_input=data.get_batch_input()
	DNN_out=DNN.DNN(imgs_input)  
	
	print imgs_input,labels_input,DNN_out
 
	pre_class=tf.reshape(DNN_out[:,:7*7*20],[batch,7,7,20])
	pre_confidence=tf.reshape(DNN_out[:,7*7*20:7*7*22],[batch,7,7,2])
	pre_bbox=tf.reshape(DNN_out[:,7*7*22:],[batch,7,7,2,4])

	gd_class=tf.reshape(labels_input[...,5:],[batch,7,7,20])
	gd_confidence=tf.reshape(labels_input[...,0],[batch,7,7,1])
	gd_bbox=tf.reshape(labels_input[...,1:5],[batch,7,7,1,4])
	




	
	with tf.Session() as sess:
	        print sess.run(gd_bbox)
	









