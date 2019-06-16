import numpy as np
import tensorflow as tf
offset = np.transpose(np.reshape(np.array([np.arange(4)] * 4 * 2),
            (2, 4,4)), (1, 2, 0))
print offset
print '.......'
a=tf.reshape(tf.constant(offset, dtype=tf.float32),[1,4,4,2]    ) 
print a             
set = tf.tile(a, [3, 1, 1, 1])      
print set
sett=tf.transpose(set,(0,2,1,3))
print sett
print 'sssssssssssssssss'
with tf.Session() as sess:
      print sess.run(a)
      print 'xxxxxxxxxxxxxxxxxx'
      print sess.run(sett)
