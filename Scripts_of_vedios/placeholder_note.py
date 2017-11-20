import tensorflow as tf
 
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))
    
#placeholder在使用时，需要在sess.run()中加入一个包含所需要变量的字典（dict）