# -*- coding: utf-8 -*-

import tensorflow as tf

A = tf.constant([[1, 2, 3, 4],
                 [4, 1, 2, 3],
                 [3, 4, 1, 2],
                 [2, 3, 4, 1]],dtype=tf.int32)

def do_argmax(aixs):
    return tf.argmax(A, aixs)

with tf.Session() as sess:
    print("when aixs equals 0: ",sess.run(do_argmax(0)))#aixs = 0
    print("when aixs equals 1: ",sess.run(do_argmax(1)))#aixs = 1