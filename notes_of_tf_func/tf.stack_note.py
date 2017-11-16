# -*- coding: utf-8 -*-

import tensorflow as tf

#原张量的shape（形状）为（3,1）,rank（秩）为R=3
a = tf.constant([[1], [2], [3]])
b = tf.constant([[4], [5], [6]])
c = tf.constant([[7], [8], [9]])

d = tf.stack([a,b,c], axis=0)#axis=0时拼接得到的张量的shape为（R,3,1）即（3,3,1）
e = tf.stack([a,b,c], axis=1)#axis=0时拼接得到的张量的shape为（3,R,1）即（3,3,1）
f = tf.stack([a,b,c], axis=-1)#axis=0时拼接得到的张量的shape为（3,1,R）即（3,1,3）
g = tf.stack([a,b,c], axis=-2)#axis=0时拼接得到的张量的shape为（3,R,1）即（3,3,1）

with tf.Session() as sess:
    print("得到的张量为：\n", sess.run(d), "\n张量的形状为：", sess.run(tf.shape(d)))
    print("得到的张量为：\n", sess.run(e), "\n张量的形状为：", sess.run(tf.shape(e)))
    print("得到的张量为：\n", sess.run(f), "\n张量的形状为：", sess.run(tf.shape(f)))
    print("得到的张量为：\n", sess.run(g), "\n张量的形状为：", sess.run(tf.shape(g)))

