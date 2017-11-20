# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

#%% 分类算法的损失函数
x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500,], 1.)

#%% Hinge损失函数
hinge_y_vals = tf.maximum(0., 1. -tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

#%% 交叉熵损失函数（Cross-entropy loss）
xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. -
                          target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

#%% Sigmoid交叉熵损失函数（Sigmoid cross entropy loss）
#sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, 
#                                 logits=None, name=None)
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_vals, 
                                                                  labels=targets)
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

#%% 加权交叉熵损失函数（Weighted cross entropy loss）
weight = tf.constant(0.5)
#weighted_cross_entropy_with_logits(targets, logits, pos_weight, name=None)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(targets,
                                                            x_vals, weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

#%% Softmax交叉熵损失函数（Softmax cross-entropy loss）
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
#softmax_cross_entropy_with_logits(labels=..., logits=..., ...)
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits,
                                                           labels=target_dist)
print(sess.run(softmax_xentropy))

#%% 稀疏Softmax交叉熵损失函数（Sparse softmax cross-entropy loss）
unscaled_logits = tf.constant([[1., -3., 10.]])
sparse_target_dist = tf.constant([2])
#sparse_softmax_cross_entropy_with_logits(labels=..., logits=..., ...)
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaled_logits,
                                                                 labels=sparse_target_dist)
print(sess.run(sparse_xentropy))

#%% 绘图
x_array = sess.run(x_vals)
plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xentropy_sigmoid_y_out, 'k-', 
         label='Cross Entropy Sotfmoid Loss')
plt.plot(x_array, xentropy_weighted_y_out, 'g:', 
         label='Weighted Cross Entropy Loss (x0.5)')
plt.ylim(-1.5, 3)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
