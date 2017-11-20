# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

#%% 回归算法的损失函数
x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)
#%% L2正则损失函数，即预测值与目标值差值的平方和
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

#%% L1正则损失函数，即对预测值与目标值差值求绝对值
l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

#%% Pseudo-Huber损失函数，是对Huber损失函数的连续、平滑估计
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + 
                        tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)

delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + 
                        tf.square((target - x_vals)/delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)

#%% 绘图
x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
plt.plot(x_array, phuber1_y_out, 'k-', label='P-Huber Loss (0.25)')
plt.plot(x_array, phuber2_y_out, 'g:', label='P-Huber Loss (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
















































