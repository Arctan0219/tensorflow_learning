# -*- coding: utf-8 -*-
'''
加载iris数据集，实现一个简单的二值分类器来预测一朵花是否为山鸢尾；
iris数据集中共有三类花，此处只预测是否为山鸢尾.
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets# to import the iris data

sess = tf.Session()

#将山鸢尾记为1，其他品种记为0；此训练中只使用两种特征：花瓣长度和宽度
iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])#这两个特征在第3、第4列

batch_size = 20
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

#定义线性模型:x1 = x2 * A +b
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target,
                                                   logits=my_output)

train = tf.train.GradientDescentOptimizer(0.05).minimize(xentropy)

sess.run(tf.global_variables_initializer())

train_steps = 1000
for step in range(train_steps):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0] for x in rand_x]]).transpose()
    rand_x2 = np.array([[x[1] for x in rand_x]]).transpose()
    rand_y = np.array([[y] for y in binary_target[rand_index]]) 
    sess.run(train, feed_dict={x1_data:rand_x1, x2_data:rand_x2, 
                               y_target:rand_y})
    if (step+1)%200==0:
        print('Step #' + str(step+1) + ' A = ' + str(sess.run(A)) + ', b = '
              + str(sess.run(b)))

#抽取模型变量并绘图
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
    ablineValues.append(slope*i + intercept)
setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1]
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]
plt.plot(setosa_x, setosa_y, 'g.', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'rx', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()










    