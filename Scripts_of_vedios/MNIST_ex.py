# -*- coding: utf-8 -*-
#读取MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#%%
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print("MNIST数据载入完成!")

x = tf.placeholder(tf.float32, [None, 784])#1*784维的向量表示一张像素为28*28的图片

W = tf.Variable(tf.zeros([784, 10]))#W的维数是784*10，即784个权值乘上像素的值后分成10个类别
b = tf.Variable(tf.zeros([10]))#b是10个类别上累加的偏置

y = tf.nn.softmax(tf.matmul(x, W) + b)#实现softmax regression模型

y_ = tf.placeholder(tf.float32, [None, 10])#存放图片正确的label值

cross_entropy = -tf.reduce_sum(y_*tf.log(y))#求交叉熵，作为代价函数

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()#初始化

sess = tf.Session()
sess.run(init)

#训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_:batch_ys})
#%%    
#对比计算结果与真实标签的对比的布尔值
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#将布尔值转变成浮点型的1、0，再求平均值，即表示准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#使用测试集验证准确率
print("测试集的准确率为：", sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))