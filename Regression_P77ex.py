# -*- coding: utf-8 -*-
import tensorflow as tf

def prediction(X):
    return tf.matmul(X, W)+b

def loss(X, Y):
    Y_pre = prediction(X)
    return tf.reduce_sum(tf.squared_difference(Y,Y_pre))#先求预测值与实际值的平方差，再将所有的平方差求和

def inputs():
    weight_age = [[84,46],[73,20],[65,52],[70,30],[76,57],[69,25],[63,28],
                  [72,36],[79,57],[75,44],[27,24],[89,31],[65,52],[57,23],
                  [59,60],[69,48],[60,34],[79,51],[75,50],[82,34],[59,46],
                  [67,23],[85,37],[55,40],[63,30]]
    blood_fat_content = [354,190,405,263,451,302,288,385,402,365,209,290,346,
                         254,395,434,220,374,308,220,311,181,274,303,244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)#将数据类型转换为float32

def train(total_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

#define weights and biases
W = tf.Variable(tf.zeros([2,1]), dtype=tf.float32, name='weights')
b = tf.Variable(0., dtype=tf.float32, name='biases')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    X, Y = inputs()
    
    total_loss = loss(X, Y)
    train_Op = train(total_loss)
    
    training_steps =10000
    for step in range(training_steps):
        sess.run(train_Op)
        if step % 1000 == 0:
            print('loss:', sess.run(total_loss))
            
    print(sess.run(prediction([[80., 25.],
                               [65., 25.],
                               [84., 46.]])))