# -*- coding: utf-8 -*-
import tensorflow as tf
import os

def combine_inputs(X):
    return tf.matmul(X, W) + b

#将计算得到的值利用sigmoid函数输出为一个推断值
def prediction(X):
    return tf.sigmoid(combine_inputs(X))

def loss(X,Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = combine_inputs(X)))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__)
                                                     + "/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)#实例化reader对象
    key, value = reader.read(filename_queue)
    
    #decode_csv会将字符串转换到具有指定默认值的由张量列构成的元组中
    #它还会为每一列设置数据类型
    decoded = tf.decode_csv(value, record_defaults = record_defaults)
    
    #实际上会读取一个文件，并加载一个张量中的batch_size行
    return tf.train.shuffle_batch(decoded, batch_size = batch_size,
                                  capacity = batch_size * 50,
                                  min_after_dequeue = batch_size)

def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
    read_csv(100, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])
    
    #转换属性数据，将其转换为多维布尔型特征
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))
    
    gender = tf.to_float(tf.equal(sex, ["female"]))
    
    #将转换后的特征排列在一个矩阵中，再对其进行转置，
    #使每一行对应一个样本，每列对应一种属性
    features = tf.transpose(tf.stack([is_first_class, is_second_class, 
                                     is_third_class, gender, age]))
    survived = tf.reshape(survived, [100, 1])
    
    return features, survived

def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

#评估训练结果
def evaluate(sess, X, Y):
    
    predicted = tf.cast(prediction(X) > 0.5, tf.float32)
    
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


#参数和变量的初始化
W = tf.Variable(tf.zeros([5,1]), name="weights")
b = tf.Variable(0., name="bias")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    X, Y = inputs()
    
    total_loss = loss(X, Y)
    train_Op = train(total_loss)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        
    training_steps = 1000
    
    for step in range(training_steps):
        sess.run(train(total_loss))
        if step%100 == 0:
            print("loss: ", sess.run(total_loss))
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()






