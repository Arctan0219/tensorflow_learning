# -*- coding: utf-8 -*-
import tensorflow as tf
import os

#difine the weights and bias
W = tf.Variable(tf.zeros([4,3]), name="weights")
b = tf.Variable(tf.zeros([3]), name="bias")

def combine_inputs(X):
    return tf.matmul(X, W) + b

def prediction(X):
    return tf.nn.softmax(combine_inputs(X))#use softmax function in multi ouput case

def loss(X,Y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits\
               (labels=Y, logits=combine_inputs(X)))

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
    sepal_length, sepal_width, petal_length, petal_width, label = \
    read_csv(100, "iris.csv", [[0.0], [0.0], [0.0], [0.0], [""]])
    #convert species to category index begin from 0
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
                        tf.equal(label, ["Iris-setosa"]),
                        tf.equal(label, ["Iris-versicolor"]),
                        tf.equal(label, ["Iris-virginica"])])), 0))
    #将转换后的特征排列在一个矩阵中，再对其进行转置，
    #使每一行对应一个样本，每列对应一种属性
    features = tf.transpose(tf.stack([sepal_length, sepal_width,
                                      petal_length, petal_width]))  

    return features, label_number

def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    predicted = tf.cast(tf.argmax(prediction(X), 1), tf.int32)
    
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))
    
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    X, Y = inputs()
    
    total_loss = loss(X, Y)
    train_Op = train(total_loss)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        
    training_steps = 1
    
    for step in range(training_steps):
        sess.run(train(total_loss))
        if step%10 == 0:
            print("loss: ", sess.run(total_loss))
#    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()


 
  