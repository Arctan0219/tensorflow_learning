import tensorflow as tf
import numpy as np

#%%定义加入一个神经层的函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))#Weights开头大写只是习惯（矩阵开头大写）
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)#因为Variable推荐初始值不为0，故加上0.1
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
    #如果是None则表示线性关系，只需要保持原型
        outputs = Wx_plus_b
    else:
    #否则将值传入激活函数中
        outputs = activation_function(Wx_plus_b)
    return outputs

#%%定义数据形式
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#%%
xs = tf.placeholder(tf.float32, None)
ys = tf.placeholder(tf.float32, None)
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)#输入层
prediction = add_layer(l1, 10, 1, activation_function=None)#预测层

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
                     reduction_indices = [1]))#计算误差

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))