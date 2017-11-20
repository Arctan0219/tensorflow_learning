import tensorflow as tf

    #定义加入一个神经层
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