import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



# MNIST数据集相关的常熟
INPUT_NODE = 784          #输入层的节点数，对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 10          #输出层的节点数，这个等于类别的数目，MNIST数据集中，有0-9一共10个类别

# 配置神经网络的参数
LAYER1_NODE = 500               #隐藏层的节点数，这里使用只有一个隐藏层的网络结构
BATCH_SIZE = 100                #一个训练batch中的训练数据个数，数字越小，训练过程越接近随机梯度下降；数字越大，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8        #基础的学习率
LEARNING_RATE_DECAY = 0.99      #学习率的衰减率
REGULARIZATION_RATE = 0.0001    #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000          #训练轮数
MOVING_AVERAGE_DECAY = 0.99     #滑动平均衰减率

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    """
    :param input_tensor:
    :param avg_class: 滑动平均类
    :param weights1:
    :param biases1:
    :param weights2:
    :param biases2:
    :return:
    """
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用avg_class.average函数来计算得出变量的滑动平均值，计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)






def train(mnist):
    """
    :param mnist:训练数据
    :return:
    """
    x = tf.placeholder(tf.float32, [None,INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None,OUTPUT_NODE], name="y-output")

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 等价于上一条语句
    #biases1 = tf.get_variable("biases1",shape=[LAYER1_NODE],initializer=tf.constant_initializer(0.1))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #不使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 训练轮数
    global_step = tf.Variable(0, trainable=False)
    # 滑动平均衰减率
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失 = 交叉熵损失 + 正则化损失
    loss = cross_entropy_mean + regularization

    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,                     # 基础学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step,                            # 当前的迭代轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY                     # 学习率衰减速度
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话过程并开始训练
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # 验证
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # 迭代地训练数据
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training setp(s), validation accuracy using average model is %g" %(i,validate_acc))
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("model accuray is %g"%(test_acc))












def main(argv=None):
    mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
    print("train data size", mnist.train.num_examples)
    print("valid data size", mnist.validation.num_examples)
    print("test data size", mnist.test.num_examples)

    print("example data show :", mnist.train.images[0])
    print("example data label show :", mnist.train.labels[0])

    '''
    batch_size = 100
    xs,ys = mnist.train.next_batch(batch_size)
    print("X shape:",xs.shape)
    print("Y shape:",ys.shape)
    '''
    train(mnist)

if __name__ == "__main__":
    tf.app.run()



