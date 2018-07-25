import tensorflow as tf
import numpy as np
from numpy.random import RandomState
def main1():
    a = tf.constant([1, 2], name="a")
    b = tf.constant([3, 4], name="b")
    print(a + b)
    g = tf.Graph()
    #指定计算运行的设备
    with g.device('/gpu:0'):
        result = a + b
        print(result)
    #创建一个会话
    sess = tf.Session()
    with sess.as_default():
        print(result.eval())
        print(sess.run(result))


def main2():
    g1 = tf.Graph()
    with g1.as_default():
        #在计算图g1中定义变量“v”，并设置初始值为0
        v = tf.get_variable(
            "v",initializer=tf.zeros_initializer(),shape=[1]
        )

    g2 = tf.Graph()
    with g2.as_default():
        #在计算图g2中定义变量“v”，并设置初始值为0
        v = tf.get_variable(
            "v", initializer=tf.ones_initializer(), shape=[1]
        )

    #在计算图g1中读取变量“v”的取值
    with tf.Session(graph=g1) as sess:
        tf.initialize_all_variables().run()
        with tf.variable_scope("",reuse=True):
            #在计算图g1中， 变量“v”的取值应该取为0，所以下面会输出[0.]
            print(sess.run(tf.get_variable("v")))

            # 在计算图g1中读取变量“v”的取值
    with tf.Session(graph=g2) as sess:
        tf.initialize_all_variables().run()
        with tf.variable_scope("", reuse=True):
            # 在计算图g2中， 变量“v”的取值应该取为1，所以下面会输出[1.]
            print(sess.run(tf.get_variable("v")))

def main3():
    #声明一个（2,3）矩阵
    weights = tf.Variable(tf.random_normal([2,3],stddev=2))

def main4():
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

    x = tf.constant([[0.7,0.9]])

    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)
    sess = tf.Session()

    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    #换另一种方式，一次性初始化所有变量
    sess.run(tf.initialize_all_variables())

    print(sess.run(y))
    sess.close()


def main5():
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1))
    w2 = tf.Variable(tf.random_normal([3,1], stddev=1))
    #定义placeholder作为存放数据的地方，这里维度不一定要定义
    #如果维度是确定的，那么给出维度会降低出错的概率
    x = tf.placeholder(tf.float32,shape=(1,2),name="input")
    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    print(sess.run(y,feed_dict={x:[[0.7,0.9]]}))

def main_test():
    #定义数据batch的大小
    batch_size = 8

    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1,seed=1))
    x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
    y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")

    #定义神经网咯前向传播的过程
    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)

    #定义损失函数和反向传播的过程，
    cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))
    )
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    #通过随机数生成一个模拟数据集
    rdm = RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size,2)
    Y = [[int(x1+x2<1)] for (x1,x2) in X]

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        print(sess.run(w1))
        print(sess.run(w2))

        #训练的轮数
        STEPS = 5000
        for i in range(STEPS):
            #每次选取batch_size个样本进行训练
            start = (i*batch_size)%dataset_size
            end = min(start + batch_size,dataset_size)

            #通过选取的样本训练神经网络并更新参数
            sess.run(train_step,
                     feed_dict={x:X[start:end],y_:Y[start:end]})
            if i % 1000 == 0:
                #每隔一段时间计算在所有数据上的交叉熵并输出
                total_cross_entropy = sess.run(
                    cross_entropy,feed_dict={x:X,y_:Y}
                )
                print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))

        print(sess.run(w1))
        print(sess.run(w2))




if __name__ == "__main__":
    #main1()
    #main2()
    #main3( )
    #main4 ()
    #main5()
    main_test()