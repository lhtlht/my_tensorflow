import tensorflow as tf
import numpy as np
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



if __name__ == "__main__":
    main1()
    #main2()