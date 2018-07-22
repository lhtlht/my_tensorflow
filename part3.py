import tensorflow as tf
import numpy as np
def main1():
    a = tf.constant([1, 2], name="a")
    b = tf.constant([3, 4], name="b")
    print(a + b)

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
    


if __name__ == "__main__":
    #main1()
    main2()