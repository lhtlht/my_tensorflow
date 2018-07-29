import tensorflow as tf
from numpy.random import RandomState
def main1():
    batch_size = 8
    x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
    #回归问题一般只有一个输出节点
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
    w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
    y = tf.matmul(x,w1)

    loss_less = 10
    loss_more = 1
    loss = tf.reduce_sum(tf.where(tf.greater(y,y_),
                                  (y - y_) * loss_more,
                                  (y - y_) * loss_less))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    #随机生成一个模拟数据集
    rdm = RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size,2)
    Y = [ [x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1,x2) in X]

    #训练神经网络
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        STEPS = 5000
        for i in range(STEPS):
            start = (i*batch_size) % dataset_size
            end = min(start+batch_size,dataset_size)
            sess.run(train_step,
                     feed_dict={x:X[start:end], y_:Y[start:end]})
            print(sess.run(w1))

    '''
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
    a = tf.nn.relu(tf.matmul(x,w1) + biases1)
    y = tf.nn.relu(tf.matmul(a,w2) + biases2)
    #定义损失函数交叉熵
    cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))
    )
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y,y_)

    #定义均方误差
    mse = tf.reduce_mean(tf.square(y_ - y))
    '''

if __name__ == "__main__":
    main1()