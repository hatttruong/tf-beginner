import tensorflow as tf
import numpy as np


# constant
TEST_DATA_SIZE = 2000
ITERATIONS = 10000
LEARNING_RATE = 0.005


def generate_test_value(data_size):
    x_train = []
    y_train = []

    for i in xrange(data_size):
        x1 = np.random.rand()
        x2 = np.random.rand()
        x3 = np.random.rand()

        y = 2 * x1 + 3 * x2 + 7 * x3 + 4
        x_train.append([x1, x2, x3])
        y_train.append(y)

    return np.array(x_train), np.transpose([y_train])


# declare variables
x = tf.placeholder(tf.float32, shape=[None, 3], name='x')
W = tf.Variable(tf.zeros([3, 1]), name='W')
b = tf.Variable(tf.zeros([1]), name='b')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')


# declare graph
model = tf.add(tf.matmul(x, W), b)

cost = tf.reduce_mean(tf.square(y - model))
train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

x_train, y_train = generate_test_value(TEST_DATA_SIZE)

init = tf.global_variables_initializer()


with tf.Session() as tf_session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", tf_session.graph)

    tf_session.run(init)

    for i in xrange(ITERATIONS):
        tf_session.run(train, feed_dict={x: x_train, y: y_train})

        if i % 100 == 0:
            print '\nITERATIONS #', i
            print 'cost = ', tf_session.run(cost,
                                            feed_dict={x: x_train, y: y_train})
            # print 'W = ', tf_session.run(W)
            # print 'b = ', tf_session.run(b)

    print '\nFINAL RESULT'
    print 'cost = ', tf_session.run(cost, feed_dict={x: x_train, y: y_train})
    print 'W = ', tf_session.run(W)
    print 'b = ', tf_session.run(b)
