import tensorflow as tf
import numpy as np


def convert(v, t=tf.float32):
    return tf.convert_to_tensor(v, dtype=t)


# declaration
m1 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m2 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m3 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m4 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m5 = convert(np.array(np.random.rand(4, 4), dtype='float32'))


# graph definition
m_transpose = tf.transpose(m1)
m_mul = tf.matmul(m1, m2)
m_det = tf.matrix_determinant(m3)
m_inv = tf.matrix_inverse(m4)
m_solve = tf.matrix_solve(m5, [[1], [1], [1], [1]])


# run with session
with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", session.graph)

    print session.run(m_transpose)
    print session.run(m_mul)
    print session.run(m_det)
    print session.run(m_inv)
    print session.run(m_solve)
