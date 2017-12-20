import tensorflow as tf
import numpy as np


def convert(v, t=tf.float32):
    return tf.convert_to_tensor(v, dtype=t)


# declare variables
x_array = np.array([(1, 2, 3), (4, 5, 6)])
print x_array.shape

x = convert(x_array, tf.int32)
bool_tensor = convert(
    [(True, False, True), (False, False, True)], tf.bool)

# define graph
red_sum = tf.reduce_sum(x)
# axis=1 if sum all columns, axis=0 if sum all rows
red_sum_column = tf.reduce_sum(x, axis=1)
red_sum_row = tf.reduce_sum(x, axis=0)

red_min = tf.reduce_min(x)
red_min_column = tf.reduce_min(x, axis=1)

red_max = tf.reduce_max(x)
red_max_column = tf.reduce_max(x, axis=1)

red_mean = tf.reduce_mean(x)
red_mean_column = tf.reduce_mean(x, axis=1)

red_bool_all_0 = tf.reduce_all(bool_tensor)
red_bool_all = tf.reduce_all(bool_tensor, axis=1)

red_bool_any_0 = tf.reduce_any(bool_tensor)
red_bool_any = tf.reduce_any(bool_tensor, axis=1)

# run with session
with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", session.graph)

    print "Reduce sum without passed axis parameter: ", session.run(red_sum)
    print("Reduce sum with passed axis=1 (sum columns): ",
          session.run(red_sum_column))
    print("Reduce sum with passed axis=0 (sum rows): ",
          session.run(red_sum_row))

    print "Reduce min without passed axis parameter: ", session.run(red_min)
    print "Reduce min with passed axis=1: ", session.run(red_min_column)

    print "Reduce max without passed axis parameter: ", session.run(red_max)
    print "Reduce max with passed axis=1: ", session.run(red_max_column)

    print "Reduce mean without passed axis parameter: ", session.run(red_mean)
    print "Reduce mean with passed axis=1: ", session.run(red_mean_column)

    print("Reduce bool all without passed axis parameter: ",
          session.run(red_bool_all_0))
    print "Reduce bool all with passed axis=1: ", session.run(red_bool_all)

    print("Reduce bool any without passed axis parameter: ",
          session.run(red_bool_any_0))
    print "Reduce bool any with passed axis=1: ", session.run(red_bool_any)
