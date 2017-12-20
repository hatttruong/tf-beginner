import tensorflow as tf
import numpy as np


def convert(v, t=tf.float32):
    return tf.convert_to_tensor(v, dtype=t)


# declare variables
x = convert(np.array([
    [2, 2, 1, 3],
    [4, 5, 6, -1],
    [0, 1, 1, -2]
]))

y = convert(np.array([1, 2, 5, 3, 7, 5]))
z = convert(np.array([1, 0, 4, 6, 2]))

# define graph
min_tensor = tf.argmin(input=x, axis=1)
max_tensor = tf.argmax(input=x, axis=1)
diff_tensor = tf.setdiff1d(x=z, y=y)
unique_tensor = tf.unique(y)

# run with session
with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", session.graph)

    print "\nIndex of min value in X", session.run(min_tensor)
    print "\nIndex of max value in X", session.run(max_tensor)
    print "\nDifference between y & z (values that are in z but not in y)"
    print 'Exptect (value): 0, 4, 6'
    print 'Actual (value):', session.run(diff_tensor)[0]
    print 'Exptect (index): 1, 2, 3'
    print 'Actual (index):', session.run(diff_tensor)[1]

    print '\nUnique value in Y = ', session.run(unique_tensor)[0]
    print '\nUnique index in Y = ', session.run(unique_tensor)[1]
