import tensorflow as tf
import numpy as np


def convert(v, t=tf.float32):
    return tf.convert_to_tensor(v, dtype=t)


# declare variables
sed_ids = tf.constant([0, 0, 1, 2, 2])
tens1 = convert(np.array([(2, 5, 3, -5),
                          (0, 3, -2, 5),
                          (4, 3, 5, 3),
                          (6, 1, 4, 0),
                          (6, 1, 4, 0)]), tf.int32)
tens2 = convert(np.array([1, 2, 3, 4, 5]), tf.int32)

# define graph
seg_sum_1 = tf.segment_sum(tens1, sed_ids)
seg_sum_2 = tf.segment_sum(tens2, sed_ids)

# run with session
with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", session.graph)

    print "Segmentation sum 1", session.run(seg_sum_1)
    print "Segmentation sum 2", session.run(seg_sum_2)
