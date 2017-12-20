import tensorflow as tf

x = tf.placeholder(tf.float32, name="x")
y = tf.placeholder(tf.float32, name="y")

z = tf.multiply(x, y, name="z")

with tf.Session() as session:
    print session.run(z, feed_dict={x: 2.1, y: 3.0})
