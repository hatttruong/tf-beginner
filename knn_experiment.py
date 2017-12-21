from prepare_data_helper import *
import os
import tensorflow as tf
import numpy as np


# # create train test dataset
# file_path = os.path.join('data', 'sample_data.csv')
# TrainTestFromFile(test_size=0.33, file_path=file_path,
#                   excluded_cols=['Time'],
#                   output_dir='data',
#                   concat_features_label=False).generate_train_test()


def load_data(filepath):
    from numpy import genfromtxt

    csv_data = genfromtxt(filepath, delimiter=",", skip_header=1)
    data = []
    labels = []

    for d in csv_data:
        data.append(d[:-1])
        labels.append(d[-1])

    return np.array(data), np.array(labels)


dataset_dir = 'data'
ccf_train_data = 'train.csv'
ccf_test_data = 'test.csv'

ccf_train_filepath = os.path.join(dataset_dir, ccf_train_data)
ccf_test_filepath = os.path.join(dataset_dir, ccf_test_data)
train_dataset, train_labels = load_data(ccf_train_filepath)
test_dataset, test_labels = load_data(ccf_test_filepath)

# print train_dataset, train_labels
# print test_dataset, test_labels
print 'train_dataset.shape: ', train_dataset.shape
print 'train_labels.shape: ', train_labels.shape

print 'test_dataset.shape: ', test_dataset.shape
print 'test_labels.shape: ', test_labels.shape

# declare variables
train_pl = tf.placeholder("float", [None, 29])
test_pl = tf.placeholder("float", [29])

# graph definition
knn_prediction = tf.reduce_sum(
    tf.abs(tf.add(train_pl, tf.negative(test_pl))), axis=1)

pred = tf.argmin(knn_prediction, 0)

# session
with tf.Session() as tf_session:
    missed = 0

    # for i in xrange(len(test_dataset)):
    for i in xrange(len(test_dataset)):
        knn_index = tf_session.run(
            pred, feed_dict={train_pl: train_dataset,
                             test_pl: test_dataset[i]})

        print('%d: Predicted class %d -- True class %d' %
              (i + 1, train_labels[knn_index], test_labels[i]))

        if train_labels[knn_index] != test_labels[i]:
            missed += 1

    tf.summary.FileWriter("logs", tf_session.graph)

    print('Missed: %d -- Total: %d' % (missed, len(test_dataset)))
