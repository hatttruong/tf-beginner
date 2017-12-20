# tf-beginner
I follow this tutorial to practice with TF https://www.kdnuggets.com/2017/12/getting-started-tensorflow.html

## Install packages

Open Terminal and direct to folder containing source code:
`pip install -r requirements.txt`


## Install TensorFlow
- create virtual environment. Note that to make it easy, we install **biopython** here instead of just **NumPy**. This includes **NumPy** and a few other packages that we will be needing
> `conda create --name TensorflowEnv biopython`
> `source activate TensorFlowEnv`

- To activate this environment, use:
> source activate TensorflowEnv

- To deactivate this environment, use:
> source deactivate TensorflowEnv

- install TensorFlow:
> `pip install tensorflow`


## Simple Expression
Let’s start with expression `y = 5*x + 13` in TensorFlow fashion.

### 1. Constants: 
In TensorFlow, constants are created using the function constant, which has the signature:
`constant(value, dtype=None, shape=None, name='Const', verify_shape=False)`

- In which, *shape, name, and verify_shape* are optional. dtype may be float32/64, int8/16, etc.

- If you need constants with specific values inside your training model, then the constant object can be used as in following example:
    `z = tf.constant(5.2, name="x", dtype=tf.float32)`

### 2. Variables
Variables in TensorFlow are in-memory buffers containing tensors which have to be explicitly initialized and used in-graph to maintain state across session. By simply calling the constructor the variable is added in computational graph.

Variables are especially useful once you start with training models, and they are used to hold and update parameters. An initial value passed as an argument of a constructor represents a tensor or object which can be converted or returned as a tensor. That means if we want to fill a variable with some predefined or random values to be used afterwards in the training process and updated over iterations, we can define it in the following way:
    `k = tf.Variable(tf.zeros([1]), name="k")`

Another way to use variables in TensorFlow is in calculations where that variable isn’t trainable and can be defined in the following way:
    `k = tf.Variable(tf.add(a, b), trainable=False)`

### 3. Sessions
In order to actually evaluate the nodes, we must run a computational graph within a session.

A session encapsulates the control and state of the TensorFlow runtime. A session without parameters will use the default graph created in the current session, otherwise the session class accepts a graph parameter, which is used in that session to be executed.

Try to run simple_expression.py: `python simple_expression.py`

### 4. Defining Computational Graphs
The good thing about working with dataflow graphs is that the execution model is separated from its execution (on CPU, GPU, or some combination) where, once implemented, software in TensorFlow can be used on the CPU or GPU where all complexity related to code execution is hidden.

The computation graph is a built-in process that uses the library without needing to call the graph object directly. A graph object in TensorFlow, which contains a set of operations and tensors as units of data, is used between operations which allows the same process and contains more than one graph where each graph will be assigned to a different session. 

TensorFlow also provides a feed mechanism for patching a tensor to any operation in the graph, where the feed replaces the output of an operation with the tensor value. The feed data are passed as an argument in the run() function call

A placeholder is TensorFlow’s way of allowing developers to inject data into the computation graph through placeholders which are bound inside some expressions. The signature of the placeholder is:

`placeholder(dtype, shape=None, name=None)`

### 5. TensorBoard
TensorBoard is a visualization tool for analyzing data flow graphs. This can be useful for gaining better understanding of machine learning models.

`pip install tensorboard`

In order to log events from session which later can be used in TensorBoard, TensorFlow provides the **FileWriter** class. It can be used to create an event file for storing summaries and events where the constructor accepts six parameters and looks like (where the **logdir** parameter is required, and others have default values):

`__init__(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None, filename_suffix=None)`

Add these two lines into simple_expression.py:
`merged = tf.summary.merge_all()`
`writer = tf.summary.FileWriter("logs", session.graph)`

Run tensorboard:
`tensorboard --logdir logs/`

## Matrix Operations

Try to run matrix_operations.py: `python simple_expression.py`

## Transforming Data

### 1. Reduction
 Reduction is an operation that removes one or more dimensions from a tensor by performing certain operations across those dimensions.

Have a look and run reduce_data.py: `python reduce_data.py`

### 2. Segmentation
[segmentation api](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/segmentation)
Here a segmentation is a partitioning of a tensor along the first dimension, i.e. it defines a mapping from the first dimension onto segment_ids. The segment_ids tensor should be the size of the first dimension, **d0**, with consecutive IDs in the range 0 to k, where k &lt; d0. In particular, a segmentation of a matrix tensor is a mapping of rows to segments.
`tf.segment_sum(data, segment_ids, name=None)`
![](https://www.tensorflow.org/versions/r0.12/images/SegmentSum.png)

Have a look and run segmentation.py: `python segmentation.py`

### 3. Sequence Utilities:
Sequence utilities include methods such as:

- **argmin** function `tf.argmin(input, axis=None, name=None, dimension=None)`, which returns the **index** with min value across the axes of the input tensor
- **argmax** function `tf.argmax(input, axis=None, name=None, dimension=None)`, which returns the **index** with max value across the axes of the input tensor
- **setdiff1d** function `tf.setdiff1d(x, y, index_dtype=tf.int32, name=None)`, which returns *a list out* that represents all values that are in x but not in y.
- **where** function, which will return elements either from two passed elements x or y, which depends on the passed condition, or
- **unique** function `tf.unique(x, out_idx=None, name=None)`, which will return unique elements in a 1-D tensor.

Have a look and run sequence_utilities.py: `python sequence_utilities.py`

## Machine Learning with TensorFlow

### 1. kNN

### 2. Linear Regression
