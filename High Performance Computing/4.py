import datetime
import read_inputs
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)
    return gpu_num


def model(X, reuse=False):
    with tf.variable_scope('L1', reuse=reuse):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
    
    with tf.variable_scope('L2', reuse=reuse):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
    

    with tf.variable_scope('L3', reuse=reuse):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)


        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv


if __name__ == '__main__':
    batch_size = 100
    learning_rate = 0.001
    total_epoch = 10

    gpu_num = check_available_gpus()

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])

    losses = []
    for gpu_id in range(int(gpu_num)):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                cost = tf.nn.softmax_cross_entropy_with_logits(
                                logits=model(X, gpu_id > 0),
                                labels=Y)
                losses.append(cost)

    loss = tf.reduce_mean(tf.concat(losses, axis=0))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, colocate_gradients_with_ops=True) 

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
    data = data_input[0]

    real_output = np.zeros( (np.shape(data[0][1])[0] , 10), dtype=np.float )
    for i in range ( np.shape(data[0][1])[0] ):
      real_output[i][data[0][1][i]] = 1.0  

    total_batch = int(50000/batch_size)

    idx = np.arange(0 , len(real_output))
    np.random.shuffle(idx)
    data_shuffle = np.asarray([data[0][0][i] for i in idx])
    labels_shuffle = np.asarray([real_output[i] for i in idx])

    start_time = datetime.datetime.now()

    for epoch in range(total_epoch):
        total_cost = 0

        for i in range(total_batch):
           batch_ini = 10*i
           batch_end = batch_ini+10

           batch_xs = data_shuffle[batch_ini:batch_end]
           batch_xs = batch_xs.reshape(-1, 28, 28, 1)
           batch_ys = labels_shuffle[batch_ini:batch_end]

           _, cost_val = sess.run([optimizer, loss],
                                   feed_dict={X: batch_xs,
                                              Y: batch_ys})
           total_cost += cost_val

        print("cost : %s" % total_cost)

    training_time = datetime.datetime.now() - start_time
    print("Time : {0} seconds /w {1} GPUs ---".format(training_time, gpu_num))
