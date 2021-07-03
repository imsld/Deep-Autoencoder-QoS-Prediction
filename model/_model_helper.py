import numpy as np
import tensorflow as tf


def _get_bias_initializer():
    return tf.zeros_initializer()

def _get_weight_initializer():
    return tf.random_normal_initializer(mean=0.0, stddev=0.05)

def _optimizer(MSE_loss, FLAGS):
    if FLAGS.l2_reg :
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        MSE_loss += FLAGS.lambda_ * l2_loss
        
    if (FLAGS.Optimizer == 0):
        train_optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(MSE_loss)
    
    if (FLAGS.Optimizer == 1):
        train_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(MSE_loss)
    
    if (FLAGS.Optimizer == 2):
        train_optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(MSE_loss)   
    return train_optimizer

def get_zero_mask(data):
    mask = tf.where(tf.equal(data, 0.0), tf.zeros_like(data), data)  # identify the zero values in the test ste
    num_test_labels = tf.cast(tf.count_nonzero(mask), dtype=tf.float64)  # count the number of non zero values
    bool_mask = tf.cast(mask, dtype=tf.bool)
    return bool_mask, num_test_labels 


def _appli_activation(FLAGS,op,val):
    #val = tf.layers.batch_normalization(val, training=True)
    
    #if FLAGS.sparsity == True:
    #        encode_act = 1
            
    if (op == 'encode'):
        if (FLAGS.encode_act == 0):
            return val
    
        if (FLAGS.encode_act == 1):
            return tf.nn.sigmoid(val)
    
        if (FLAGS.encode_act == 2):
            return tf.nn.tanh(val)
    
        if (FLAGS.encode_act == 3):
            return tf.nn.softmax(val)
    
        if (FLAGS.encode_act == 4):
            return tf.nn.relu(val)
    
        if (FLAGS.encode_act == 5):
            return tf.nn.softplus(val)
        
        if (FLAGS.encode_act == 6):
            return tf.nn.elu(val)
        
        if (FLAGS.encode_act == 7):
            return tf.nn.selu(val)
        
        if (FLAGS.encode_act == 8):
            return tf.nn.relu6(val)
        
        if (FLAGS.encode_act == 9):
            return tf.nn.relu(val) - 0.2 * tf.nn.relu(-val)
    
    if (op == 'decode'):
        if (FLAGS.decode_act == 0):
            return val
    
        if (FLAGS.decode_act == 1):
            return tf.nn.sigmoid(val)
    
        if (FLAGS.decode_act == 2):
            return tf.nn.tanh(val)
    
        if (FLAGS.decode_act == 3):
            return tf.nn.softmax(val)
    
        if (FLAGS.decode_act == 4):
            return tf.nn.relu(val)
    
        if (FLAGS.decode_act == 5):
            return tf.nn.softplus(val)
        
        if (FLAGS.decode_act == 6):
            return tf.nn.elu(val)
        
        if (FLAGS.decode_act == 7):
            return tf.nn.selu(val)
        
        if (FLAGS.decode_act == 8):
            return tf.nn.relu6(val)
        
        if (FLAGS.decode_act == 9):
            return tf.nn.relu(val) - 0.2 * tf.nn.relu(-val)
        
def masking_noise(X, FLAGS):
    """ Apply masking noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is forced to zero.
    :param X: array_like, Input data
    :param v: int, fraction of elements to distort
    :return: transformed data
    """
    
    corruption_ratio = np.round(FLAGS.noise * int(X.get_shape()[1])).astype(np.int)
    X_noise = X

    n_samples = X.get_shape()[0]
    n_features = X.get_shape()[1]

    for i in range(n_samples):
        mask = np.random.randint(0, n_features, corruption_ratio)

        for m in mask:
            X_noise[i][m] = 0.

    return X_noise