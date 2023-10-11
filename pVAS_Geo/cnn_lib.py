"""
This module contains methods for building the CNN.
"""
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


#weight initialization
def weight_variable(shape, 
              v_name):
  """
    Weight variable wrapper.
    
    Parameters
    ---------- 
      shape: list
        list of sizes
      v_name: str
			variable name    
    
    Returns 
    --------
     tf variable of the given shape
      
    """   
    
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=v_name)

def bias_variable(shape, 
            v_name):
    """
    Bias variable wrapper.
    
    Parameters
    ---------- 
      shape: list
        list of sizes
      v_name: str
			variable name    
    
    Returns 
    --------
     tf variable of the given shape
      
    """ 
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=v_name)


def tf_variable(shape, 
            value, 
            v_name):
    """
    TensorFlow variable.
    
    Parameters
    ---------- 
      shape: list
        list of sizes
      value: `Tensor`   
        value to be assigned
      v_name: str
			variable name    
    
    Returns 
    --------
     tf variable of the given shape and value
      
    """  
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=v_name)



def conv2d(x, 
         W, 
         v_name):
    """
    Convolutional layer wrapper.
    
    Parameters
    ---------- 
      x: `Tensor`
        input tensor
      W: `Tensor`   
        weights
      v_name: str
			variable name    
    
    Returns 
    --------
    convolutional layer 
      
    """
    
    return tf.nn.conv2d(x, 
                W, 
                strides=[1, 1, 1, 1], 
                padding='SAME', 
                name=v_name)


def max_pool_2x2(x,
             v_name):
  
    """
    Max pool 2x2 wrapper.
    
    Parameters
    ---------- 
      x: `Tensor`
        input tensor.
      v_name: str
			variable name    
    
    Returns 
    --------
     max pool layer 
      
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], 
                padding='SAME', 
                name=v_name)

			

def batch_norm_wrapper(x, 
                 phase, 
                 scope):
    """
    Batch normalisation wrapper.
    
    Parameters
    ---------- 
      x: `Tensor`
        input tensor.
      phase: tf.bool
        1 for training, 0 for test
      scope: variable scope    
    
    Returns 
    --------
      h: tf layer
        layer with normalised weigths
      
    """  
    with tf.variable_scope(scope):
      h = tf.contrib.layers.batch_norm(x, 
                             center=True, 
                             scale=True, 
                             is_training=phase, 
                             scope='bn')
    
    return h


def build_VGG_network_raw(x, 
            keep_prob1,
            keep_prob2, 
            phase, 
            input_shape=[640, 512]):
    """
    Builds a convolutional neural network with x as input tensor. The architecture
    is VGG inspired. 
    
    Parameters
    ---------- 
      x: `Tensor`
        input tensor.
      keep_prob1: float
        dropout coefficient for the first FC layer.
      keep_prob1: float
        dropout coefficient for the second FC layer.
      phase: tf.bool
        1 for training, 0 for test
      input_shape: list of 2 elements, optional
        shape of the input image; both parameters must be multiple of 64; 
        default: [640, 512]
    
    Returns
    --------
      y_conv: output tensor
     """   
    
    # Reshape the convolutional layer
    x_image = tf.reshape(x, [-1,input_shape[0],input_shape[1],1])
    
    # Convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32], 'w_conv_1')
    b_conv1 = bias_variable([32], 'w_bias_1')
    
      
    # Apply the convolution with batch normalization
    BN1 = batch_norm_wrapper(conv2d(x_image, W_conv1, 'conv_1') + b_conv1, phase, 'bn1')
    h_conv1 = tf.nn.relu(BN1, name='h_conv_1')
    
    # Convolutional layer
    W_conv2 = weight_variable([3, 3, 32, 32], 'w_conv_2')
    b_conv2 = bias_variable([32], 'w_bias_2')

    # Apply the convolution with batch normalization
    BN2 = batch_norm_wrapper(conv2d(h_conv1, W_conv2, 'conv_2') + b_conv2, phase, 'bn2')
    h_conv2 = tf.nn.relu(BN2, name='h_conv_2')
    
    # 1st max pooling layer 
    h_pool1 = max_pool_2x2(h_conv2, 'h_pool_1')
    

    # Convolutional layer
    W_conv3 = weight_variable([3, 3, 32, 32], 'w_conv_3')
    b_conv3 = bias_variable([32], 'w_bias_3')
    
    # Apply the convolution with batch normalization
    BN3 = batch_norm_wrapper(conv2d(h_pool1, W_conv3, 'conv_3') + b_conv3, phase, 'bn3')
    h_conv3 = tf.nn.relu(BN3, name='h_conv_3')
    
    
    # Convolutional layer
    W_conv4 = weight_variable([3, 3, 32, 32], 'w_conv_4')
    b_conv4 = bias_variable([32], 'w_bias_4')
    
    # Apply the convolution with batch normalization
    BN4 = batch_norm_wrapper(conv2d(h_conv3, W_conv4, 'conv_4') + b_conv4, phase, 'bn4')
    h_conv4 = tf.nn.relu(BN4, name='h_conv_4')
    
    # Max pooling layer 
    h_pool4 = max_pool_2x2(h_conv4, 'h_pool_2')
    
    
    # Convolutional layer
    W_conv5 = weight_variable([3, 3, 32, 32], 'w_conv_5')
    b_conv5 = bias_variable([32], 'w_bias_5')
    #apply the 3nd convolution
    
    BN5 = batch_norm_wrapper(conv2d(h_pool4, W_conv5, 'conv_5') + b_conv5, phase, 'bn5')
    h_conv5 = tf.nn.relu(BN5, name='h_conv_5')
    
    
    #fourth convolutional layer
    W_conv6 = weight_variable([3, 3, 32, 32], 'w_conv_6')
    b_conv6 = bias_variable([32], 'w_bias_6')
    #apply the 4th convolution
    
    BN6 = batch_norm_wrapper(conv2d(h_conv5, W_conv6, 'conv_6') + b_conv6, phase, 'bn6')
    h_conv6 = tf.nn.relu(BN6, name='h_conv_6')
    
    #2nd max pooling layer
    h_pool3 = max_pool_2x2(h_conv6, 'h_pool_3')
    
    
    #third convolutional layer
    W_conv7 = weight_variable([3, 3, 32, 64], 'w_conv_7')
    b_conv7 = bias_variable([64], 'w_bias_7')
    #apply the 3nd convolution
    
    BN7 = batch_norm_wrapper(conv2d(h_pool3, W_conv7, 'conv_7') + b_conv7, phase, 'bn7')
    h_conv7 = tf.nn.relu(BN7, name='h_conv_7')
    
    
    #fourth convolutional layer
    W_conv8 = weight_variable([3, 3, 64, 64], 'w_conv_8')
    b_conv8 = bias_variable([64], 'w_bias_8')
    #apply the 4th convolution
    
    BN8 = batch_norm_wrapper(conv2d(h_conv7, W_conv8, 'conv_8') + b_conv8, phase, 'bn8')
    h_conv8 = tf.nn.relu(BN8, name='h_conv_8')
    
    
    h_pool4 = max_pool_2x2(h_conv8, 'h_pool_4')
    
    #convolutional layer
    W_conv9 = weight_variable([3, 3, 64, 64], 'w_conv_9')
    b_conv9 = bias_variable([64], 'w_bias_9')
    #apply the 3nd convolution
    
    BN9 = batch_norm_wrapper(conv2d(h_pool4, W_conv9, 'conv_9') + b_conv9, phase, 'bn9')
    h_conv9 = tf.nn.relu(BN9, name='h_conv_9')
    
    
    #fourth convolutional layer
    W_conv10 = weight_variable([3, 3, 64, 64], 'w_conv_10')
    b_conv10 = bias_variable([64], 'w_bias_10')
    #apply the 4th convolution
    
    BN10 = batch_norm_wrapper(conv2d(h_conv9, W_conv10, 'conv_10') + b_conv10, phase, 'bn10')
    h_conv10 = tf.nn.relu(BN10, name='h_conv_10')
    
    
    h_pool5 = max_pool_2x2(h_conv10, 'h_pool_5')
    
    
    #third convolutional layer
    W_conv11 = weight_variable([3, 3, 64, 128], 'w_conv_11')
    b_conv11 = bias_variable([128], 'b_conv_11')
    #apply the 3nd convolution
    
    BN11 = batch_norm_wrapper(conv2d(h_pool5, W_conv11, 'conv_11') + b_conv11, phase, 'bn11')
    h_conv11 = tf.nn.relu(BN11, name='h_conv_11')
    
    
    #fourth convolutional layer
    W_conv12 = weight_variable([3, 3, 128, 128], 'w_conv_12')
    b_conv12 = bias_variable([128], 'b_conv_12')
    #apply the 4th convolution
    
    BN12 = batch_norm_wrapper(conv2d(h_conv11, W_conv12, 'conv_12') + b_conv12, phase, 'bn12')
    h_conv12 = tf.nn.relu(BN12, name='h_conv_12')
    
    
    h_pool6 = max_pool_2x2(h_conv12, 'h_pool_6')
   
    #1st densely connected layer
    W_fc1 = weight_variable([input_shape[0] * input_shape[1] *128 // (64*64), 512], 'w_fc_1') #here 1/64 of x_image layer and conv 6 maps 
    b_fc1 = bias_variable([512], 'b_fc_1')
    
    h_pool6_flat = tf.reshape(h_pool6, [-1, input_shape[0] * input_shape[1]*128 // (64*64)], name='h_pool_6_flat')
    BN13 = batch_norm_wrapper(tf.matmul(h_pool6_flat, W_fc1) + b_fc1, phase, 'bn13')
    h_fc1 = tf.nn.relu(BN13, name='h_fc_1')
    
    #dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1, name='h_fc_drop_1')
    
    
    
    #second densely connected layer
    W_fc2 = weight_variable([512, 512], 'w_conv_14')  
    b_fc2 = bias_variable([512], 'b_conv_14')
    
    BN14 = batch_norm_wrapper(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, phase, 'bn14')
    h_fc2 = tf.nn.relu(BN14, name='h_fc_2')
    
    #dropout
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2, name='h_fc_drop_2')
    
    
    #readout layer
    W_fc3 = weight_variable([512, 1], 'w_fc_3')
    b_fc3 = bias_variable([1], 'b_fc_3')
    
    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    
    return y_conv



def build_VGG_network_processed(x, 
            keep_prob1,
            keep_prob2, 
            phase, 
            input_shape=[640, 512]):
    """
    Builds a convolutional neural network with x as input tensor. The architecture
    is VGG inspired. 
    
    Parameters
    ---------- 
      x: `Tensor`
        input tensor.
      keep_prob1: float
        dropout coefficient for the first FC layer.
      keep_prob1: float
        dropout coefficient for the second FC layer.
      phase: tf.bool
        1 for training, 0 for test
      input_shape: list of 2 elements, optional
        shape of the input image; both parameters must be multiple of 64; 
        default: [640, 512]
    
    Returns
    --------
      y_conv: output tensor
     """   
    
    #reshape the convolutional layer
    x_image = tf.reshape(x, [-1,input_shape[0],input_shape[1],1])
    
    #first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32], 'w_conv_1')
    b_conv1 = bias_variable([32], 'w_bias_1')
  
      
    #apply the 1st convolution with batch normalization
    BN1 = batch_norm_wrapper(conv2d(x_image, W_conv1, 'conv_1') + b_conv1, phase, 'bn1')
    h_conv1 = tf.nn.relu(BN1, name='h_conv_1')
    
    #first convolutional layer
    W_conv1_2 = weight_variable([3, 3, 32, 32], 'w_conv_2')
    b_conv1_2 = bias_variable([32], 'w_bias_1')
    
      
    #apply the 1st convolution with batch normalization
    BN1_2 = batch_norm_wrapper(conv2d(h_conv1, W_conv1_2, 'conv_2') + b_conv1_2, phase, 'bn2')
    h_conv1_2 = tf.nn.relu(BN1_2, name='h_conv_2')
  
    #1st max pooling layer 
    h_pool0 = max_pool_2x2(h_conv1_2, 'h_pool_1')
    
    
    #second convolutional layer
    W_conv0_2 = weight_variable([3, 3, 32, 32], 'w_conv_3')
    b_conv0_2 = bias_variable([32], 'w_bias_1')
  
    #apply the 2nd convolution
    
    BN0_2 = batch_norm_wrapper(conv2d(h_pool0, W_conv0_2, 'conv_3') + b_conv0_2, phase, 'bn3')
    h_conv0_2 = tf.nn.relu(BN0_2, name='h_conv_3')
  
  
    #second convolutional layer
    W_conv2 = weight_variable([3, 3, 32, 32], 'w_conv_4')
    b_conv2 = bias_variable([32], 'w_bias_1')
    
    #apply the 2nd convolution
    BN2 = batch_norm_wrapper(conv2d(h_conv0_2, W_conv2, 'conv_4') + b_conv2, phase, 'bn4')
    h_conv2 = tf.nn.relu(BN2, name='h_conv_4')
    
    
    #1st max pooling layer 
    h_pool1 = max_pool_2x2(h_conv2, 'h_pool_2')
    
    
    #third convolutional layer
    W_conv3 = weight_variable([3, 3, 32, 32], 'w_conv_5')
    b_conv3 = bias_variable([32], 'w_bias_1')
    #apply the 3nd convolution
    
    BN3 = batch_norm_wrapper(conv2d(h_pool1, W_conv3, 'conv_5') + b_conv3, phase, 'bn5')
    h_conv3 = tf.nn.relu(BN3, name='h_conv_5')
    
    
    #fourth convolutional layer
    W_conv4 = weight_variable([3, 3, 32, 32], 'w_conv_6')
    b_conv4 = bias_variable([32], 'w_bias_1')
    #apply the 4th convolution
    
    BN4 = batch_norm_wrapper(conv2d(h_conv3, W_conv4, 'conv_6') + b_conv4, phase, 'bn6')
    h_conv4 = tf.nn.relu(BN4, name='h_conv_6')
    
    #2nd max pooling layer
    h_pool2 = max_pool_2x2(h_conv4, 'h_pool_3')
  
  
    #third convolutional layer
    W_conv5 = weight_variable([3, 3, 32, 64], 'w_conv_7')
    b_conv5 = bias_variable([64], 'w_bias_1')
    #apply the 3nd convolution
    
    BN5 = batch_norm_wrapper(conv2d(h_pool2, W_conv5, 'conv_7') + b_conv5, phase, 'bn7')
    h_conv5 = tf.nn.relu(BN5, name='h_conv_7')
    
    
    #fourth convolutional layer
    W_conv6 = weight_variable([3, 3, 64, 64], 'w_conv_8')
    b_conv6 = bias_variable([64], 'w_bias_1')
    #apply the 4th convolution
    
    BN6 = batch_norm_wrapper(conv2d(h_conv5, W_conv6, 'conv_8') + b_conv6, phase, 'bn8')
    h_conv6 = tf.nn.relu(BN6, name='h_conv_8')
    
    
    h_pool3 = max_pool_2x2(h_conv6, 'h_pool_4')
  
  
  
    #third convolutional layer
    W_conv7 = weight_variable([3, 3, 64, 64], 'w_conv_9')
    b_conv7 = bias_variable([64], 'w_bias_1')
    #apply the 3nd convolution
    
    BN7 = batch_norm_wrapper(conv2d(h_pool3, W_conv7, 'conv_9') + b_conv7, phase, 'bn9')
    h_conv7 = tf.nn.relu(BN7, name='h_conv_9')
    
    
    #fourth convolutional layer
    W_conv8 = weight_variable([3, 3, 64, 64], 'w_conv_10')
    b_conv8 = bias_variable([64], 'w_bias_1')
    #apply the 4th convolution
    
    BN8 = batch_norm_wrapper(conv2d(h_conv7, W_conv8, 'conv_10') + b_conv8, phase, 'bn10')
    h_conv8 = tf.nn.relu(BN8, name='h_conv_10')
    
    
    h_pool4 = max_pool_2x2(h_conv8, 'h_pool_5')
      
    
    #third convolutional layer
    W_conv9 = weight_variable([3, 3, 64, 128], 'w_conv_11')
    b_conv9 = bias_variable([128], 'b_conv_1')
    #apply the 3nd convolution
      
    BN9 = batch_norm_wrapper(conv2d(h_pool4, W_conv9, 'conv_11') + b_conv9, phase, 'bn11')
    h_conv9 = tf.nn.relu(BN9, name='h_conv_11')
      
      
    #fourth convolutional layer
    W_conv10 = weight_variable([3, 3, 128, 128], 'w_conv_12')
    b_conv10 = bias_variable([128], 'b_conv_12')
    #apply the 4th convolution
      
    BN10 = batch_norm_wrapper(conv2d(h_conv9, W_conv10, 'conv_12') + b_conv10, phase, 'bn12')
    h_conv10 = tf.nn.relu(BN10, name='h_conv_12')
      
      
    h_pool5 = max_pool_2x2(h_conv10, 'h_pool_6')
      
    
    #1st densely connected layer
    W_fc1 = weight_variable([input_shape[0] * input_shape[1] *128 // (64*64), 1024], 'w_fc_1') #here 1/64 of x_image layer and conv 6 maps 
    b_fc1 = bias_variable([1024], 'b_fc_1')
      
    h_pool5_flat = tf.reshape(h_pool5, [-1, input_shape[0] * input_shape[1]*128 // (64*64)], name='h_pool_6_flat')
    BN13 = batch_norm_wrapper(tf.matmul(h_pool5_flat, W_fc1) + b_fc1, phase, 'bn13')
    h_fc1 = tf.nn.relu(BN13, name='h_fc_1')
      
    #dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1, name='h_fc_drop_1')
    
      
      
    #second densely connected layer
    W_fc2 = weight_variable([1024, 1024], 'w_conv_14')  
    b_fc2 = bias_variable([1024], 'b_conv_14')
      
    BN14 = batch_norm_wrapper(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, phase, 'bn14')
    h_fc2 = tf.nn.relu(BN14, name='h_fc_2')
      
    #dropout
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2, name='h_fc_drop_2')
      
      
    #readout layer
    W_fc3 = weight_variable([1024, 1], 'w_fc_3')
    b_fc3 = bias_variable([1], 'b_fc_3')
      
    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    
    return y_conv
