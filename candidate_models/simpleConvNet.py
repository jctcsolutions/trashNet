import tensorflow as tf


def simpleCNN(x_dict, n_classes, dropout, reuse, is_training):

    # define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF estimator input (dict), note due to our pre-processing, our input is 
        # already in 4-D array format of [batch, width, height, channel] format
        x = x_dict['feature']

        # convolution layer # 1
        conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, strides=(1,1), 
                               padding='same', data_format='channels_last', activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=(2,2), padding='same')

        # convolution layer # 2
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=3, strides=(1,1), 
                               padding='same', data_format='channels_last', activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=(2,2), padding='same')

        # convolution layer # 3
        conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=3, strides=(1,1), 
                               padding='same', data_format='channels_last', activation=tf.nn.relu)
        conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=(2,2), padding='same')

        # flatten layer
        flatten = tf.contrib.layers.flatten(conv3)

        # fully connected layers
        fc1 = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)
        fc1 = tf.layers.dropout(inputs=fc1, rate=dropout, training=is_training)
        fc2 = tf.layers.dense(inputs=fc1, units=512, activation=tf.nn.relu)
        fc2 = tf.layers.dropout(inputs=fc2, rate=dropout, training=is_training)
        fc3 = tf.layers.dense(inputs=fc2, units=256, activation=tf.nn.relu)
        fc3 = tf.layers.dropout(inputs=fc3, rate=dropout, training=is_training)        
        # output layer
        pred = tf.layers.dense(inputs=fc1, units=n_classes, activation=None)

    return pred