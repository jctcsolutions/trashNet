import tensorflow_hub as hub
import tensorflow as tf


def relu6(x):
    return K.relu(x, max_value=6)


def mobileNet(x_dict, n_classes, dropout, reuse, is_training):

    # transfer learning using tf-hub modules
    module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/2")
    height, width = hub.get_expected_image_size(module)
    features = module(x_dict['feature'])  # Features with shape [batch_size, num_features].

    # define a scope for reusing the variables
    with tf.variable_scope('my_mobileNet', reuse=reuse):
        
        # fully connected layers
        fc1 = tf.layers.dense(inputs=features, units=1280, activation=tf.nn.relu)
        fc1 = tf.layers.dropout(inputs=fc1, rate=dropout, training=is_training)
        fc2 = tf.layers.dense(inputs=fc1, units=640, activation=tf.nn.relu)
        fc2 = tf.layers.dropout(inputs=fc2, rate=dropout, training=is_training)
        fc3 = tf.layers.dense(inputs=fc2, units=320, activation=tf.nn.relu)
        fc3 = tf.layers.dropout(inputs=fc3, rate=dropout, training=is_training)        
        # output layer
        pred = tf.layers.dense(inputs=fc1, units=n_classes, activation=None)

    return pred
