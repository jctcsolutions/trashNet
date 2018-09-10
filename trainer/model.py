#!/usr/bin/env python
import trainer.task
import numpy as np
import pandas as pd
import tensorflow as tf
import math
from trainer.prepare_data import prep_data

#import candidate model to use in training
from candidate_models.simpleConvNet import simpleCNN
from candidate_models.mobileNet import mobileNet

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def img_processing(img_string, width, height, channels):
    """ """
    # preprocessing of images
    image = tf.image.decode_image(img_string, channels=channels)
    image.set_shape([None, None, None])
    image = tf.image.resize_images(image, [height, width])  # resize
    image = image/255.0  # normalize pixel intensity between [0,1]
    image.set_shape([height, width, channels])
    return image


def input_fn(img_list, labels, height, width, channels,
             perform_shuffle=False, repeat_count=1, 
             batch_size=1, oneHotLabel=False):
    """TEMPLATE images input function referenced from:
    https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/
    """
    def __parse_function(img_file, label):
        # read image from file
        img_string = tf.read_file(img_file)
        image = img_processing(img_string, width, height, channels)
        if oneHotLabel:
            # one hot encode label
            n_classes = len(set(label))
            label = tf.one_hot(tf.cast(label,tf.int32), depth=n_classes)        
        d = dict(zip(['feature'], [image])), label
        return d

    # some preprocessing on the labels
    if labels is None:
        labels = [0]*len(filenames)  
    labels=np.array(labels)
    # Expand the shape of "labels" if necessory
    if (oneHotLabel==False & len(labels.shape) == 1):
        labels = np.expand_dims(labels, axis=1)
  
    # make dataset
    filenames = tf.constant(img_list)
    labels = tf.cast(tf.constant(labels), tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if perform_shuffle:
        # randomize input using a buffer length across the entire dataset 
        # this is equivalent to uniform shuffling
        dataset = dataset.shuffle(buffer_size=len(img_list))
    dataset = dataset.map(__parse_function) # Apply preprocessing procedures on feature/label
    dataset = dataset.prefetch(buffer_size=batch_size) # Pre-fetch X samples into buffer in the background
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels


class Model():

    def __init__(self, config=None):
        #specify model directory
        self.model_dir = trainer.task.HYPER_PARAMS.job_dir
        #create datasets
        self.data = prep_data(dataset_path=trainer.task.HYPER_PARAMS.src_data_path,
                              test_size=trainer.task.HYPER_PARAMS.test_size)
        #number of distinct labels
        self.n_classes = len(set(self.data['train']['label']))
        #physical names of category classes
        self.labels = ['cardboard','glass','metal','paper','plastic','trash']
        #set config
        self.config = config
        #dropout rate to use in the model
        self.dropout = trainer.task.HYPER_PARAMS.dropout_rate
        #learning rate to use with the model
        self.learning_rate = trainer.task.HYPER_PARAMS.learning_rate

  
    def __model_fn(self, features, labels, mode):
        """
        model function that declares the model architecture and the 
        training operation, prediction operation, evaluation operation
        associated with the model. All wrapped under a tf.estimator obj.
        """
        #Build the neural network graph
        #Note: because dropout layers have different behavior at training and prediction
        #time, we need to create 2 distinct computation graphs that share the same weights. 
        
        # # example of using custom-built tf graph 
        # logits_train= simpleCNN(features, self.n_classes, self.dropout, reuse=False, is_training=True)
        # logits_test = simpleCNN(features, self.n_classes, self.dropout, reuse=True, is_training=False)
        
        # example of using tf-hub mobileNet module to perform transfer learning
        logits_train= mobileNet(features, self.n_classes, self.dropout, reuse=False, is_training=True)
        logits_test = mobileNet(features, self.n_classes, self.dropout, reuse=True, is_training=False)

        #Prediction operations (where dropout is NOT applied)
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_proba = tf.nn.softmax(logits_test)

        #Define the export output spec
        label_values = tf.constant(self.labels)
        export_outputs = {'prediction': tf.estimator.export.PredictOutput({'classes': tf.gather(label_values, pred_classes),
                                                                           'scores': tf.reduce_max(pred_proba, axis=1)})}

        #if prediction mode is requested, return predictions
        if mode==tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              predictions=pred_classes, 
                                              export_outputs=export_outputs)

        #Training operations (where dropout IS applied)
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels,dtype=tf.int32), 
                                                                                logits=logits_train))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        #Define the evaluation function to evaluate the performance
        accuracy_op = tf.metrics.accuracy(labels=labels,
                                          predictions=pred_classes)

        #for train and eval mode
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=pred_classes,
                                          loss=loss_op,
                                          train_op=train_op,
                                          eval_metric_ops={'accuracy': accuracy_op})


    def run_experiment(self):
        """ """            
        # train_steps = trainer.task.HYPER_PARAMS.num_epochs*int(math.ceil(len(self.data['train']['img'])/trainer.task.HYPER_PARAMS.train_batch_size))
        self.train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(img_list=self.data['train']['img'],
                                                                           labels=self.data['train']['label'], 
                                                                           height=trainer.task.HYPER_PARAMS.img_height_pixel,
                                                                           width=trainer.task.HYPER_PARAMS.img_width_pixel,
                                                                           channels=trainer.task.HYPER_PARAMS.img_channels,
                                                                           perform_shuffle=True, 
                                                                           repeat_count=trainer.task.HYPER_PARAMS.num_epochs, 
                                                                           batch_size=trainer.task.HYPER_PARAMS.train_batch_size, 
                                                                           oneHotLabel=False),
                                                 max_steps=int(len(self.data['train']['img'])*\
                                                               trainer.task.HYPER_PARAMS.num_epochs/\
                                                               trainer.task.HYPER_PARAMS.train_batch_size)
                                                )
        self.eval_spec  = tf.estimator.EvalSpec(input_fn=lambda: input_fn(img_list=self.data['test']['img'],
                                                                          labels=self.data['test']['label'],
                                                                          height=trainer.task.HYPER_PARAMS.img_height_pixel,
                                                                          width=trainer.task.HYPER_PARAMS.img_width_pixel,
                                                                          channels=trainer.task.HYPER_PARAMS.img_channels,
                                                                          perform_shuffle=False, 
                                                                          repeat_count=1, 
                                                                          batch_size=trainer.task.HYPER_PARAMS.test_batch_size, 
                                                                          oneHotLabel=False),
                                                steps=None)
        """Run the training and evaluation operations."""
        self.model = tf.estimator.Estimator(config=self.config,
                                            model_fn=self.__model_fn) 

        tf.estimator.train_and_evaluate(self.model, self.train_spec, self.eval_spec)
        return



    def __serving_input_receiver_fn(self):
        """Convert to TensorFlow SavedModel."""
        receiver_tensors = {
            'image_filename': tf.placeholder(tf.string,shape=[])
        }
        width = trainer.task.HYPER_PARAMS.img_width_pixel
        height = trainer.task.HYPER_PARAMS.img_height_pixel
        channels = trainer.task.HYPER_PARAMS.img_channels
        features = {
            'feature': tf.map_fn(lambda x: img_processing(x, width, height, channels), 
                                 elems=tf.reshape(tf.read_file(receiver_tensors['image_filename']),(1,)),
                                 dtype=tf.float32)
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,
                                                        features=features)

        

    def to_savedmodel(self, export_path):
        """ """
        #export model to path dir
        self.model.export_savedmodel(export_dir_base=export_path, 
                                     serving_input_receiver_fn=self.__serving_input_receiver_fn)

