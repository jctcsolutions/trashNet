#!/usr/bin/env python
import os
import argparse
from datetime import datetime
import tensorflow as tf
import json
import glob
import numpy as np
import trainer.model
import pandas as pd
# import trainer.helper as helper
from keras.callbacks import EarlyStopping, TensorBoard, Callback, ModelCheckpoint
from keras.models import load_model
from tensorflow.python.lib.io import file_io


def initialise_hyper_params(args_parser):
    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable

    Args:
        args_parser
    """

    #data source params
    #--------------------------------------------
    args_parser.add_argument(
        '--src-data-path',
        required=True,
        type=str,
        default='dataset',
        help='Source dataset filepath.'
    )

    args_parser.add_argument(
        '--img-width-pixel',
        required=True,
        type=int,
        default=100,
        help='Width of images (after rescaling) in pixels.'
    )

    args_parser.add_argument(
        '--img-height-pixel',
        required=True,
        type=int,
        default=75,
        help='Height of images (after rescaling) in pixels.'
    )

    args_parser.add_argument(
        '--img-channels',
        required=True,
        type=int,
        default=3,
        help='Number of channels of images.'
    )

    args_parser.add_argument(
        '--test-size',
        required=True,
        type=float,
        default=0.2,
        help='Fraction of data to use for testing.'
    )

    #experiment configs
    #--------------------------------------------
    args_parser.add_argument(
        '--job-dir',
        required=True,
        default='results',
        type=str,
        help='dir-path to write checkpoints and export model'
    )

    args_parser.add_argument(
        '--train-batch-size',
        type=int,
        default=64,
        help='Batch size for training steps'
    )

    args_parser.add_argument(
        '--test-batch-size',
        type=int,
        default=64,
        help='Batch size for evaluation steps'
    )

    args_parser.add_argument(
        '--log-step-count-steps',
        type=int,
        default=1,
        help="""The frequency, in number of global steps, that the \
          global step/sec and the loss will be logged during training."""
    )

    args_parser.add_argument(
        '--save-checkpoints-secs',
        type=int,
        default=300,
        help="""Save checkpoints every this many seconds."""
    )

    args_parser.add_argument(
        '--keep-checkpoint-max',
        type=int,
        default=3,
        help=""" The maximum number of recent checkpoint files to keep.\
          As new files are created, older files are deleted. If None or \
          0, all checkpoint files are kept."""
    )

    #training configs
    #--------------------------------------------
    args_parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.003,
        help='Learning rate'
    )

    args_parser.add_argument(
        '--first-layer-size',
        type=int,
        default=1000,
        help='specifies the number of hidden unit in the first layer of the fully connected NN'
    )

    args_parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of layers in DNN'
    )

    args_parser.add_argument(
        '--layer-sizes-scale-factor',
        type=float,
        default=1,
        help="""\
          Rate of decay size of layer for Deep Neural Net.
          max(2, int(first_layer_size * scale_factor**i)) \
          """
    )

    args_parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.3,
        help='Dropout rate to use on the dense layers of the network'
    )

    args_parser.add_argument(
        '--num-epochs',
        type=int,
        default=1000,
        help='Maximum number of epochs on which to train'
    )

    args_parser.add_argument(
        '--checkpoint-epochs',
        type=int,
        default=2,
        help='Checkpoint per n training epochs'
    )

    args_parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help="""\
          Keras validation split; specifies the percentage of training
          data reserved for validation during training process 
          """
    )

    args_parser.add_argument(
        '--threshold',
        type=float,
        default=0.22,
        help='Threshold'
    )

    args_parser.add_argument(
        '--verbosity',
        choices=['DEBUG',
                 'ERROR',
                 'FATAL',
                 'INFO',
                 'WARN'],
        default='INFO'
    )

    args_parser.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""\
          Flag to decide if the model checkpoint should
          be re-used from the job-dir. If False then the
          job-dir will be deleted"""
    )

    parsed_args, unknown = args_parser.parse_known_args()

    return parsed_args



# ******************************************************************************
# THIS IS ENTRY POINT FOR THE TRAINER TASK
# ******************************************************************************
def main():

    # Set python level verbosity
    tf.logging.set_verbosity(HYPER_PARAMS.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[HYPER_PARAMS.verbosity] / 10)
    print("")
    print('Hyper-parameters:')
    print(".......................................")
    print(HYPER_PARAMS)
    print("")
    # If job_dir_reuse is False then remove the job_dir if it exists
    print("Resume training:", HYPER_PARAMS.reuse_job_dir)
    print(".......................................")
    if not HYPER_PARAMS.reuse_job_dir:
        if tf.gfile.Exists(HYPER_PARAMS.job_dir):
            tf.gfile.DeleteRecursively(HYPER_PARAMS.job_dir)
            print("Deleted job_dir {} to avoid re-use".format(HYPER_PARAMS.job_dir))
        else:
            print("No job_dir available to delete")
    else:
        print("Reusing job_dir {} if it exists".format(HYPER_PARAMS.job_dir))      
    run_config = tf.estimator.RunConfig(
        tf_random_seed=19830610,
        log_step_count_steps=HYPER_PARAMS.log_step_count_steps,
        save_checkpoints_secs=HYPER_PARAMS.save_checkpoints_secs,
        keep_checkpoint_max=HYPER_PARAMS.keep_checkpoint_max,
        model_dir=HYPER_PARAMS.job_dir # Directory to store output model and checkpoints
        # train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS) # for distributed GPU training
    )
    print("")
    print("Model Directory:", run_config.model_dir)
    print(".......................................")
    # Instantiate model object  
    print("")
    print("Instantiating model obj & loading necessary files:")
    print(".......................................")
    model = trainer.model.Model(run_config)
    # Run training and evaluation operations
    model.run_experiment()
    # Save model
    model.to_savedmodel(HYPER_PARAMS.job_dir)


args_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialise_hyper_params(args_parser)
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'



if __name__ == "__main__":
  
    main()
