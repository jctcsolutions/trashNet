#!/bin/bash
echo "Training ML model locally, sourcing dataset locally"

PACKAGE_PATH=trainer
VERSION="v1"
MODEL_NAME=trashNet_${VERSION}
MODEL_DIR=trained_models/${MODEL_NAME}
SRC_DATA_PATH=dataset
IMG_WIDTH_PIXEL=160
IMG_HEIGHT_PIXEL=160
IMG_CHANNELS=3
TEST_SIZE=0.2


gcloud ml-engine local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        --job-dir=${MODEL_DIR} \
        -- \
        --src-data-path=${SRC_DATA_PATH} \
        --img-width-pixel=${IMG_WIDTH_PIXEL} \
        --img-height-pixel=${IMG_HEIGHT_PIXEL} \
        --img-channels=${IMG_CHANNELS} \
        --test-size=${TEST_SIZE} \
        --train-batch-size=128 \
        --test-batch-size=128 \
        --log-step-count-steps=1 \
        --save-checkpoints-secs=300 \
        --keep-checkpoint-max=3 \
        --learning-rate=0.0001 \
        --first-layer-size=100 \
        --num-layers=4 \
        --layer-sizes-scale-factor=0.9 \
        --dropout-rate=0.7 \
        --num-epochs=10 \
        --verbosity="INFO"
        # --reuse-job-dir


# ls ${MODEL_DIR}/export/estimator
# MODEL_LOCATION=${MODEL_DIR}/export/estimator/$(ls ${MODEL_DIR}/export/estimator | tail -1)
# echo ${MODEL_LOCATION}
# ls ${MODEL_LOCATION}

# # invoke trained model to make prediction given new data instances
# gcloud ml-engine local predict --model-dir=${MODEL_LOCATION} --json-instances=data/new-data.json

