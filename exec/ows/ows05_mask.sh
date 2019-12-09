#!/usr/bin/env bash
CFG_FILE="configs/ows/ret_ows05.yaml"
SRC_FILE="tools/train_net.py"
NUM_GPUS=8

# export CUDA_VISIBLE_DEVICES=3,4,5,6

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55265

