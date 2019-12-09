#!/usr/bin/env bash
CFG_FILE="configs/ows/ret_ows02.yaml"
SRC_FILE="tools/train_net.py"
NUM_GPUS=4

export CUDA_VISIBLE_DEVICE=0,1,2,3

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE
