#!/usr/bin/env bash
CFG_FILE="configs/ows/ret_ows00.yaml"
SRC_FILE="tools/train_net.py"
NUM_GPUS=8
# PORT=(2**15 + 2**14 + hash(i) % 2 ** 14)

# export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55264

