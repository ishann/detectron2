CFG_FILE="configs/inference/ret0018_rpn_R_50_FPN_1x.yaml"
SRC_FILE="tools/inference.py"
NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55265
