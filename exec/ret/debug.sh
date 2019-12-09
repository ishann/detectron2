CFG_FILE="configs/ret/debug_rpn_R_50_FPN_1x.yaml"
SRC_FILE="tools/train_net.py"
# NUM_GPUS=1
NUM_GPUS=2

# export CUDA_VISIBLE_DEVICES=7
export CUDA_VISIBLE_DEVICES=6,7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55264
