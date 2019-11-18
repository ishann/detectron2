CFG_FILE="configs/ret/ret_ows01.yaml"
SRC_FILE="tools/train_net.py"
NUM_GPUS=4

export CUDA_VISIBLE_DEVICES=4,5,6,7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55265
