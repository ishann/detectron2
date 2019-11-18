CFG_FILE="configs/ret/ret0009_lvis_retinanet_R_50_FPN_1x.yaml"
SRC_FILE="tools/train_net.py"
NUM_GPUS=7

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE
