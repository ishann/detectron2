CFG_FILE="configs/ret/ret0013_mask_rcnn_R_50_FPN_1x.yaml"
SRC_FILE="tools/plain_train_net_custom_data.py"
NUM_GPUS=4

export CUDA_VISIBLE_DEVICES=0,1,2,7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE
