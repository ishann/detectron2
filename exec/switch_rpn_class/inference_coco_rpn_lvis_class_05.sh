CFG_FILE="configs/switch_rpn_class/lvis_mask_rcnn_R_50_FPN_1x_0_05.yaml"
SRC_FILE="./tools/inference.py"
NUM_GPUS=1

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE




