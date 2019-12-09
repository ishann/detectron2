CFG_FILE="configs/inference/ret0020b_mask_rcnn_R_50_FPN_1x.yaml"
SRC_FILE="tools/inference.py"
NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=5

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE




