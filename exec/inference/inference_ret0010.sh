CFG_FILE="configs/inference/ret0010_lvis_retinanet_R_50_FPN_1x.yaml"
SRC_FILE="tools/inference.py"
NUM_GPUS=1

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE




