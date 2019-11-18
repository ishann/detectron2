CFG_FILE="configs/ret/ret0008_lvis_retinanet_R_50_FPN_1x.yaml"
SRC_FILE="tools/train_net.py"
NUM_GPUS=8

# export CUDA_VISIBLE_DEVICES=0,1

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE
