# Experiment.
EXP_NAME="cpn_dbug"
CFG_FILE="projects/ConPropNet/configs/cpn_dbug_ConPropNet_R_50_FPN_1x.yaml"
SRC_FILE="projects/ConPropNet/train_net.py"
NUM_GPUS=1
# NUM_GPUS=2

export CUDA_VISIBLE_DEVICES=7
# export CUDA_VISIBLE_DEVICES=6,7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55264 \
                 --exp-name $EXP_NAME
