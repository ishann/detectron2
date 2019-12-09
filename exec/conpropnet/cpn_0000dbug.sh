# Experiment.
CFG_FILE="projects/ConPropNet/configs/cpn_0000dbug_ConPropNet_R_50_FPN_1x.yaml"
SRC_FILE="projects/ConPropNet/train_net.py"
EXP_NAME="cpn_dbug"
NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55264 \
                 --exp-name $EXP_NAME
