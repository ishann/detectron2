# Experiment.
CFG_FILE="configs/ret/ret0022_fast_rcnn_R_50_FPN_1x.yaml"
SRC_FILE="tools/train_net.py"
NUM_GPUS=8

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE

# OWS.
CFG_FILE="configs/ows/ret_ows05.yaml"
SRC_FILE="tools/train_net.py"
NUM_GPUS=8

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE
