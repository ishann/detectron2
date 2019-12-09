# Experiment.
CFG_FILE="projects/ConPropNet/configs/cpn_0001_ConPropNet_R_50_FPN_1x.yaml"
SRC_FILE="projects/ConPropNet/train_net.py"
NUM_GPUS=8

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55265
# OWS.
CFG_FILE="configs/ows/ret_ows05.yaml"
SRC_FILE="tools/train_net.py"
NUM_GPUS=8

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55265
