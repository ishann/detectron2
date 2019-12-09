CFG_FILE="projects/ConPropNet/configs/inference/cpn_0000_ConPropNet_R_50_FPN_1x.yaml"
SRC_FILE="projects/ConPropNet/inference.py"
NUM_GPUS=1
EXP_NAME="inference_cpn0000"

export CUDA_VISIBLE_DEVICES=6

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55264 \
                 --exp-name $EXP_NAME
