CFG_FILE="projects/ConPropNet/configs/inference/cpn_0001_ConPropNet_R_50_FPN_1x.yaml"
SRC_FILE="projects/ConPropNet/inference.py"
NUM_GPUS=1
EXP_NAME="inference_cpn0001"

export CUDA_VISIBLE_DEVICES=7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE \
                 --dist-url tcp://127.0.0.1:55265 \
                 --exp-name $EXP_NAME
