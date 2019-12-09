CFG_FILE="./projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml"
SRC_FILE="./projects/TridentNet/train_net.py"
#NUM_GPUS=1
NUM_GPUS=2
#export CUDA_VISIBLE_DEVICES=7
export CUDA_VISIBLE_DEVICES=6,7

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE

