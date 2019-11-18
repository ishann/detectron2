CFG_FILE="configs/trident_net/trident_net_0000_fast_R_50_C4_1x.yaml"
SRC_FILE="projects/TridentNet/train_net.py"
NUM_GPUS=2

export CUDA_VISIBLE_DEVICES=0,1

python $SRC_FILE --num-gpus $NUM_GPUS \
                 --config-file $CFG_FILE
