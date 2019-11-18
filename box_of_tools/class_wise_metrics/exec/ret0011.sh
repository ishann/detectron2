EXP_NAME="ret0011"
MODEL_NAME="model_0089999"
DATASET="lvis"
AREA_TYPE=0
ANN_TYPE="bbox"

python main.py --exp_name ${EXP_NAME} \
               --model_name ${MODEL_NAME} \
               --dataset ${DATASET} \
               --area_type ${AREA_TYPE} \
               --ann_type ${ANN_TYPE}
