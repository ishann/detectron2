EXP_NAME="ret0010"
MODEL_NAME="model_0144999"
DATASET="lvis"
AREA_TYPE=0
ANN_TYPE="bbox"

RESULTS_PATH="./data/ret0010/model_144_results.json"
APS_JSON_PATH="./data/ret0010/model_144_class_aps.json"
PREC_PKL_PATH="./data/ret0010/model_144_precisions.pkl"

python main.py --exp_name ${EXP_NAME} \
               --model_name ${MODEL_NAME} \
               --dataset ${DATASET} \
               --area_type ${AREA_TYPE} \
               --ann_type ${ANN_TYPE} \
               --results_path ${RESULTS_PATH} \
               --aps_json_path ${APS_JSON_PATH} \
               --prec_pkl_path ${PREC_PKL_PATH} \

