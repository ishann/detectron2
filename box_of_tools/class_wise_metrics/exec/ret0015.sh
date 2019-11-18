EXP_NAME="ret0015"
MODEL_NAME="model_0089999"
DATASET="lvis"
AREA_TYPE=0
ANN_TYPE="bbox"

RESULTS_PATH="./data/${EXP_NAME}/model_089_results.json"
APS_JSON_PATH="./data/${EXP_NAME}/model_089_class_aps.json"
PREC_PKL_PATH="./data/${EXP_NAME}/model_089_precisions.pkl"

python main.py --exp_name ${EXP_NAME} \
               --model_name ${MODEL_NAME} \
               --dataset ${DATASET} \
               --area_type ${AREA_TYPE} \
               --ann_type ${ANN_TYPE} \
               --results_path ${RESULTS_PATH} \
               --aps_json_path ${APS_JSON_PATH} \
               --prec_pkl_path ${PREC_PKL_PATH} \

