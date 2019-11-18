EXP_NAME="ret0012_lvis_dla_1x_dgx4"
MODEL_NAME="model_089"
DATASET="lvis"
AREA_TYPE=0
ANN_TYPE="bbox"
RESULTS_PATH="./data/CenterNet/ret0012_lvis_dla_1x_dgx4/model_089_results.json"

APS_JSON_PATH="./data/CenterNet/ret0012_lvis_dla_1x_dgx4/model_089_class_aps.json"
PREC_PKL_PATH="./data/CenterNet/ret0012_lvis_dla_1x_dgx4/model_089_precisions.pkl"

python main.py --exp_name ${EXP_NAME} \
               --model_name ${MODEL_NAME} \
               --dataset ${DATASET} \
               --area_type ${AREA_TYPE} \
               --ann_type ${ANN_TYPE} \
               --aps_json_path ${APS_JSON_PATH} \
               --prec_pkl_path ${PREC_PKL_PATH} \
               --results_path ${RESULTS_PATH}
