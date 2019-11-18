EXP_NAME="ret0011"
MODEL_NAME="model_0089999"
DATASET="lvis"
AREA_TYPE=0
ANN_TYPE="bbox"
RESULTS_PATH="/scratch/cluster/ishann/data/lvis/samples/lvis_results_100.json"
ANN_PATH="/scratch/cluster/ishann/data/lvis/samples/lvis_val_100.json"

python main.py --exp_name ${EXP_NAME} \
               --model_name ${MODEL_NAME} \
               --dataset ${DATASET} \
               --area_type ${AREA_TYPE} \
               --ann_type ${ANN_TYPE} \
               --ann_path ${ANN_PATH} \
               --results_path ${RESULTS_PATH}
