SRC_FILE="box_of_tools/infer_all/main.py"

ROOT_OUT_DIR="output/inference_ret0016"
CHKPTS_PATH="output/ret0016"
CONFIG_BASE="configs/ret/ret0016_mask_rcnn_R_50_FPN_1x.yaml"

export CUDA_VISIBLE_DEVICES=4

python $SRC_FILE --root_out_dir ${ROOT_OUT_DIR} \
                 --chkpts_path ${CHKPTS_PATH} \
                 --config_base ${CONFIG_BASE}




