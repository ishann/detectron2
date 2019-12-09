EXP_ID="ret0017b"
SRC_FILE="box_of_tools/infer_all/main.py"

ROOT_OUT_DIR="output/inference_${EXP_ID}"
CHKPTS_PATH="output/${EXP_ID}"
CONFIG_BASE="configs/ret/${EXP_ID}_rpn_R_50_FPN_1x.yaml"

export CUDA_VISIBLE_DEVICES=1

python $SRC_FILE --root_out_dir ${ROOT_OUT_DIR} \
                 --chkpts_path ${CHKPTS_PATH} \
                 --config_base ${CONFIG_BASE}




