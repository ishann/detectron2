################################################################################
##  Import packages.                                                          ##
################################################################################
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


from embed_learn.utils import setup_cfg, get_parser, print_args, print_args_stdout


"""
There appears to be a mismatch in the loaded model weights and
the default model created by reading the cfg file.
"""
################################################################################
##  Define things.                                                            ##
################################################################################
CFG_PATH = "configs/ret/ret0007_embed_mask_rcnn_R_50_FPN_1x.yaml"
FILE_PATH = "model_zoo/lvis_R50_FPN_571f7c.pkl"


################################################################################
##  Define main.                                                              ##
################################################################################
def main()

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    args.config_file = CFG_PATH

    cfg = setup_cfg(args)

    model = build_model(cfg)  # returns a torch.nn.Module
    DetectionCheckpointer(model).load(FILE_PATH)


################################################################################
##  Execute main.                                                             ##
################################################################################
if __name__ == "__main__":
    main()


