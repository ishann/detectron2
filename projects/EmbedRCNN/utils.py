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


################################################################################
##  Use the argument parser output to set up config.                          ##
################################################################################
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


################################################################################
##  Fetch an argument parser.                                                 ##
################################################################################
def get_parser():

    parser = argparse.ArgumentParser(description="Detectron2 Demo")

    parser.add_argument(
        "--config-file",
        default="configs/ret/ret0007_embed_mask_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file")

    return parser


################################################################################
##  Print argparse.                                                           ##
################################################################################
def print_args(args):
    dict_args = vars(args)

    for k in sorted(dict_args):
        log.debug('{}: {}'.format(k.upper(), dict_args[k]))


################################################################################
##  Print argparse to stdout.                                                 ##
################################################################################
def print_args_stdout(args):
    dict_args = vars(args)

    for k in sorted(dict_args):
        print('{}: {}'.format(k.upper(), dict_args[k]))



