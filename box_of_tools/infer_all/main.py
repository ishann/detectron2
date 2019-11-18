"""
ALGORITHM:
    1. Read all checkpoints for an experiment.
    2. Read a base config file for an experiment.
    3. For each checkpoint:
        [a] Create a new dir where everything will be dumped.
        [b] Write a custom config file to this dir that will be sued.
        [c] Writie a custom bash file which will run the experiment.
        [d] Execute the bash file.


TODOs:
    1. Pretty print commandline args to screen/ (eventually) logger.
    2. Add logger to store high-level progress of this script.
"""
################################################################################
##  Import packages.                                                          ##
################################################################################
from os.path import normpath
from glob import glob
import subprocess
import argparse
import shutil
import yaml
import copy
import ipdb
import os
import sys

from detectron2.config import get_cfg
from utils import execute_bash_command, setup, prepare_exec
from utils import fetch_args, print_args


################################################################################
##  Define things.                                                            ##
################################################################################
ROOT_D2 = "/scratch/cluster/ishann/research/recognizing-every-thing/D2/detectron2"


################################################################################
##  Do things.                                                                ##
################################################################################
def main():

    args = fetch_args()
    print_args_stdout(args)

    CHKPTS = sorted(glob(os.path.join(ROOT_D2, args.chkpts_path, "*.pth")))
    if not os.path.exists(args.root_out_dir): os.makedirs(args.root_out_dir)

    with open(os.path.normpath(args.config_base)) as file_:
        config_base = yaml.load(file_, Loader=yaml.FullLoader)

    BASE_CFG_SRC = "./configs/Base-RCNN-FPN.yaml"
    BASE_CFG_DST = os.path.join(args.root_out_dir, os.path.basename(BASE_CFG_SRC))
    shutil.copy(BASE_CFG_SRC, BASE_CFG_DST)

    for idx, chkpt in enumerate(CHKPTS):

        print("\nCheckpoint: {}/{}\n".format(idx+1, len(CHKPTS)))

        model_name = os.path.basename(chkpt)
        chkpt = os.path.join(args.chkpts_path, os.path.basename(chkpt))

        # Generate a dir for all processing and storage.
        out_base_dir = model_name[:-4]
        out_dir = os.path.join(args.root_out_dir, out_base_dir)
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        # Generate a config YAML for running the inference.
        # ipdb.set_trace()
        config_ = copy.deepcopy(config_base)
        config_['OUTPUT_DIR'] = out_dir
        config_['MODEL']['WEIGHTS'] = chkpt
        config_['MODEL']['MASK_ON'] = False   # Ain't nobody got time for that.

        out_cfg_file = os.path.join(out_dir, "config_{}.yaml".format(out_base_dir))
        with open(out_cfg_file, 'w') as file_: yaml.dump(config_, file_)

        # Generate a SH file.
        out_exec_file = os.path.join(out_dir, "exec_{}.sh".format(out_base_dir))
        exec_ = prepare_exec(out_cfg_file)
        with open(out_exec_file, 'w') as file_: file_.writelines(exec_)

        # Alright, alright, alright.
        cmd = "sh {}".format(out_exec_file)
        execute_bash_command(cmd)


if __name__=="__main__":

    main()




