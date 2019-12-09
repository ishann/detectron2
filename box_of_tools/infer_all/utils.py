################################################################################
##  Import packages.                                                          ##
################################################################################
from setproctitle import setproctitle
from os.path import normpath
from glob import glob
import subprocess
import argparse
import yaml
import copy
import ipdb
import os
import sys

from detectron2.config import get_cfg


################################################################################
##  EXecute bash command.                                                     ##
################################################################################
def execute_bash_command(bash_comm):

    process = subprocess.Popen(bash_comm.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


################################################################################
##  Setup for plain_train_net.py. Not used ATM.                               ##
################################################################################
def setup(cfg_file, cfg_opts):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(cfg_opts)
    cfg.freeze()
    # default_setup(cfg, args)

    return cfg


################################################################################
##  Prepare the exec bash file.                                               ##
################################################################################
def prepare_exec(cfg_file):

    params = {"CFG_FILE" : cfg_file,
              "SRC_FILE" : "tools/inference.py",
              "NUM_GPUS" : 1}
    cmd = 'python $SRC_FILE --num-gpus $NUM_GPUS --config-file $CFG_FILE'

    out_exec = []

    for k, v in params.items():
        if type(v)==int:v=str(v)
        line_ = k + '="' + v + '"\n'
        # print(line_)
        out_exec.append(line_)

    out_exec.append("\n")
    out_exec.append(cmd)
    out_exec.append("\n")

    return out_exec


################################################################################
##  Argument parser.                                                          ##
################################################################################
def fetch_args(ROOT_D2):

    parser = argparse.ArgumentParser(description="Infer everything.")

    parser.add_argument("--root_d2", default=ROOT_D2, metavar="FILE",
                        help="path to config file")
    parser.add_argument("--root_out_dir", required=True,
                        help="root for experiment")
    parser.add_argument("--chkpts_path", required=True,
                        help="directory where all checkpoints exist")
    parser.add_argument("--config_base", required=True,
                        help="base config file")

    args = parser.parse_args()

    return args


################################################################################
##  Print argparse.                                                           ##
################################################################################
def print_args(args):
    '''
    This is prettily written code. 'nuf said.
    '''
    dict_args = vars(args)

    for k in sorted(dict_args):
        # log.debug('{}: {} ({})'.format(k.upper(), dict_args[k], type(dict_args[k])))
        log.debug('{}: {}'.format(k.upper(), dict_args[k]))


################################################################################
##  Print argparse to stdout.                                                 ##
################################################################################
def print_args_stdout(args):
    '''
    This is prettily written code. 'nuf said.
    '''
    dict_args = vars(args)

    for k in sorted(dict_args):
        print('{}: {}'.format(k.upper(), dict_args[k]))





