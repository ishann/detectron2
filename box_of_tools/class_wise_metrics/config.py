################################################################################
##  Import packages.                                                          ##
################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


################################################################################
##  Default values.                                                           ##
################################################################################
DATASET = "lvis"
ANN_TYPE = 'bbox'
AREA_TYPE = 0
ANN_PATH = "/scratch/cluster/ishann/data/lvis/lvis_v0.5_val.json"


################################################################################
##  Generate and return argument parser.                                      ##
################################################################################
def fetch_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True,
                        help='experiment name')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name')
    parser.add_argument('--ann_path', type=str, default=ANN_PATH,
                        help='gt annotation path')
    parser.add_argument('--results_path', type=str,
                        help='predictions path')
    parser.add_argument('--dataset', type=str, default=DATASET,
                        help='dataset name')
    parser.add_argument('--area_type', type=int, default=AREA_TYPE,
                        help='area type: 0=all, 1=s, 2=m, 3=l')
    parser.add_argument('--ann_type', type=str, default=ANN_TYPE,
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--aps_json_path', type=str,
                        help='classwise aps json path')
    parser.add_argument('--prec_pkl_path', type=str,
                        help='4D precision object path')

    args = parser.parse_args()
    config = parser.parse_args()

    # config.results_root = ("/scratch/cluster/ishann/data/detectron2/output/" +
    #                       "inference_{}/{}/inference/".format(config.exp_name,
    #                       config.model_name) + "lvis_v0.5_val/")
    config.results_root = ""
    if not config.results_path:
        config.results_path = os.path.join(config.results_root,
                                           "lvis_instances_results.json")
    if not config.aps_json_path:
        config.aps_json_path = os.path.join(config.results_root,
                           "class_aps_{}_{}.json".format(config.exp_name,
                                                         config.model_name))
    if not config.prec_pkl_path:
        config.prec_pkl_path = os.path.join(config.results_root,
                           "precisions_{}_{}.pkl".format(config.exp_name,
                                                         config.model_name))
    return config



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


