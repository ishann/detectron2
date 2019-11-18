"""
    Most of the logic/ heavy-lifting is based off of:
    github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/coco_evaluation.py#L225
"""

################################################################################
##  Import packages.                                                          ##
################################################################################
from lvis import LVIS, LVISResults, LVISEval
from operator import itemgetter
from tabulate import tabulate
import numpy as np
import itertools
import logging
import pickle
import json
import math
import os

from config import fetch_config, print_args_stdout
from utils import fetch_aps, evaluate_map, print_aps

import ipdb


################################################################################
##  Run stuff.                                                                ##
################################################################################
def main():

    config = fetch_config()
    print_args_stdout(config)
    ipdb.set_trace()

    print("Running eval.")
    lvis_eval = LVISEval(config.ann_path, config.results_path, config.ann_type)
    lvis_eval.run()
    lvis_eval.print_results()
    print("Finished eval.")

    ipdb.set_trace()
    # All precision values: 10 x 101 x 1230 x 4
    # precision has dims (iou, recall, cls, area range)
    precisions = lvis_eval.eval['precision']

    with open(config.ann_path, 'r') as outfile:
        gt = json.load(outfile)
    cat_metas = gt['categories']
    cats = []
    for cat_meta in cat_metas:
        cats.append((cat_meta['id'], cat_meta['name']))
    cats.sort(key=itemgetter(0))
    class_names = [cat[1] for cat in cats]

    area_type = 0
    results_per_category, per_cat_results = fetch_aps(precisions, class_names, area_type)
    print("mAP for area type {}: {}".format(area_type, evaluate_map(results_per_category)))

    # Print for eye-balling.
    # print_aps(results_per_category, class_names, n_cols=6)

    # Store results_per_category into a JSON.
    with open(config.aps_json_path, 'w') as json_file:
        json.dump(per_cat_results, json_file, indent=4)

    # Store the 4D precisions tensor as a PKL.
    with open(config.prec_pkl_path, 'wb') as pkl_file:
        pickle.dump(precisions, pkl_file)


################################################################################
##  Execute main.                                                             ##
################################################################################
if __name__ == '__main__':

    main()


