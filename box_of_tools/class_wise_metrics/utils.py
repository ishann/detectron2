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


################################################################################
##  Base on area type, compute the per-class APs.                             ##
################################################################################
def fetch_aps(precisions, class_names, area_type=0):

    results_per_category = []
    per_cat_results = {}
    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        precision = precisions[:, :, idx, area_type]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category.append(("{}".format(name), float(ap * 100)))

        per_cat_results[name] = ap

    return results_per_category, per_cat_results


################################################################################
##  Get float scalar value from list of per-class APs.                        ##
################################################################################
def evaluate_map(results_per_category):

    aps = []
    for result in results_per_category:
        aps.append(result[1])

    float_aps = []
    for ap in aps:
        if math.isnan(ap):
            continue
        else:
            float_aps.append(ap)

    return np.mean(float_aps)


################################################################################
##  Base on area type, compute the per-class APs.                             ##
################################################################################
def print_aps(results_per_category, class_names, n_cols=6):

    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(*[results_flatten[i::n_cols] for i in range(n_cols)])
    table = tabulate(results_2d, tablefmt="pipe", floatfmt=".3f",
                     headers=["category", "AP"] * (n_cols // 2), numalign="left")
    print(table)


