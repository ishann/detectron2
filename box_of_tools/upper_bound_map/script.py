################################################################################
##  Import packages.                                                          ##
################################################################################
import torch
import pickle
import json
import copy
import ipdb
import time

from lvis import LVIS, LVISResults, LVISEval

from utils import test_gt_boxes_as_anns, bbox_to_coco_box, coco_box_to_bbox
from utils import compute_area, pairwise_iou, fetch_optimal_proposal


################################################################################
##  Define things.                                                            ##
################################################################################
ANN_PATH = "./data/lvis_v0.5_val.json"
PRE_PATH = "./data/lvis_preds_ret0015/lvis_instances_results.json"
# BOX_PATH = "./data/ret0017_trn-coco_infer-cocoval_box_proposals.pkl"
BOX_PATH = "./data/mock_constrained_rpn_trn-lvis_infer-cocoval_box_proposals.pkl"
PRE_OUT_PATH = "./data/coco_prop_pred.json"
ANN_TYPE = 'bbox'


DO_GT_CHECK = False
SAMPLING_WITH_REPLACEMENT = True


################################################################################
##  Import things.                                                            ##
################################################################################
with open(ANN_PATH, "r") as f:
    ann = json.load(f)
print("Loaded GT LVIS JSON.\n")

with open(PRE_PATH, "r") as f:
    pre = json.load(f)
print("Loaded lvis predictions JSON.\n")

with open(BOX_PATH, "rb") as f:
    box = pickle.load(f)
print("Loaded trn-lvis_infer-cocoval box proposals.\n")


################################################################################
##  Do things.                                                                ##
################################################################################
# Test if the GT annotations can provide perfect LVIS AP.
if DO_GT_CHECK:
    test_gt_boxes_as_anns(ann, ANN_PATH)

# Get down to business.
all_prop_boxes = box['boxes']
all_prop_ids = box['ids']
annotations = ann['annotations']

template_pre = {'image_id': 0, 'category_id': 0,
                'bbox': [0., 0., 0., 0.], 'score': 1.}

gt_to_pre = []
for idx, annotation in enumerate(annotations):

    pre = copy.deepcopy(template_pre)

    pre['image_id'] = annotation['image_id']
    pre['category_id'] = annotation['category_id']
    pre['score'] = 1.0
    img_proposals = all_prop_boxes[all_prop_ids.index(annotation['image_id'])]
    gt_bbox = coco_box_to_bbox(annotation['bbox'])
    opt_prop = fetch_optimal_proposal(gt_bbox, img_proposals)
    pre['bbox'] = opt_prop
    gt_to_pre.append(pre)

with open(PRE_OUT_PATH, "w") as f:
    pre = json.dump(gt_to_pre, f)
print("Stored GT to pre_out JSON.\n")

# Eval how well we did.
lvis_eval = LVISEval(ANN_PATH, PRE_OUT_PATH, ANN_TYPE)
print("Constructed lvis_eval object.")
lvis_eval.run()
print("Finished lvis_eval.run()")
lvis_eval.print_results()










