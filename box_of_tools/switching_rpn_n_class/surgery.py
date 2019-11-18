################################################################################
## Import packages.                                                           ##
################################################################################
import torch

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from fvcore.common.file_io import PathManager

import pickle
import copy
from collections import OrderedDict
import numpy as np
import ipdb


################################################################################
## Define things.                                                             ##
################################################################################
config_file_coco = "./configs/coco_mask_rcnn_R_50_FPN_1x.yaml"
config_file_lvis = "./configs/lvis_mask_rcnn_R_50_FPN_1x.yaml"

OUTPATH_COCO_RPN_LVIS_CLASS = "./data/chkpts/coco_rpn_lvis_class.pth"
OUTPATH_LVIS_RPN_COCO_CLASS = "./data/chkpts/lvis_rpn_coco_class.pth"

COCO_MODEL_ZOO = "../../model_zoo/maskrcnn_coco_R50_FPN_a54504.pkl"
LVIS_MODEL_ZOO = "../../model_zoo/maskrcnn_lvis_R50_FPN_571f7c.pkl"

SAVE_MODE = "fvcore"


################################################################################
## Get COCO.                                                                  ##
################################################################################
print("\n##$$  Get COCO.  $$##\n")
with open(COCO_MODEL_ZOO, "rb") as f:
     chkpt_coco = pickle.load(f)
weights_coco = chkpt_coco['model']             # Dictionary of keys 'n' weights.
layers_coco = list(weights_coco.keys())        # List of just keys.

layers_backbone_coco = layers_coco[:281]
layers_rpn_coco = layers_coco[281:292]
layers_roiheads_coco = layers_coco[292:]

for layer in layers_roiheads_coco:
    if "predictor" in layer:
        print("{}\t{}".format(layer, weights_coco[layer].shape))

coco_shapes = {}
for layer in layers_coco:
    coco_shapes[layer] = weights_coco[layer].shape


################################################################################
## Get LVIS.                                                                  ##
################################################################################
print("\n##$$  Get LVIS.  $$##\n")
with open(LVIS_MODEL_ZOO, "rb") as f:
     chkpt_lvis = pickle.load(f)
weights_lvis = chkpt_lvis['model']
layers_lvis = list(weights_lvis.keys())

layers_backbone_lvis = layers_lvis[:281]
layers_rpn_lvis = layers_lvis[281:292]
layers_roiheads_lvis = layers_lvis[292:]

for layer in layers_roiheads_lvis:
    if "predictor" in layer:
        print("{}\t{}".format(layer, weights_lvis[layer].shape))

lvis_shapes = {}
for layer in layers_lvis:
    lvis_shapes[layer] = weights_lvis[layer].shape


################################################################################
## Compare.                                                                   ##
################################################################################
print("\n##$$  Compare shapes.  $$##\n")
for k in layers_coco:
    if not(coco_shapes[k]==lvis_shapes[k]):
        print("[{}]\tCOCO:{}\tLVIS:{}".format(k, coco_shapes[k], lvis_shapes[k]))


print("\n##$$  Compare RPN and ROIHeads weights.  $$##\n")
for layer in (layers_rpn_coco + layers_roiheads_coco):
    wgt_coco = weights_coco[layer]
    wgt_lvis = weights_lvis[layer]
    coco_avg, coco_std = np.mean(wgt_coco), np.std(wgt_coco)
    lvis_avg, lvis_std = np.mean(wgt_lvis), np.std(wgt_lvis)
    diff_avg, diff_std = (coco_avg-lvis_avg), (coco_std-lvis_std)

    print("{}:\t{:.4f}+-{:.4f}".format(layer, diff_avg, diff_std))

# ipdb.set_trace()


################################################################################
## Commence surgery.                                                          ##
################################################################################
print("\n##$$  Commence surgery.  $$##\n")
dummy_model = OrderedDict()
for layer in layers_coco:
    dummy_model[layer] = None

coco_rpn_lvis_class = copy.deepcopy(dummy_model)
lvis_rpn_coco_class = copy.deepcopy(dummy_model)

print("\n##$$  coco_rpn_lvis_class  $$##\n")
# Take coco_rpn_lvis_class. Add COCO backbone and RPN. Add LVIS ROIHeads.
for layer in (layers_backbone_coco + layers_rpn_coco):
    coco_rpn_lvis_class[layer] = weights_coco[layer]
for layer in (layers_roiheads_lvis):
    coco_rpn_lvis_class[layer] = weights_lvis[layer]

# Save
chkpt_coco_rpn_lvis_class = {}
chkpt_coco_rpn_lvis_class['__author__'] = chkpt_coco['__author__']
chkpt_coco_rpn_lvis_class['model'] = coco_rpn_lvis_class
if SAVE_MODE=="manual":
    torch.save(chkpt_coco_rpn_lvis_class, OUTPATH_COCO_RPN_LVIS_CLASS)
elif SAVE_MODE=="fvcore":
    with PathManager.open(OUTPATH_COCO_RPN_LVIS_CLASS, "wb") as f:
        torch.save(chkpt_coco_rpn_lvis_class, f)
print("Saved chkpt_coco_rpn_lvis_class to: {}".format(OUTPATH_COCO_RPN_LVIS_CLASS))


print("\n##$$  lvis_rpn_coco_class  $$##\n")
# Take lvis_rpn_coco_class. Add LVIS backbone and RPN. Add COCO ROIHeads.
for layer in (layers_backbone_lvis + layers_rpn_lvis):
    lvis_rpn_coco_class[layer] = weights_lvis[layer]
for layer in (layers_roiheads_coco):
    lvis_rpn_coco_class[layer] = weights_coco[layer]

# Save.
chkpt_lvis_rpn_coco_class = {}
chkpt_lvis_rpn_coco_class['__author__'] = chkpt_lvis['__author__']
chkpt_lvis_rpn_coco_class['model'] = lvis_rpn_coco_class
if SAVE_MODE=="manual":
    torch.save(chkpt_lvis_rpn_coco_class, OUTPATH_LVIS_RPN_COCO_CLASS)
elif SAVE_MODE=="fvcore":
    with PathManager.open(OUTPATH_LVIS_RPN_COCO_CLASS, "wb") as f:
        torch.save(chkpt_lvis_rpn_coco_class, f)
print("Saved chkpt_lvis_rpn_coco_class to: {}".format(OUTPATH_LVIS_RPN_COCO_CLASS))


print("\n##$$  Compare RPN and ROIHeads weights for stored models.  $$##\n")
for layer in (layers_rpn_coco + layers_roiheads_coco):
    wgt_coco = lvis_rpn_coco_class[layer]
    wgt_lvis = coco_rpn_lvis_class[layer]
    coco_avg, coco_std = np.mean(wgt_coco), np.std(wgt_coco)
    lvis_avg, lvis_std = np.mean(wgt_lvis), np.std(wgt_lvis)
    diff_avg, diff_std = (coco_avg-lvis_avg), (coco_std-lvis_std)

    print("{}:\t{:.4f}+-{:.4f}".format(layer, diff_avg, diff_std))








