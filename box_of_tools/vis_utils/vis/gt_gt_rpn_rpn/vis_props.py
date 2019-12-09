"""
   Draw RPN proposals from trained models.

   NOTE: It appears that there is no NMS at test time. Whyy?!
"""
################################################################################
##  Import.                                                                   ##
################################################################################
# Import lvis
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import os
import sys
import cv2
import numpy as np
import pickle
import ipdb

np.random.seed(0)  # Get the same colors for the same class every time.


################################################################################
##  Define things.                                                            ##
################################################################################
DATA = 'coco'
OUT_TYPE = 'prop'
DEBUG_ = False
VIS_THR = 0.2
LABEL = False
OUT_NAME_SUFFIX = "{}_{}".format(OUT_TYPE, DATA)
IMG_PATH = '/scratch/cluster/ishann/data/lvis/val2017/'
OUT_PATH = '/scratch/cluster/ishann/data/cross/{}_{}_bboxes'.format(DATA, OUT_TYPE)
if not os.path.isdir(OUT_PATH): os.makedirs(OUT_PATH)

#ANN_PATH = "/scratch/cluster/ishann/data/detectron2/output/inference/ret0017_fewer/inference/coco_2017_val/box_proposals.pkl"
ANN_PATH = "/scratch/cluster/ishann/data/detectron2/output/inference/ret0018_fewer/inference/lvis_v0.5_val/box_proposals.pkl"

COLOR = [ 135,  74, 32]


################################################################################
##  Define functions.                                                         ##
################################################################################
def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.int32)
    return bbox


def add_box(image, bbox, sc):
    BOX_OFFSET = 2
    color = np.array(COLOR).astype(np.int32).tolist()
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    return image


################################################################################
##  Define main.                                                              ##
################################################################################
def main():

    print("\nDATA:{}\nOUT_TYPE:{}\n".format(DATA, OUT_TYPE))

    with open(ANN_PATH, "rb") as f:
        proposals = pickle.load(f)

    img_ids = proposals['ids']
    bboxes = proposals['boxes']
    logits = proposals['objectness_logits']
    num_images = len(img_ids)

    # ipdb.set_trace()

    for i, img_id in enumerate(img_ids):

        if i%10==0: print("{}/{}".format(i, len(img_ids)), end="\r")

        # ipdb.set_trace()
        if DEBUG_ and i>DEBUG_: break

        img_bboxes = bboxes[i]
        img_logits = logits[i]
        #thresh_logits = sorted(img_logits)[-100]

        img_name = "COCO_val2014_"+str(img_id).zfill(12)+".jpg"
        img_path = IMG_PATH + img_name
        img = cv2.imread(img_path)
        gt_img = img.copy()

        for j, pred in enumerate(img_bboxes):
            logit = img_logits[j]
            #if logit<thresh_logits: continue
            bbox = _coco_box_to_bbox(pred)
            gt_img = add_box(gt_img, bbox, 1.00)

        img_name = '{}_{}.jpg'.format(str(img_id).zfill(12), OUT_NAME_SUFFIX )
        cv2.imwrite(os.path.join(OUT_PATH, img_name), gt_img)




################################################################################
##  Execute main.                                                             ##
################################################################################
if __name__ == '__main__':
    main()

