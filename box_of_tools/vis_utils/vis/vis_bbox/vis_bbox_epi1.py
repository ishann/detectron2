"""
    This is basically Xingyi's code adapted for LVIS.
    We only work with GT annotations. This script does not parse predictions.

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
import mmcv


################################################################################
##  Define things.                                                            ##
################################################################################
IMG_PATH = '/scratch/cluster/ishann/data/lvis/val2017/'
ANN_PATH = '/scratch/cluster/ishann/data/lvis/annotations/lvis_v0.5_val.json'
OUT_PATH = '/scratch/cluster/ishann/data/lvis/vis/gt/val2017'

DEBUG_ = False
VIS_THR = 0.2

_cat_ids = list(range(1, 1231))

NUM_CLASSES = 1230
_classes = {ind + 1: cat_id for ind, cat_id in enumerate(_cat_ids)}
_to_order = {cat_id: ind for ind, cat_id in enumerate(_cat_ids)}
coco = coco.COCO(ANN_PATH)

CAT_NAMES = [coco.loadCats([_classes[i + 1]])[0]['name'] \
              for i in range(NUM_CLASSES)]
COLORS = [((np.random.random((3, )) * 0.6 + 0.4)*255).astype(np.uint8) \
              for _ in range(NUM_CLASSES)]


################################################################################
##  Define functions.                                                         ##
################################################################################
def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.int32)
    return bbox


def add_box(image, bbox, sc, cat_id):
    BOX_OFFSET = 2
    cat_id = _to_order[cat_id]
    cat_name = CAT_NAMES[cat_id]
    cat_size  = cv2.getTextSize(cat_name + '_0.00', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    color = np.array(COLORS[cat_id]).astype(np.int32).tolist()
    txt = '{}_{:.2f}'.format(cat_name, sc)

    if bbox[1] - cat_size[1] - 2 < 0:
        cv2.rectangle(image,
                      (bbox[0], bbox[1] + BOX_OFFSET),
                      (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + BOX_OFFSET),
                      color, -1)
        cv2.putText(image, txt,
                    (bbox[0], bbox[1] + cat_size[1] + BOX_OFFSET),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
    else:
        cv2.rectangle(image,
                      (bbox[0], bbox[1] - cat_size[1] - 2),
                      (bbox[0] + cat_size[0], bbox[1] - 2),
                      color, -1)
        cv2.putText(image, txt,
                    (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)

    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    return image


################################################################################
##  Define main.                                                              ##
################################################################################
def main():

    dets = []
    # Get img_ids from lvis, instead.
    img_ids = coco.getImgIds()
    num_images = len(img_ids)

    # import pdb; pdb.set_trace()
    prog_bar = mmcv.ProgressBar(len(img_ids))

    for i, img_id in enumerate(img_ids):

        # ipdb.set_trace()
        if DEBUG_ is True:
            if i>100: break

        img_info = coco.loadImgs(ids=[img_id])[0]
        img_path = IMG_PATH + img_info['file_name']
        img = cv2.imread(img_path)
        gt_ids = coco.getAnnIds(imgIds=[img_id])
        gts = coco.loadAnns(gt_ids)
        gt_img = img.copy()

        for j, pred in enumerate(gts):
            bbox = _coco_box_to_bbox(pred['bbox'])
            cat_id = pred['category_id']
            gt_img = add_box(gt_img, bbox, 1.00, cat_id)

        img_name = '{}_gt.jpg'.format(str(img_id).zfill(12))
        cv2.imwrite(os.path.join(OUT_PATH, img_name), gt_img)

        prog_bar.update()



################################################################################
##  Execute main.                                                             ##
################################################################################
if __name__ == '__main__':
    main()

