"""
    Conclusion:
    Proposals we got from RPN were not the same as proposals we got from the GT.
    Use _coco_box_to_bbox to convert GT before using as proposals in testing MaskRCNN.
"""
################################################################################
##  Import.                                                                   ##
################################################################################
from pycocotools.cocoeval import COCOeval
import pycocotools.coco as coco
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import numpy as np
import pickle
import ipdb
import mmcv
import sys
import cv2
import os


################################################################################
##  Define things.                                                            ##
################################################################################
IMG_PATH = '/scratch/cluster/ishann/data/lvis/val2017/'
ANN_PATH = '/scratch/cluster/ishann/data/lvis/annotations/lvis_val_100.json'
OUT_PATH = '/scratch/cluster/ishann/data/lvis/vis/rpn-debug'
VID_NAME = '/scratch/cluster/ishann/data/lvis/vis/rpn-debug/lvis_sam.mp4'
# RPN_PROPS= 'rpn.pkl'
RPN_PROPS = '/scratch/cluster/ishann/data/lvis/proposals/lvis_val_100_props_gt.pkl'

with open(RPN_PROPS, "rb") as file_: 
    props = pickle.load(file_) 

DEBUG_ = False
DEBUG_FRAMES = 60
VIS_THR = 1-1e-2

_cat_ids = list(range(1, 1231))

NUM_CLASSES = 1230
_classes = {ind + 1: cat_id for ind, cat_id in enumerate(_cat_ids)}
_to_order = {cat_id: ind for ind, cat_id in enumerate(_cat_ids)}
coco = coco.COCO(ANN_PATH)

CAT_NAMES = [coco.loadCats([_classes[i + 1]])[0]['name'] \
              for i in range(NUM_CLASSES)]
COLORS = [((np.random.random((3, )) * 0.6 + 0.4)*255).astype(np.uint8) \
              for _ in range(NUM_CLASSES)]

CAN_HGT, CAN_WDT = 640, 640


################################################################################
##  Define functions.                                                         ##
################################################################################
def _coco_box_to_bbox(box):
    """
        Get cv2 compatible bbox from COCO compatible bbox. 
    """
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.int32)
    return bbox


def add_box(image, bbox, sc, cat_id):
    """
        Add cv2 box + text on image.
    """
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


def add_to_canvas(img):
    """
        Add canvas to each image.
    """
    hgt, wdt = img.shape[:2]

    if hgt==CAN_HGT and wdt==CAN_WDT: return img

    if hgt==CAN_HGT: hgt_offset = 0
    else: hgt_offset = (CAN_HGT-hgt)//2
    if wdt==CAN_WDT: wdt_offset = 0
    else: wdt_offset = (CAN_WDT-wdt)//2

    canvas = np.zeros((CAN_HGT, CAN_WDT, 3), dtype=np.uint8)
    canvas[hgt_offset:hgt_offset+hgt, wdt_offset:wdt_offset+wdt, :] = img

    return canvas


def make_video(images, outvid, fps=0.5, format="mp4v", is_color=True):
    """
	Use only after canvas-ing. Assumes all images are CAN_HGT x CAN_WDT.

	images = List of numpy arrays of images.
	outvid = Output video name (full path).
	fps    = frames per second.
        is_color= Self-explanatory.
	format = Don't worry about it. Use the default.
    """

    size = (CAN_HGT, CAN_WDT)
    fourcc = VideoWriter_fourcc(*format)
    vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)

    for img in images:
        vid.write(img)

    vid.release()

    # return vid


################################################################################
##  Define main.                                                              ##
################################################################################
def main():
 
    img_seq = []
    # Get img_ids from lvis, instead.
    img_ids = coco.getImgIds()
    num_images = len(img_ids)

    # import pdb; pdb.set_trace()
    prog_bar = mmcv.ProgressBar(len(img_ids))

    for i, img_id in enumerate(img_ids):

        print(i)
        # ipdb.set_trace()
        if DEBUG_ is True:
            if i==DEBUG_FRAMES: break

        img_info = coco.loadImgs(ids=[img_id])[0]
        img_path = IMG_PATH + img_info['file_name']
        img = cv2.imread(img_path)

        #gt_ids = coco.getAnnIds(imgIds=[img_id])
        #gts = coco.loadAnns(gt_ids)
        prop_img = img.copy()
        img_props = props[i]
        # img_props = img_props[img_props[:,-1]>VIS_THR, :4]

        for j, prop in enumerate(img_props):

            bbox = _coco_box_to_bbox(prop)
            # bbox = np.asarray(prop, dtype=np.int32)
            cat_id = 1
            prop_img = add_box(prop_img, bbox, VIS_THR, cat_id)

        # ipdb.set_trace()
        img_name = 'prop_{}.jpg'.format(str(img_id).zfill(12))

        prop_img = add_to_canvas(prop_img)

        cv2.imwrite(os.path.join(OUT_PATH, img_name), prop_img)
        img_seq.append(prop_img)

        prog_bar.update()

    make_video(img_seq, VID_NAME)

    # ipdb.set_trace()


################################################################################
##  Execute main.                                                             ##
################################################################################
if __name__ == '__main__':
    main() 
  
