"""
    Changelog:
    [1] Builds on top of epi2 and epi3.
    [2] Works for both gt and bbox outputs of lvis_test.py.
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
ANN_PATH = '/scratch/cluster/ishann/data/lvis/annotations/lvis_v0.5_val.json'
PRE_PATH = "/scratch/cluster/ishann/research/recognizing-every-thing/logs/ret/" + \
           "ret0001/results_1e-3_thresh_300dets_img_infer_1e-3/" + \
           "ret0001_30_json_results.bbox.json"
OUT_PATH = '/scratch/cluster/ishann/data/lvis/vis/gt/val2017_sup'
VID_NAME = '/scratch/cluster/ishann/data/lvis/vis/gt/val2017_sup/lvis_val.mp4'

DEBUG_ = True
DEBUG_FRAMES = 60
VIS_THR = 0.4

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
SUP_CAN_HGT, SUP_CAN_WDT = 640, 1280


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

    size = (SUP_CAN_HGT, SUP_CAN_WDT)
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

    # import ipdb; ipdb.set_trace()
    dets = []
    dets.append(coco.loadRes(PRE_PATH))

    prog_bar = mmcv.ProgressBar(len(img_ids))
    for i, img_id in enumerate(img_ids):

        # ipdb.set_trace()
        if DEBUG_ is True:
            if i==DEBUG_FRAMES: break

        img_info = coco.loadImgs(ids=[img_id])[0]
        img_path = IMG_PATH + img_info['file_name']
        img = cv2.imread(img_path)

        # Create a gt labeled img.
        gt_ids = coco.getAnnIds(imgIds=[img_id])
        gts = coco.loadAnns(gt_ids)
        gt_img = img

        for j, pred in enumerate(gts):
            bbox = _coco_box_to_bbox(pred['bbox'])
            cat_id = pred['category_id']
            gt_img = add_box(gt_img, bbox, 1.00, cat_id)
        gt_img = add_to_canvas(gt_img)
       
        # Create a predictions labeled img.
        for k in range(len(dets)):
            pred_ids = dets[k].getAnnIds(imgIds=[img_id])
            preds = dets[k].loadAnns(pred_ids)
            pred_img = img.copy()
            for j, pred in enumerate(preds):
                bbox = _coco_box_to_bbox(pred['bbox'])
                sc = pred['score']
                cat_id = pred['category_id']
                if sc > VIS_THR:
                    pred_img = add_box(pred_img, bbox, sc, cat_id)
          
        # ipdb.set_trace()
        pred_img = add_to_canvas(pred_img)

        # Create a super image and save it.
        sup_img = np.concatenate((gt_img, pred_img), axis=1)        
        sup_img_name = 'pre_{}.jpg'.format(str(img_id).zfill(12))
        cv2.imwrite(os.path.join(OUT_PATH, sup_img_name), sup_img)
        img_seq.append(sup_img)

        prog_bar.update()

    # ipdb.set_trace()

    # For some reason, video is not working. Comment for now. Move to debugging 
    # make_video(img_seq, VID_NAME)


################################################################################
##  Execute main.                                                             ##
################################################################################
if __name__ == '__main__':
    main() 
 
