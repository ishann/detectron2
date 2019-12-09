import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import sys
import cv2
import numpy as np
import pickle
from color_map import colormap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pycocotools.mask as mask_util
plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

IMG_PATH = '../data/coco/val2017/'
ANN_PATH = '../data/coco/annotations/instances_val2017.json'
DEBUG = False

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

def _coco_box_to_bbox(box):
  bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                  dtype=np.int32)
  return bbox

_cat_ids = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
  24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
  37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
  58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
  72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
  82, 84, 85, 86, 87, 88, 89, 90
]
num_classes = 80
_classes = {
  ind + 1: cat_id for ind, cat_id in enumerate(_cat_ids)
}
_to_order = {cat_id: ind for ind, cat_id in enumerate(_cat_ids)}
coco = coco.COCO(ANN_PATH)
CAT_NAMES = [coco.loadCats([_classes[i + 1]])[0]['name'] \
              for i in range(num_classes)]
color_list = colormap(rgb=True)

def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=2):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def vis_octagon(img, extreme_points, col, border_thick=2):
    """Visualizes a single binary mask."""

    img = img.astype(np.uint8)
    COL = (col).astype(np.uint8).tolist()
    # print('col', COL)
    extreme_points = np.array(extreme_points).reshape(8, 1, 2).astype(np.int32)
    cv2.polylines(img, [extreme_points], 
                  True, COL, border_thick)

    return img.astype(np.uint8)

def vis_ex(img, extreme_points, col, border_thick=2):
    """Visualizes a single binary mask."""

    img = img.astype(np.uint8)
    COL = (col).astype(np.uint8).tolist()
    # print('col', COL)
    ex = np.array(extreme_points).reshape(4, 2).astype(np.int32)
    
    L = 6
    T = 0.5
    cv2.arrowedLine(img, (ex[0][0], ex[0][1] + L), (ex[0][0], ex[0][1]), COL, border_thick, tipLength=T)
    cv2.arrowedLine(img, (ex[1][0] + L, ex[1][1]), (ex[1][0], ex[1][1]), COL, border_thick, tipLength=T)
    cv2.arrowedLine(img, (ex[2][0], ex[2][1] - L), (ex[2][0], ex[2][1]), COL, border_thick, tipLength=T)
    cv2.arrowedLine(img, (ex[3][0] - L, ex[3][1]), (ex[3][0], ex[3][1]), COL, border_thick, tipLength=T)
    
    '''
    R = 6
    cv2.circle(img, (ex[0][0], ex[0][1]), R, COL, -1)
    cv2.circle(img, (ex[1][0], ex[1][1]), R, COL, -1)
    cv2.circle(img, (ex[2][0], ex[2][1]), R, COL, -1)
    cv2.circle(img, (ex[3][0], ex[3][1]), R, COL, -1)
    cv2.circle(img, (ex[0][0], ex[0][1]), R, _WHITE, 2)
    cv2.circle(img, (ex[1][0], ex[1][1]), R, _WHITE, 2)
    cv2.circle(img, (ex[2][0], ex[2][1]), R, _WHITE, 2)
    cv2.circle(img, (ex[3][0], ex[3][1]), R, _WHITE, 2)
    '''
    return img.astype(np.uint8)


def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # cv2.rectangle(img, back_tl, back_br, _GRAY, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    # cv2.putText(img, txt, txt_tl, font, font_scale, (46, 52, 54), lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, thick=2):
    """Visualizes a bounding box."""
    img = img.astype(np.uint8)
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img

def get_octagon(ex):
  ex = np.array(ex).reshape(4, 2)
  w, h = ex[3][0] - ex[1][0], ex[2][1] - ex[0][1]
  t, l, b, r = ex[0][1], ex[1][0], ex[2][1], ex[3][0]
  x = 8.
  octagon = [[min(ex[0][0] + w / x, r), ex[0][1], \
                            max(ex[0][0] - w / x, l), ex[0][1], \
                            ex[1][0], max(ex[1][1] - h / x, t), \
                            ex[1][0], min(ex[1][1] + h / x, b), \
                            max(ex[2][0] - w / x, l), ex[2][1], \
                            min(ex[2][0] + w / x, r), ex[2][1], \
                            ex[3][0], min(ex[3][1] + h / x, b), \
                            ex[3][0], max(ex[3][1] - h / x, t)
                            ]]
  return octagon

def get_octagon_mask(ann):
  ann['segmentation'] = get_octagon(ann['extreme_points'])
  mask = coco.annToMask(ann)
  return mask

def get_bbox_mask(ann):
  bbox = _coco_box_to_bbox(pred['bbox'])
  segm = [[bbox[0], bbox[1], bbox[2], bbox[1], 
           bbox[2], bbox[3], bbox[0], bbox[3]]]
  ann['segmentation'] = segm
  mask = coco.annToMask(ann)
  return mask

if __name__ == '__main__':
  dets = []
  img_ids = coco.getImgIds()
  num_images = len(img_ids)
  for k in range(1, len(sys.argv)):
    pred_path = sys.argv[k]
    dets.append(coco.loadRes(pred_path))
  # import pdb; pdb.set_trace()
  for i, img_id in enumerate(img_ids):
    img_info = coco.loadImgs(ids=[img_id])[0]
    img_path = IMG_PATH + img_info['file_name']
    img = cv2.imread(img_path)
    # gt_ids = coco.getAnnIds(imgIds=[img_id])
    # gts = coco.loadAnns(gt_ids)
    # gt_img = img.copy()
    # for j, pred in enumerate(gts):
    #   bbox = _coco_box_to_bbox(pred['bbox'])
    #   cat_id = pred['category_id']
    #  gt_img = add_box(gt_img, bbox, 0, cat_id)
    for k in range(len(dets)):
      pred_ids = dets[k].getAnnIds(imgIds=[img_id])
      preds = dets[k].loadAnns(pred_ids)
      im = img.copy()
      mask_color_id = 0
      for j, pred in enumerate(preds):
        sc = pred['score']
        if sc >= 0.5:
          cat_id = pred['category_id']
          bbox = _coco_box_to_bbox(pred['bbox'])
          mask = mask_util.decode(pred['segmentation'])
          im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
          im = vis_class(
            im, (bbox[0], bbox[1] - 2), '{} {:.2f}'.format(CAT_NAMES[_to_order[cat_id]], sc))
          color_mask = color_list[mask_color_id % len(color_list), 0:3]
          mask_color_id += 1
          # print('cl', color_mask)
          pred_ = pred.copy()
          # bbox_mask = get_bbox_mask(pred_)
          # im = vis_mask(im, bbox_mask, np.array(_WHITE), show_border=False)
          octagon = get_octagon_mask(pred_)
          # im = vis_mask(im, octagon, color_mask)
          im = vis_mask(im, mask, color_mask)
          # im = vis_ex(im, pred['extreme_points'], color_mask)
          # im = vis_octagon(im, get_octagon(pred['extreme_points']), color_mask)

          
    cv2.imwrite('vis/mask_supp/{}_pred{}.png'.format(i, k), im)
    # cv2.imshow('gt', gt_img)
    # cv2.imwrite('vis/{}_gt.png'.format(i), gt_img)
    if DEBUG:
      cv2.imshow('pred{}'.format(k), im)
      cv2.waitKey()
  # coco_eval.evaluate()
  # coco_eval.accumulate()
  # coco_eval.summarize()

  
