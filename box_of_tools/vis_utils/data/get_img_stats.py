import numpy as np
# import cv2
from glob import glob
import os
import mmcv
from PIL import Image

DEBUG_ = True
TRN_IMG_PATH = "/scratch/cluster/ishann/data/lvis/train2017"
VAL_IMG_PATH = "/scratch/cluster/ishann/data/lvis/val2017"

trn_imgs = glob(os.path.join(TRN_IMG_PATH, "*.jpg"))
val_imgs = glob(os.path.join(VAL_IMG_PATH, "*.jpg"))

print("Num of trn images: {}\n".format(len(trn_imgs)))
print("Num of val images: {}\n".format(len(val_imgs)))

trn_hgt = []
trn_wdt = []

prog_bar = mmcv.ProgressBar(len(trn_imgs))

for idx, img_path in enumerate(trn_imgs):

    if DEBUG_ is True:
        if idx > 100 : break

    im = Image.open(img_path)
    width, height = im.size

    trn_wdt.append(width)
    trn_hgt.append(height)

    prog_bar.update()


print("Avg/ Max/ Min trn img width: {} / {} / {}\n".format(sum(trn_wdt)/len(trn_wdt),
                                                            max(trn_wdt), min(trn_wdt)))

print("Avg/ Max/ Min trn img height: {} / {} / {}\n".format(sum(trn_hgt)/len(trn_hgt),
                                                            max(trn_hgt), min(trn_hgt)))



val_hgt = []
val_wdt = []

prog_bar = mmcv.ProgressBar(len(val_imgs))

for idx, img_path in enumerate(val_imgs):

    if DEBUG_ is True:
        if idx > 100 : break

    im = Image.open(img_path)
    width, height = im.size

    val_wdt.append(width)
    val_hgt.append(height)

    prog_bar.update()


print("Avg/ Max/ Min val img width: {} / {} / {}\n".format(sum(val_wdt)/len(val_wdt),
                                                            max(val_wdt), min(val_wdt)))

print("Avg/ Max/ Min val img height: {} / {} / {}\n".format(sum(val_hgt)/len(val_hgt),
                                                            max(val_hgt), min(val_hgt)))









