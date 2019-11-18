"""
    Convert files of the form COCO_val2014_XXXXXXXXXXXX.jpg to XXXXXXXXXXXX.jpg.
"""
from glob import glob
import os
import shutil
from tqdm import tqdm

SRC_DIR = "/scratch/cluster/ishann/data/coco/val2017_lvis/"
DST_DIR = "/scratch/cluster/ishann/data/coco/val2017/"


files = glob(os.path.join(SRC_DIR, "*.jpg"))

for file_ in tqdm(files):

    file_basename = os.path.basename(file_)
    new_file_basename = file_basename[13:]
    new_filename = os.path.join(DST_DIR, new_file_basename)

    shutil.copy(file_, new_filename)




