import json
import os

ROOT = "/scratch/cluster/ishann/data/detectron2/output"

ret0006_json = os.path.join(ROOT, "inference_ret0006/model_0089999/lvis_v0.5_val/class_aps_ret0006_model_0089999.json")
ret0011_json = os.path.join(ROOT, "inference_ret0011/model_0089999/inference/lvis_v0.5_val/class_aps_ret0011_model_0089999.json")

with open(ret0006_json, "r") as file_:
    data_06 = json.load(file_)

with open(ret0011_json, "r") as file_:
    data_11 = json.load(file_)


diff_aps = {}
for k in data_06.keys():
    diff_aps[k] = abs(data_06[k] - data_11[k])

import math
vals = [val for val in diff_aps.values() if (not math.isnan(val))]








