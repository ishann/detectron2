import re

log_file = "./logs/retinanet_shapes.log"
regex_ = "[0-9]+, [0-9]+, [0-9]+, [0-9]+"

FOR_ONE_IMG = True

with open(log_file, 'r') as f:
    logs = f.readlines()


def estimate_act_map_sizes(logs, DATA):
    tensor_els = 0
    how_many_720 = 0

    for idx, log in enumerate(logs):

        matches = re.findall(regex_, log)

        if len(matches)>0:
            match = matches[0]
        else:
            continue

        match = [int(m) for m in match.split(',')]
        print(match)

        n, c, h, w = match

        if c==720:
            how_many_720 += 1
            if DATA=="LVIS":
                c *= 15.375

        if FOR_ONE_IMG:
            n = 1

        tensor_size = int(n*c*h*w)

        tensor_els += tensor_size


    return (tensor_els*4/1000000)
    #print("Activations take: {} MBs".format(tensor_els*4/1000000))

print("COCO: {:.2f} MBs.".format(estimate_act_map_sizes(logs, DATA="COCO")))
print("LVIS: {:.2f} MBs.".format(estimate_act_map_sizes(logs, DATA="LVIS")))


