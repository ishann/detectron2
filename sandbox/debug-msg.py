import torch
from torch import nn
from setproctitle import setproctitle as set_name
import time

GPUS = [1,3,5,7]
# GPUS = [0,1,2,3,4,5,6,7]

DEBUG_MSG = "pls-dont-run-high-cpu_my-job-is-crashing"

print("\n", DEBUG_MSG, "\n")

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)


def main():

    print("Init.")
    set_name(DEBUG_MSG)

    torch.device('cuda')
    models = [network()] * len(GPUS)

    print("Messaging GPUS: {}".format(GPUS))

    for idx, dev in enumerate(GPUS):
        print("Sending model {} to GPU {}.".format(idx, dev))
        model = models[idx]
        cuda_str = 'cuda:'+str(dev)
        model.cuda(device=cuda_str)
        # emodel.eval()

    print("Sleep")
    time.sleep(int(1e6))
    print("Awake. Goodbye!")


if __name__ == '__main__':
    main()




