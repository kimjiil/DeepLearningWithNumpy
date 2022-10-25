if __name__ == "__main__":
    from Module import myModule, Parameter
else:
    from .Module import myModule, Parameter

import cupy as cp
import numpy as np

class testloss(myModule):
    def __init__(self):
        super(testloss, self).__init__()

    def __call__(self, pred, label):
        return 0.5 * cp.mean(cp.sum((pred - label) ** 2))