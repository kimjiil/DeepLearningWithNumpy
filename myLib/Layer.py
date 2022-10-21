from typing import Callable, Any, List
from Module import myModule
import cupy as cp
import numpy as np

class BaseLayer(myModule):
    def __init__(self):
        super(BaseLayer, self).__init__()

    def forward(self, x):
        ...

    # __call__: Callable[..., Any] = forward

    def _backward(self):
        pass

    def _update(self):
        pass

    def get_params(self):
        pass


class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        out = self.op.maximum(0, x)
        return out


if __name__ == "__main__":
    dev = cp.cuda.Device(1)
    print(isinstance("cpu", cp.cuda.Device))

    a = ReLU()

    b = np.random.randn(3)

    c = a(b)

    print()




