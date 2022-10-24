from typing import Callable, Any, List
from .Module import myModule
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
        x = self.op.maximum(0, x)
        return x


if __name__ == "__main__":
    dev = cp.cuda.Device(1)
    print(isinstance("cpu", cp.cuda.Device))

    a = ReLU().to("cuda:0")

    b = np.random.randn(224*224*2000)
    b_cuda = cp.asarray(b)

    c = a(b_cuda)
    c = a(b_cuda) # 레이어 통과할때마다 메모리먹으면 메모리 부족??
    print(c)




