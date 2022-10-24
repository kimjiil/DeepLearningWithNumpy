from typing import Callable, Any, List
if __name__ == "__main__":
    from Module import myModule, Parameter
else:
    from .Module import myModule, Parameter
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
        return self.op.maximum(0, x)


class Linear(BaseLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        _k = np.sqrt(1/in_features)
        self.weight = Parameter(self.op.random.uniform(low=-_k, high=_k, size=(in_features, out_features)))
        if bias:
            self.bias = Parameter(self.op.random.uniform(low=-_k, high=_k, size=out_features))

    def forward(self, x): # N C_in -> N C_out
        if self.bias:
            x = self.op.dot(x, self.weight.data) + self.bias.data
        else:
            x = self.op.dot(x, self.weight.data)
        return x

if __name__ == "__main__":

    a = np.array([0])
    relu_layer = ReLU().to("cuda:0")
    linear_layer = Linear(in_features=50000, out_features=1000, bias=True)
    linear_layer.to("cuda:0")
    b = np.random.randn(2000, 50000)
    b_cuda = cp.asarray(b)

    c = linear_layer(b_cuda)
    linear_layer.to("cpu")
    c = linear_layer(b)
    print()
    # c = a(b_cuda) # 레이어 통과할때마다 메모리먹으면 메모리 부족??





