from typing import Callable, Any, List

import cupy as cp
import numpy as np
if __name__ == "__main__":
    from .Module import myModule, cupyTensor, Parameter, mySequential
else:
    from .Module import myModule, cupyTensor, Parameter, mySequential

class BaseLayer(myModule):
    def __init__(self):
        super(BaseLayer, self).__init__()

    def forward(self, x:cupyTensor):
        ...

    # __call__: Callable[..., Any] = forward

    # def _backward(self):
    #     pass
    #
    # def _update(self):
    #     pass
    #
    # def get_params(self):
    #     pass

class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x: cupyTensor):
        return self.op.maximum(0, x)

    def backward(self, x: cupyTensor):
        print("relu back test")
        return x
class operator_test_layer(BaseLayer):
    def __init__(self):
        super(operator_test_layer, self).__init__()

        self.binary_test_sample = Parameter(np.array([[2,3], [5,6]], dtype=np.float32))

    def forward(self, x: cupyTensor):
        ########## binary operator test ########################
        # add
        temp = x + self.binary_test_sample
        # mul
        temp = x * self.binary_test_sample
        # sub
        temp = x - self.binary_test_sample
        # div
        temp = x / self.binary_test_sample
        # dot
        temp = self.op.dot(x, self.binary_test_sample)
        # maxium
        temp = self.op.maximum(2, x)
        # pow **
        temp = x ** 2
        # += iadd
        temp +=  x
        # *= imul
        temp *= x
        # /= idiv
        temp /= x
        # // ???
        temp = self.binary_test_sample // x
        # == eq
        temp = self.binary_test_sample == x
        # <=
        temp = self.binary_test_sample <= x
        # >=
        temp = self.binary_test_sample >= x
        # >
        temp = self.binary_test_sample > x
        # <
        temp = self.binary_test_sample < x
        # # | or
        # temp = self.binary_test_sample | x
        # # ^ xor
        # temp = self.binary_test_sample ^ x
        # # & and
        # temp = self.binary_test_sample & x

        ############## unary operator test ###############################

        # mean
        temp = self.op.mean(x, axis=1)
        # sum
        temp = self.op.sum(x, axis=1)
        # exp
        temp_exp = self.op.exp(x)
        # reshape
        temp = self.op.reshape(x, (1, 4))
        # negative
        temp = -x
        # positive
        temp = +x
        # log
        temp = self.op.log(temp_exp)
        # abs
        temp = self.op.abs(-x)


class Linear(BaseLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        _k = np.sqrt(1/in_features)
        self.weight = Parameter(self.op.random.uniform(low=-_k, high=_k, size=(in_features, out_features)))
        if bias:
            self.bias = Parameter(self.op.random.uniform(low=-_k, high=_k, size=out_features))

    def forward(self, x: cupyTensor): # N C_in -> N C_out
        if self.bias:
            x = self.op.dot(x, self.weight) + self.bias
        else:
            x = self.op.dot(x, self.weight)
        return x

    def backward(self, back:cupyTensor):
        print("linear back test")
        self.backward_fn(back)
        # return back

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





