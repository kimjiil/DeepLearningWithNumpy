from typing import Callable, Any, List

import cupy as cp
import numpy as np
if __name__ == "__main__":
    from .Module import myModule, myTensor, myParameter, mySequential
else:
    from .Module import myModule, myTensor, myParameter, mySequential

class BaseLayer(myModule):
    def __init__(self):
        super(BaseLayer, self).__init__()

    def forward(self, x: myTensor):
        ...

    def backward(self, *args, **kwargs):
        if self.backward_fn:
            # 레이어가 sequeatial에 포함되지않고 단독으로 있을 경우
            temp = self._backward(*args, **kwargs)
            self.backward_fn(temp, *args[1:], **kwargs)
        else:
            # 이 레이어가 Sequential에 포함될 때
            return self._backward(*args, **kwargs)


class Sigmoid(BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: myTensor):
        self._backward_save = 1 / (1 + self.op.exp(-x))
        return self._backward_save

    def _backward(self, *args, **kwargs):
        return args[0] * self._backward_save * (1 - self._backward_save)

class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x: myTensor):
        self._backward_save = x > 0
        return self.op.maximum(0, x)

    def _backward(self, *args, **kwargs):
       return self._backward_save * args[0]

class operator_test_layer(BaseLayer):
    def __init__(self):
        super(operator_test_layer, self).__init__()

        self.binary_test_sample = myParameter(np.array([[2, 3], [5, 6]], dtype=np.float32))

    def forward(self, x: myTensor):
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
        self.weight = myParameter(self.op.random.uniform(low=-_k, high=_k, size=(in_features, out_features)))
        if bias:
            self.bias = myParameter(self.op.random.uniform(low=-_k, high=_k, size=out_features))

    def forward(self, x: myTensor): # N C_in -> N C_out
        self._backward_save = x
        if self.bias:
            x = self.op.matmul(x, self.weight) + self.bias
        else:
            x = self.op.matmul(x, self.weight)
        return x

    def _backward(self, *args, **kwargs):
        self.weight.grad = self.op.matmul(self.op.transpose(self._backward_save), args[0])
        self.bias.grad = self.op.sum(args[0], axis=0)
        _back = self.op.matmul(args[0], self.op.transpose(self.weight))
        return _back

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





