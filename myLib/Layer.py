from typing import Callable, Any, List

import cupy as cp
from functools import wraps
import time
import numpy as np
if __name__ == "__main__":
    from Module import myModule, myTensor, myParameter, mySequential
else:
    from .Module import myModule, myTensor, myParameter, mySequential

time_decorator_active = True
def time_decorator(type, function_string):
    import time
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            print(f"{type} class {function_string} function call - spend time : {round(end_time - start_time, 4)}s")
            return
        return wrapper
    return decorator

class BaseLayer(myModule):
    def __init__(self):
        super(BaseLayer, self).__init__()

    def forward(self, x: myTensor) -> myTensor:
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

    def forward(self, x: myTensor) -> myTensor:
        self._backward_save = 1 / (1 + self.op.exp(-x))
        return self._backward_save

    def _backward(self, *args, **kwargs):
        return args[0] * self._backward_save * (1 - self._backward_save)

class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x: myTensor) -> myTensor:
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

    def forward(self, x: myTensor) -> myTensor: # N C_in -> N C_out
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


class MaxPool2d(BaseLayer):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(MaxPool2d, self).__init__()
        self.kernel_size = self._set_tuple(kernel_size)
        self.padding = self._set_tuple(padding)
        self.stride = self._set_tuple(stride)
        self.dilation = self._set_tuple(dilation)

    def _set_tuple(self, param):
        if isinstance(param, tuple):
            return param
        elif isinstance(param, int):
            return tuple([param, param])
        else:
            raise TypeError

    def forward(self, x: myTensor) -> myTensor:
        start_time = time.time()
        # N C H_in W_in -> N C H_out W_out
        # C H_in W_in -> C H_out W_out
        x_shape = x.shape
        N, C, self.H_in, self.W_in = x_shape[:]

        padding_x = self.op.zeros(x_shape[:-2] + tuple([self.H_in + 2 * self.padding[0], self.W_in + 2 * self.padding[1]]))

        self._back_coord = dict()

        padding_x[:, :, self.padding[0]:(self.H_in+self.padding[0]), self.padding[1]:(self.W_in+self.padding[1])] = x

        H_out = int((self.H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        W_out = int((self.W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        new_shape = x_shape[:-2] + tuple([H_out, W_out])
        out_matrix = self.op.zeros(new_shape)

        for h in range(H_out):
            h_start = self.stride[0] * h
            h_end = self.stride[0] * h + self.kernel_size[0]
            for w in range(W_out):
                w_start = self.stride[1] * w
                w_end = self.stride[1] * w + self.kernel_size[1]

                kernel = padding_x[:, :, h_start:h_end, w_start:w_end]
                max_value = self.op.max(kernel, axis=(-2, -1))
                out_matrix[:, :, h, w] = max_value

                # create back gradient mask
                res = self.op.reshape(kernel, (N*C, self.kernel_size[0] * self.kernel_size[1]))
                temp = self.op.argmax(res, axis=1)
                temp = self.op.reshape(temp, (N, C, 1))
                _mask = self.op.eye(self.kernel_size[0] * self.kernel_size[1])[temp]
                _mask = self.op.reshape(_mask, (N, C, self.kernel_size[0], self.kernel_size[1]))
                self._back_coord[(h, w)] = _mask
                # self.back_gradient_mask[:, :, h_start:h_end, w_start:w_end] += _mask

        end_time = time.time()
        print(f'forward time {end_time - start_time}s')
        return out_matrix

    def _backward(self, *args, **kwargs):
        start_time = time.time()
        N, C, H_out, W_out = args[0].shape
        _back_gradient = self.op.zeros(tuple([N, C] + [self.H_in + 2 * self.padding[0], self.W_in + 2 * self.padding[1]]))
        for h in range(H_out):
            h_start = self.stride[0] * h
            h_end = self.stride[0] * h + self.kernel_size[0]
            for w in range(W_out):
                w_start = self.stride[1] * w
                w_end = self.stride[1] * w + self.kernel_size[1]

                _back_mask = self._back_coord[(h, w)] * args[0][:, :, h:h+1, w:w+1]
                _back_gradient[:, :, h_start:h_end, w_start:w_end] += _back_mask

        end_time = time.time()
        print(f'backward time {end_time - start_time}s')
        return _back_gradient[:, :, self.padding[0]:self.H_in + 1 - self.padding[0], self.padding[1]:self.W_in + 1 - self.padding[1]]

def _test_MaxFool2d():
    np.random.seed(1)
    a = np.random.randint(low=1, high=10, size= 4*3*28*28)
    a = a.reshape((4,3,28,28))
    # a = np.ones((4, 3, 28, 28))
    tensor = myTensor(a)

    m = MaxPool2d(3, 2, 1)
    out = m(tensor)
    back = m._backward(out)
    print('end MaxFool2d')

class Conv2d(BaseLayer): # ☆☆☆☆
    def __init__(self):
        super(Conv2d, self).__init__()

    def forward(self, x: myTensor) -> myTensor:
        ...

    def _backward(self, *args ,**kwargs):
        ...


class Dropout(BaseLayer): # ☆☆☆☆
    def __init__(self):
        super(Dropout, self).__init__()

    def forward(self, x: myTensor) -> myTensor:
        ...

    def _backward(self, *args, **kwargs):
        ...

class BatchNorm2d(BaseLayer): # ☆☆☆☆☆
    def __init__(self):
        super(BatchNorm2d, self).__init__()

    def forward(self, x: myTensor) -> myTensor:
        ...

    def _backward(self, *args, **kwargs):
        ...


class Flatten(BaseLayer): # ☆
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: myTensor) -> myTensor:
        x_shape = x.shape
        self._original_shape = x.shape

        if self.end_dim == -1:
            _shape = np.prod(x_shape[self.start_dim:])
            new_shape = x_shape[:self.start_dim] + tuple([_shape])
        else:
            _shape = np.prod(x_shape[self.start_dim:self.end_dim+1])
            new_shape = x_shape[:self.start_dim] + tuple([_shape]) + x_shape[self.end_dim+1:]

        out = self.op.reshape(x, new_shape)

        return out

    def _backward(self, *args, **kwargs):
        back = self.op.reshape(args[0], self._original_shape)
        return back

def _test_Flatten():
    a = np.zeros((2, 3, 4, 5, 6, 7))
    tensor = myTensor(a).to("cuda:0")

    m = Flatten(0,2)
    out = m(tensor)
    back = m._backward(out)
    print(out)

if __name__ == "__main__":
    _test_MaxFool2d()
    _test_Flatten()




