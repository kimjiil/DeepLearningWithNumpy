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

    def _set_tuple(self, param):
        if isinstance(param, tuple):
            return param
        elif isinstance(param, int):
            return tuple([param, param])
        else:
            raise TypeError

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
        if self.bias:
            self.bias.grad = self.op.sum(args[0], axis=0)
        _back = self.op.matmul(args[0], self.op.transpose(self.weight))
        return _back

def _test_Linear():
    a = np.random.randint(1, 10, size= (4, 16))
    tensor = myTensor(a)
    m = Linear(16, 32)
    out = m(tensor)
    back = m._backward(out)
    print('end Linear')

class MaxPool2d(BaseLayer):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(MaxPool2d, self).__init__()
        self.kernel_size = self._set_tuple(kernel_size)
        self.padding = self._set_tuple(padding)
        self.stride = self._set_tuple(stride)
        self.dilation = self._set_tuple(dilation)

    def forward(self, x: myTensor) -> myTensor:
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

        ##########################################################################
        small_mask = self.op.reshape(self.op.arange(self.kernel_size[0] * self.kernel_size[1], 0, -1), (self.kernel_size[0], self.kernel_size[1]))* np.finfo(float).eps * 10
        small_mask = self.op.tile(small_mask, [N, C, 1, 1])
        ##########################################################################

        for h in range(H_out):
            h_start = self.stride[0] * h
            h_end = self.stride[0] * h + self.kernel_size[0]
            for w in range(W_out):
                w_start = self.stride[1] * w
                w_end = self.stride[1] * w + self.kernel_size[1]

                kernel = padding_x[:, :, h_start:h_end, w_start:w_end] + small_mask
                max_value = self.op.max(kernel, axis=(-2, -1))
                out_matrix[:, :, h, w] = max_value
                self._back_coord[(h, w)] = kernel >= max_value.reshape((N, C, 1, 1))

        return out_matrix

    def _backward(self, *args, **kwargs):
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
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._set_tuple(kernel_size)
        self.stride = self._set_tuple(stride)
        self.padding = self._set_tuple(padding)
        self.dilation = self._set_tuple(dilation)

        _k = np.sqrt(1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))

        if bias:
            #create bias
            self.bias = myParameter(self.op.random.uniform(low=-_k, high=_k, size=self.out_channels))
        else:
            self.bias = None

        self.weight = myParameter(self.op.random.uniform(low=-_k, high=_k,
                                                         size=(self.in_channels,
                                                               self.kernel_size[0],
                                                               self.kernel_size[1],
                                                               self.out_channels)))

    def forward(self, x: myTensor) -> myTensor:
        self._backward_save = x
        # N C_in H_in W_in => N C_out H_out W_out
        N, C, self.H_in, self.W_in = x.shape

        self.H_out = int((self.H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        self.W_out = int((self.W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        self.H_padding = self.H_in + 2 * self.padding[0]
        self.W_padding = self.W_in + 2 * self.padding[1]

        padding_x = self.op.zeros((N, C, self.H_in + 2 * self.padding[0], self.W_in + 2 * self.padding[1]))
        padding_x[:, :, self.padding[0]:self.H_padding - self.padding[0], self.padding[1]:self.W_padding - self.padding[1]] = x

        output = self.op.zeros((N, self.out_channels, self.H_out, self.W_out))

        for h in range(self.H_out):
            h_start = h * self.stride[0]
            h_end = h * self.stride[0] + self.kernel_size[0]
            for w in range(self.W_out):
                w_start = w * self.stride[1]
                w_end = w * self.stride[1] + self.kernel_size[1]
                window = padding_x[:, :, h_start:h_end, w_start:w_end] # N, C_in, k_h, k_w
                window = self.op.reshape(window, (N, self.in_channels, self.kernel_size[0], self.kernel_size[1], 1))
                # weight : C_in, k_h, k_w, C_out => 1, C_in, k_h, k_w, C_out
                # _out : N, C_in, k_h, k_w, C_out
                _out = window * self.op.reshape(self.weight, (1, self.in_channels, self.kernel_size[0], self.kernel_size[1], self.out_channels))
                _out = self.op.sum(_out, axis=(1, 2, 3)) # N, C_out

                output[:, :, h, w] = _out

        return output

    def _backward(self, *args, **kwargs):
        # self.bias.grad = None

        # back in shape : N, C_out, H, W
        # back out shape : N, C_in, H, W
        _back_in = args[0]

        N, C, self.H_in, self.W_in = self._backward_save.shape

        self.weight.grad = self.op.zeros_like(self.weight)
        # _back_gradient = self.op.zeros_like(self._backward_save)

        padding_x = self.op.zeros((N, C, self.H_in + 2 * self.padding[0], self.W_in + 2 * self.padding[1]))
        _back_gradient = self.op.zeros_like(padding_x)

        padding_x[:, :, self.padding[0]:self.H_padding - self.padding[0], self.padding[1]:self.W_padding - self.padding[1]] = self._backward_save

        for h_i in range(self.H_out):
            h_start = h_i * self.stride[0]
            h_end = h_start + self.kernel_size[0]
            for w_i in range(self.W_out):
                w_start = w_i * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                # padding_x shape 4, 3, 28, 28
                # _back_in shape 4, 16, 13, 13
                # weight shape 3, 3, 3, 16

                # calculate weight gradient
                window_x = padding_x[:, :, h_start:h_end, w_start:w_end].reshape((N, self.in_channels, self.kernel_size[0], self.kernel_size[1], 1))
                _grad = self.op.transpose(_back_in[:, :, h_i, w_i].reshape((N, self.out_channels, 1, 1, 1)), axes=(0, 4, 2, 3, 1))
                self.weight.grad += self.op.sum(window_x * _grad, axis=0)

                # back_out shape 4, 3, 28, 28 => 4 3 3 3 / N Cin H W
                # _back_in shape 4, 16, 13, 13 =>  4 16 1 1  / N Cout H W
                # weight shape 3, 3, 3, 16 / Cin H W Cout

                # _grad N Cin(1) H W Cout
                # weight => 1 3 3 3 16 / N Cin H W Cout
                window_backgradient = _grad * self.weight.reshape((1, *self.weight.shape))
                _back_gradient[:, :, h_start:h_end, w_start:w_end] += self.op.sum(window_backgradient, axis=-1)

        return _back_gradient[:, :, self.padding[0]:self.H_padding - self.padding[0], self.padding[1]:self.W_padding - self.padding[1]]

def _test_Conv2d():
    np.random.seed(1)
    a = np.random.randint(low=1, high=10, size= 4*3*28*28)
    a = a.reshape((4, 3, 28, 28))
    a = a.astype(np.float32)
    # a = np.ones((4, 3, 28, 28))
    tensor = myTensor(a)

    m = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0)
    out = m(tensor)
    back = m._backward(out)
    print('end conv2d')


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
    _test_Linear()
    _test_MaxFool2d()
    _test_Flatten()
    _test_Conv2d()




