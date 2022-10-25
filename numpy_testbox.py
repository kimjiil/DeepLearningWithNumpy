import numpy as np
# from typing import Callable, Any
#
# class testclass():
#     def _init(self):
#         self.text = "hello"
#
#     __init__ : Callable[..., Any] = _init
#
#     def _say_hello(self, text: str):
#         print(text)
#
#     __call__ : Callable[..., Any] = _say_hello
#
#
#
# model = testclass()
#
# model((11, 1))
#
# from typing import List, Dict
#
# a: List[int] = (1, 2, 3)
#
# print(a)
#
# b = [1,2,3,4,5, ... , 9999999999]
#
# print(b)
#
# import torch
#
# input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
# weight1 = torch.tensor([[2, 3], [5, 6]], dtype=torch.float32, requires_grad=True)
# weight2 = torch.tensor([[5, 2], [2, 4]], dtype=torch.float32,requires_grad=True)
#
# out1 = input * weight1
# out2 = out1 * weight2
# z = out2.mean()
#
# z.backward()
#
# print(input.grad)
# print(weight1.grad)
# print(weight2.grad)

import numpy as np
from typing import Callable, Any, Tuple, DefaultDict, List
from collections import OrderedDict
import cupy as cp



class cupyTensor(np.ndarray):
    def __init__(self, data: np.ndarray):
        ...
        # super().__array__(np.array(data))
        # self.data = data
        # self.grad_fn = 'init'
        # self.requires_grad = 'init'
        # self.grad = 'init'
        # return super().__new__(array=data)

    def __new__(cls, input_array: list):
        print('__mynew__')
        obj = np.asarray(input_array).view(cls)
        obj.grad = 'new'
        obj.grad_fn = 'new'
        obj.requires_grad = 'new'
        # obj.data = input_array
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        self.grad = getattr(obj, 'grad', None)
        self.grad_fn = getattr(obj, 'grad_fn', None)
        self.requires_grad = getattr(obj, 'requires_grad', False)
        # self.data = getattr(obj, 'data', None)
    def __repr__(self):
        return f"cupyTensor - shape:{self.data.shape}, grad_fn:{self.grad_fn}, data:{self.data}"

    def backward(self):
        print("tensor backward!")

d = np.array([[2,2],[4,3]])
a = cupyTensor([2,1])
a.grad = [34234234234]
a.backward()
b = cupyTensor(np.array([[1,2],[3,4]]))

a_cuda = cp.array(a)
c = a + b
print(c)

print(a)