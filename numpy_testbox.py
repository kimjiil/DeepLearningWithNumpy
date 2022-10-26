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


from myLib.Module import cupyTensor

a = np.array([[2, 2],
              [4, 3]])
b = cupyTensor([2, 1])

n

