import numpy as np
from typing import Callable, Any, Tuple, DefaultDict, List
from collections import OrderedDict
import cupy as cp

import operator
from functools import wraps

decorator_active = True

def print_decorator(type, function_string, active):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if active:
                print(f"{type} class {function_string} call!!")
            return func(*args, **kwargs)
        return wrapper
    return decorator

class myModule:
    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._dev = dict()
        self._dev['device'] = 'cpu'  # default
        self._dev['id'] = -1  # cpu default
        self.op = op() # default
        self.backward_fn = None
        # self.op = np

    def forward_call(self, *args, **kwargs):
        self.backward_fn = args[0].backward_fn
        args[0].backward_fn = self.backward

        return self.forward(*args, **kwargs)

    __call__: Callable[..., Any] = forward_call

    def backward(self):
        print("backward!!!")
        print("backward!!!")
        print("backward!!!")
        print("backward!!!")

    def eval(self):
        pass

    def train(self):
        pass

    def parameters(self):
        pass

    # 현재 함수의 하위 함수를 모두 호출하기 위한 iterator
    def children(self):
        module_list = self.__dict__['_modules']
        for module_key in module_list:
            module = module_list[module_key]
            yield module

    def to(self, *args, **kwargs):
        '''
        [function call]
        to()
        to("cpu")
        to("cuda:0")
        to(device = "cpu")
        to(device = "cuda:0")
        '''

        if len(args) > 1 or len(kwargs) > 1:
            raise "too many parameters, cpu or cuda:{}"
        elif args:
            _args = args[0]
        elif kwargs:
            if "device" in kwargs:
                _args = kwargs['device']
            else:
                raise "unvalid parameter key"
        else:
            _args = None


        for child in self.children():
            child.to(*args, **kwargs)

        if _args == None: # call: to()
            self._dev['id'] = 0
            self._dev['device'] = "cuda"
            self.op.set_op(cp)
            # self.op = cp
            # cp.cuda.Device(self._dev['id']).use()
            cp.cuda.runtime.setDevice(self._dev['id'])

            _parameters = self.__dict__['_parameters']
            for _param_key in _parameters:
                _parameters[_param_key].to(f"{self._dev['device']}:{self._dev['id']}")

        elif "cpu" in _args: # call: to("cpu") or to(device="cpu")
            self._dev['id'] = -1
            self._dev['device'] = "cpu"
            self.op.set_op(np)
            # self.op = np

            _parameters = self.__dict__['_parameters']
            for _param_key in _parameters:
                _parameters[_param_key].to(f"{self._dev['device']}:{self._dev['id']}")

        elif "cuda" in _args:
            gpu_id = int(_args.split(":")[-1])

            self._dev['id'] = gpu_id
            self._dev['device'] = "cuda"
            self.op.set_op(cp)
            # self.op = cp

            # cp.cuda.Device(self._dev['id']).use()
            cp.cuda.runtime.setDevice(self._dev['id'])

            _parameters = self.__dict__['_parameters']
            for _param_key in _parameters:
                _parameters[_param_key].to(f"{self._dev['device']}:{self._dev['id']}")
        return self

    def __setattr__(self, key, value):
        if isinstance(value, myModule):
            modules = self.__dict__.get("_modules")
            modules[key] = value
            # super(myModule, self).__setattr__(key, value)
            object.__setattr__(self, key, value)
        elif isinstance(value, Parameter):
            parameters = self.__dict__.get("_parameters")
            parameters[key] = value
            # super(myModule, self).__setattr__(key, value)
            object.__setattr__(self, key, value)
        else:
            # super(myModule, self).__setattr__(key, value)
            object.__setattr__(self, key, value)

        # return object
    # def __getattr__(self, item):
    #     print("get!!")
    #     ...

    def add_module(self, key, module):
        self._modules[key] = module

class op:
    def __init__(self):
        self._op = np #defualt
        self.random = self._random(self._op.random)

    def set_op(self, op):
        self._op = op
    ###### original function end #####################


    # ↓ numpy and cupy class wrapper
    class _random:
        def __init__(self, op):
            self._op = op

        def uniform(self, low=0.0, high=1.0, size=None):
            return self._op.uniform(low=low, high=high, size=size)

    # ↓ numpy and cupy function wrapper
    def maximum(self, x1, x2, *args, **kwargs):

        if isinstance(x1, cupyTensor):
            temp_x1 = x1.data
            temp_fn = x1.backward_fn
            temp_op = x1.op
        else:
            temp_x1 = x1

        if isinstance(x2, cupyTensor):
            temp_x2 = x2.data
            temp_fn = x2.backward_fn
            temp_op = x2.op
        else:
            temp_x2 = x2
        new_temp = self._op.maximum(temp_x1, temp_x2, *args, **kwargs)
        # new_temp = self._op.maximum(x1, x2, *args, **kwargs)
        new = cupyTensor(new_temp)
        new.grad_fn = "<opertaion.maximum>"
        new.backward_fn = temp_fn
        new.op = temp_op
        return new
        # return self._op.maximum(x1, x2, *args, **kwargs)

    def where(self, *args, **kwargs):
        ...

    def dot(self, *args,  **kwargs):
        return self._op.dot(*args,  **kwargs)

    @print_decorator("operator", "mean", decorator_active)
    def mean(self, *args, **kwargs):
        obj = self._op.mean(*args, **kwargs)
        return obj

    @print_decorator("operator", "sum", decorator_active)
    def sum(self, *args, **kwargs):
        obj = self._op.sum(*args, **kwargs)
        return obj

    @print_decorator("operator", "exp", decorator_active)
    def exp(self, *args, **kwargs):
        obj = self._op.exp(*args, **kwargs)
        return obj

    def reshape(self, *args, **kwargs):
        obj = self._op.reshape(*args, **kwargs)
        return obj

    def log(self, *args, **kwargs):
        obj = self._op.log(*args, **kwargs)
        return obj

class cupyTensor(myModule):
    def __init__(self, data):
        super(cupyTensor, self).__init__()
        self.data = data
        self.grad_fn = None
        self.requires_grad = False
        self.grad = None
        # self._op = np

    # def to(self, dev):
    #     if "cuda" in dev:
    #         cp.cuda.runtime.setDevice(int(dev.split(":")[-1]))
    #         self._op = cp
    #         self.data = cp.asarray(self.data)
    #     elif "cpu" in dev:
    #         self._op = np
    #         self.data = cp.asnumpy(self.data)
    #         cp._default_memory_pool.free_all_blocks()
    #
    #     return self

    def to(self, *args, **kwargs):
        if len(args) > 1 or len(kwargs) > 1:
            raise "too many parameters, cpu or cuda:{}"
        elif args:
            _args = args[0]
        elif kwargs:
            if "device" in kwargs:
                _args = kwargs['device']
            else:
                raise "unvalid parameter key"
        else:
            _args = None

        if "cuda" in _args:
            cp.cuda.runtime.setDevice(int(_args.split(":")[-1]))
            # self._op = cp
            self.op.set_op(cp)
            self.data = cp.asarray(self.data)
        elif "cpu" in _args:
            # self._op = np
            self.op.set_op(np)
            self.data = cp.asnumpy(self.data)
            cp._default_memory_pool.free_all_blocks()
        else:
            cp.cuda.runtime.setDevice(0)
            # self._op = cp
            self.op.set_op(cp)
            self.data = cp.asarray(self.data)

        return self

    def __repr__(self):
        return f"cupyTensor - shape:{self.data.shape}, grad_fn:{self.grad_fn}, data:{self.data}"

    def backward(self):
        print("tensor backward!")
        self.backward_fn(self)

    #slicing call
    def __setslice__(self, i, j, sequence):
        ...

    #slicing call
    def __getitem__(self, item):
        self.data = self.data[item]
        return self

    def __setitem__(self, key, value):
        self.data[key] = value
        return self


    def binary_operator_function_call(self, operator, other):
        if isinstance(other, cupyTensor):
            temp_data = other.data
        else:
            temp_data = other

        _new_data = operator(self.data, temp_data)
        _new = cupyTensor(_new_data)
        _new.grad_fn = f"{operator}"
        _new.op = self.op
        _new.backward_fn = self.backward_fn
        return _new

    def unary_operator_function_call(self, *args, **kwargs):
        # args[0]은 항상 operator로 들어옴
        print("unary operator function call!!")
        operator = args[0]

        _new_data = operator(self.data, *args[1:], **kwargs)
        # if not isinstance(type(_new_data), np.ndarray):
        #     _new_data = np.array
        _new = cupyTensor(_new_data)
        _new.grad_fn = f"{operator}"
        _new.backward_fn = self.backward_fn
        return _new

    def __add__(self, other):
        return self.binary_operator_function_call(operator.add, other=other)

    def __radd__(self, other):
        return self.binary_operator_function_call(operator.add, other=other)

    def __mul__(self, other):
        return self.binary_operator_function_call(operator.mul, other=other)

    def __rmul__(self, other):
        return self.binary_operator_function_call(operator.mul, other=other)

    def __truediv__(self, other):
        return self.binary_operator_function_call(operator.truediv, other=other)

    def __rtruediv__(self, other):
        return self.binary_operator_function_call(operator.truediv, other=other)

    def __sub__(self, other):
        return self.binary_operator_function_call(operator.sub, other=other)

    def __rsub__(self, other):
        return self.binary_operator_function_call(operator.sub, other=other)

    def __pow__(self, power, modulo=None):
        return self.binary_operator_function_call(operator.pow, other=power)

    def __rpow__(self, power, modulo=None):
        return self.binary_operator_function_call(operator.pow, other=power)

    # ==
    def __eq__(self, other):
        return self.binary_operator_function_call(operator.eq, other=other)

    # <
    def __lt__(self, other):
        return self.binary_operator_function_call(operator.lt, other=other)

    # <=
    def __le__(self, other):
        return self.binary_operator_function_call(operator.le, other=other)

    # !=
    def __ne__(self, other):
        return self.binary_operator_function_call(operator.ne, other=other)

    # >
    def __gt__(self, other):
        return self.binary_operator_function_call(operator.gt, other=other)

    # >=
    def __ge__(self, other):
        return self.binary_operator_function_call(operator.ge, other=other)

    # +=
    def __iadd__(self, other):
        return self.binary_operator_function_call(operator.iadd, other=other)

    # -=
    def __isub__(self, other):
        return self.binary_operator_function_call(operator.isub, other=other)

    # *=
    def __imul__(self, other):
        return self.binary_operator_function_call(operator.imul, other=other)

    # /=
    def __itruediv__(self, other):
        return self.binary_operator_function_call(operator.itruediv, other=other)

    # -(obj)
    def __neg__(self):
        return self.unary_operator_function_call(operator.neg)

    # //
    def __floordiv__(self, other):
        return self.binary_operator_function_call(operator.floordiv, other)

    def __rfloordiv__(self, other):
        return self.binary_operator_function_call(operator.floordiv, other)

    # +(obj)
    def __pos__(self):
        return self.unary_operator_function_call(operator.pos)

    # | or
    def __or__(self, other):
        return self.binary_operator_function_call(operator.or_, other)

    def __ror__(self, other):
        return self.binary_operator_function_call(operator.or_, other)

    # ^ xor
    def __xor__(self, other):
        return self.binary_operator_function_call(operator.xor, other)

    def __rxor__(self, other):
        return self.binary_operator_function_call(operator.xor, other)

    # & and
    def __and__(self, other):
        return self.binary_operator_function_call(operator.and_, other)

    def __rand__(self, other):
        return self.binary_operator_function_call(operator.and_, other)

    def dot(self, b, out):
        return self.binary_operator_function_call(self.op._op.dot, other=b)

    def sum(self, *args, **kwargs):
        return self.unary_operator_function_call(self.op._op.sum, *args, **kwargs)

    def mean(self, *args, **kwargs):
        # print("cupyTensor mean call!!")
        return self.unary_operator_function_call(self.op._op.mean, *args, **kwargs)

    def exp(self, *args, **kwargs):
        return self.unary_operator_function_call(self.op._op.exp, *args, **kwargs)

    def reshape(self, *args, **kwargs):
        return self.unary_operator_function_call(self.op._op.reshape, *args, **kwargs)

    def log(self, *args, **kwargs):
        return self.unary_operator_function_call(self.op._op.log, *args, **kwargs)

class Parameter(cupyTensor):
    def __init__(self, data):
        super(Parameter, self).__init__(data)



class mySequential(myModule):
    def __init__(self, *args):
        super(mySequential, self).__init__()

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x):
        module = self.__dict__['_modules']
        for module_key in module:
            x = module[module_key].forward(x)
        return x

    def backward(self, back):
        module = self.__dict__['_modules']
        for module_key in reversed(module):
            back = module[module_key].backward(back)
        return back
    # def backward(self, back_x):
############################### test ####################################################


# if __name__ == "__main__":
#     # operator unit test
#     operator_unit_test()
