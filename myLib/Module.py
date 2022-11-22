import numpy as np
from typing import Callable, Any, Tuple, DefaultDict, List
from collections import OrderedDict
import cupy as cp

import operator
from functools import wraps

decorator_active = False

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
        if args[0].backward_prev:
            self.backward_fn = args[0].backward_prev
        else:
            self.backward_fn = args[0].backward_fn
        args[0].backward_prev = self.backward

        return self.forward(*args, **kwargs)

    __call__: Callable[..., Any] = forward_call

    def _get_op(self):
        return self.op

    def backward(self, *args, **kwargs):
        # print("myModule backward!!!")
        if hasattr(self, '_backward'):
            temp = self._backward(*args, **kwargs)
            self.backward_fn(temp, *args[1:], **kwargs)
        else:
            self.backward_fn(*args, **kwargs)

    def eval(self):
        pass

    def train(self):
        pass

    def parameters(self):
        # return trainable parameters iterator
        for child in self.children():
            params = child.parameters() #여기서 generator 리턴받음
            for p in params:
                yield p

        for param in self._parameters:
            yield self._parameters[param]

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
        if isinstance(value, myParameter):
            parameters = self.__dict__.get("_parameters")
            parameters[key] = value
            # super(myModule, self).__setattr__(key, value)
            object.__setattr__(self, key, value)

        elif isinstance(value, myModule):
            modules = self.__dict__.get("_modules")
            modules[key] = value
            # super(myModule, self).__setattr__(key, value)
            object.__setattr__(self, key, value)

        else:
            # super(myModule, self).__setattr__(key, value)
            object.__setattr__(self, key, value)

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

    def unary_function_wrapper(self, operator, *args, **kwargs):
        temp_prev = None
        if isinstance(args[0], myTensor):
            temp_x1 = args[0].data
            # temp_fn = x1.backward_fn
            if args[0].backward_prev:
                temp_prev = args[0].backward_prev
            temp_op = args[0].op
        else:
            temp_x1 = args[0]

        new_temp = operator(temp_x1, *args[1:], **kwargs)
        new = myTensor(new_temp)
        new.grad_fn = f"<{operator}>"
        new.backward_fn = temp_prev
        new.backward_prev = temp_prev
        new.op = temp_op
        return new

    def binary_function_wrapper(self, operator, *args, **kwargs):
        temp_prev = None
        if isinstance(args[0], myTensor):
            temp_x1 = args[0].data
            # temp_fn = x1.backward_fn
            if args[0].backward_prev:
                temp_prev = args[0].backward_prev
            temp_op = args[0].op
        else:
            temp_x1 = args[0]

        if isinstance(args[1], myTensor):
            temp_x2 = args[1].data
            # temp_fn = x2.backward_fn
            if args[1].backward_prev:
                temp_prev = args[1].backward_prev
            temp_op = args[1].op
        else:
            temp_x2 = args[1]

        new_temp = operator(temp_x1, temp_x2, *args[2:], **kwargs)
        new = myTensor(new_temp)
        new.grad_fn = f"<{operator}>"
        new.backward_fn = temp_prev
        new.backward_prev = temp_prev
        new.op = temp_op
        return new

    def zeros_like(self, *args, **kwargs):
        return self.unary_function_wrapper(self._op.zeros_like, *args, **kwargs)

    # ↓ numpy and cupy function wrapper
    def maximum(self, *args, **kwargs):
        return self.binary_function_wrapper(self._op.maximum, *args, **kwargs)

    def where(self, *args, **kwargs):
        ...

    def sqrt(self, *args, **kwargs):
        return self.unary_function_wrapper(self._op.sqrt, *args, **kwargs)

    def dot(self, *args,  **kwargs):
        return self.binary_function_wrapper(self._op.dot, *args, **kwargs)

    def matmul(self, *args, **kwargs):
        return self.binary_function_wrapper(self._op.matmul, *args, **kwargs)

    @print_decorator("operator", "mean", decorator_active)
    def mean(self, *args, **kwargs):
        return self.unary_function_wrapper(self._op.mean, *args, **kwargs)

    @print_decorator("operator", "sum", decorator_active)
    def sum(self, *args, **kwargs):
        return self.unary_function_wrapper(self._op.sum, *args, **kwargs)

    @print_decorator("operator", "exp", decorator_active)
    def exp(self, *args, **kwargs):
        return self.unary_function_wrapper(self._op.exp, *args, **kwargs)

    def reshape(self, *args, **kwargs):
        return self.unary_function_wrapper(self._op.reshape, *args, **kwargs)

    def log(self, *args, **kwargs):
        return self.unary_function_wrapper(self._op.log, *args, **kwargs)

    def transpose(self, *args, **kwargs):
        return self.unary_function_wrapper(self._op.transpose, *args, **kwargs)

class myTensor(myModule):
    def __init__(self, data):
        super(myTensor, self).__init__()
        self.data = data
        self.shape = self.data.shape
        self.grad_fn = None
        self.requires_grad = False
        self.grad = None
        self.backward_fn = self.backward
        self.backward_prev = None
        # self._op = np

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
        return f"myTensor - shape:{self.data.shape}, grad_fn:{self.grad_fn}, data:{self.data}"

    def backward(self, *args, **kwargs):
        # print("tensor backward!")
        if self.backward_fn == self.backward:
            # print("this is start Tensor")
            ...
        else:
            # print("this is end Tensor")
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

    def binary_operator_function_call(self, operator, other, reverse=False):
        if isinstance(other, myTensor):
            temp_data = other.data
            if other.backward_prev:
                temp_prev = other.backward_prev
        else:
            temp_data = other

        if self.backward_prev:
            temp_prev = self.backward_prev

        if reverse:
            _new_data = operator(temp_data, self.data)
        else:
            _new_data = operator(self.data, temp_data)
        _new = myTensor(_new_data)
        _new.grad_fn = f"{operator}"
        _new.op = self.op
        _new.backward_fn = temp_prev
        _new.backward_prev = temp_prev
        return _new

    def unary_operator_function_call(self, *args, **kwargs):
        # args[0]은 항상 operator로 들어옴
        # print("unary operator function call!!")
        operator = args[0]

        _new_data = operator(self.data, *args[1:], **kwargs)
        _new = myTensor(_new_data)
        _new.grad_fn = f"{operator}"
        _new.backward_fn = self.backward_prev
        _new.backward_prev = self.backward_prev
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
        return self.binary_operator_function_call(operator.truediv, other=other, reverse=True) #서순이 문제가됨.

    def __sub__(self, other):
        return self.binary_operator_function_call(operator.sub, other=other)

    def __rsub__(self, other):
        return self.binary_operator_function_call(operator.sub, other=other, reverse=True) #서순이 문제가됨.

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
        return self.binary_operator_function_call(operator.floordiv, other, reverse=True)

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

    def __matmul__(self, other):
        print('test')

    def matmul(self, *args, **kwargs):
        print("test")

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

class myParameter(myTensor):
    def __init__(self, data):
        super(myParameter, self).__init__(data)
        self._opt_file = dict()

    def update_parameter(self, tensor):
        self.data = tensor.data

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

    def backward(self, *args, **kwargs):
        back = args[0]
        module = self.__dict__['_modules']
        for module_key in reversed(module):
            back = module[module_key].backward(back)

        self.backward_fn(back)

############################### test ####################################################


# if __name__ == "__main__":
#     # operator unit test
#     operator_unit_test()
