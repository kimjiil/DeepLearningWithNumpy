import numpy as np
from typing import Callable, Any, Tuple, DefaultDict, List
from collections import OrderedDict
import cupy as cp

import operator

class cupyTensor:
    def __init__(self, data):
        self.data = data
        self.grad_fn = None
        self.requires_grad = False
        self.grad = None

        self._oper_dict = self._create_operator_dict()

    def _create_operator_dict(self):
        temp = dict()
        temp['add'] = operator.add
        temp['div'] = operator.truediv
        temp['mul'] = operator.mul
        temp['sub'] = operator.sub

        return temp

    def to(self, dev):
        if dev['device'] == "cuda":
            cp.cuda.runtime.setDevice(dev['id'])
            self.data = cp.asarray(self.data)
        elif dev['device'] == "cpu":
            self.data = cp.asnumpy(self.data)
            cp._default_memory_pool.free_all_blocks()

    def __repr__(self):
        return f"cupyTensor - shape:{self.data.shape}, grad_fn:{self.grad_fn}, data:{self.data}"

    def backward(self):
        print("tensor backward!")

    #slicing call
    def __setslice__(self, i, j, sequence):
        ...
    #slicing call
    def __getitem__(self, item):
        self.data = self.data[item]
        return self

    def operator_function_call(self, operator, other):
        _new_data = self._oper_dict[operator](self.data, other.data)
        _new = cupyTensor(_new_data)
        _new.grad_fn = f"np.{operator}()"
        return _new

    def __add__(self, other):
        # _new_data = self.data + other.data
        # _new = cupyTensor(_new_data)
        # _new.grad_fn = "np.add()"
        # return _new
        return self.operator_function_call('add', other)

    def __mul__(self, other):
        # _new_data = self.data * other.data
        # _new = cupyTensor(_new_data)
        # _new.grad_fn = "np.mul()"
        # return _new
        return self.operator_function_call('mul', other)

    def __truediv__(self, other):
        # _new_data = self.data / other.data
        # _new = cupyTensor(_new_data)
        # _new.grad_fn = "np.div()"
        # return _new
        return self.operator_function_call('div', other)

    def __sub__(self, other):
        # _new_data = self.data - other.data
        # _new = cupyTensor(_new_data)
        # _new.grad_fn = "np.sub()"
        # return _new
        return self.operator_function_call('sub', other)

class Parameter(cupyTensor):
    def __init__(self, data):
        super(Parameter, self).__init__(data)

    # def __repr__(self):
    #     return "hello parameters"

class myModule:
    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()

        self._dev = dict()
        self._dev['device'] = 'cpu'  # default
        self._dev['id'] = -1  # cpu default
        self.op = np # default

    def forward_call(self, x):
        return self.forward(x)

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
            self.op = cp
            # cp.cuda.Device(self._dev['id']).use()
            cp.cuda.runtime.setDevice(self._dev['id'])

            _parameters = self.__dict__['_parameters']
            for _param_key in _parameters:
                _parameters[_param_key].to(self._dev)

        elif "cpu" in _args: # call: to("cpu") or to(device="cpu")
            self._dev['id'] = -1
            self._dev['device'] = "cpu"
            self.op = np

            _parameters = self.__dict__['_parameters']
            for _param_key in _parameters:
                _parameters[_param_key].to(self._dev)

        elif "cuda" in _args:
            gpu_id = int(_args.split(":")[-1])

            self._dev['id'] = gpu_id
            self._dev['device'] = "cuda"
            self.op = cp
            # cp.cuda.Device(self._dev['id']).use()
            cp.cuda.runtime.setDevice(self._dev['id'])

            _parameters = self.__dict__['_parameters']
            for _param_key in _parameters:
                _parameters[_param_key].to(self._dev)
        return self

    # def _ret_lib(self):
    #     if self._dev['device'] == "cpu":
    #         return np
    #     else:
    #         cp.cuda.Device(self._dev['id']).use()
    #         return cp

    def __setattr__(self, key, value):
        if isinstance(value, myModule):
            modules = self.__dict__.get("_modules")
            modules[key] = value
            object.__setattr__(self, key, value)
        elif isinstance(value, Parameter):
            parameters = self.__dict__.get("_parameters")
            parameters[key] = value
            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)

    def __getattr__(self, item):
        return

    def add_module(self, key, module):
        self._modules[key] = module


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
############################### test ####################################################

class testModel(myModule):
    def __init__(self):
        super(testModel, self).__init__()

        self.layer1 = mySequential(
            [1,2],
            [3,4],
            [5,6]
        )


if __name__ == "__main__":
    test = testModel()
