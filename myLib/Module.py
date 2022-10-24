import numpy as np
from typing import Callable, Any, Tuple, DefaultDict, List
from collections import OrderedDict
import cupy as cp

class cupyTensor:
    def __init__(self, data):
        self.data = data
        self.grad_fn = None
        self.required_grad = False

    def to(self, dev):
        if dev['device'] == "cuda":
            cp.cuda.runtime.setDevice(dev['id'])
            self.data = cp.asarray(self.data)
        elif dev['device'] == "cpu":
            self.data = cp.asnumpy(self.data)
            cp._default_memory_pool.free_all_blocks()

    def __repr__(self):
        return f"cupyTensor: {self.data.shape}, {self.grad_fn}, {self.data}"

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
