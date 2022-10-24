import numpy as np
from typing import Callable, Any, Tuple, DefaultDict, List
from collections import OrderedDict
import cupy as cp

class myModule:
    def __init__(self):
        self._module = OrderedDict()

        self._dev = dict()
        self._dev['device'] = 'cpu'  # default
        self._dev['id'] = -1  # cpu default
        self.op = np # default

    def forward_call(self, x):
        return self.forward(x)

    __call__: Callable[..., Any] = forward_call

    def backward(self):
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def parameters(self):
        pass

    def children(self):
        module_list = self.__dict__['_module']
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
            param = args[0]
        elif kwargs:
            if "device" in kwargs:
                param = kwargs['device']
            else:
                raise "unvalid parameter key"
        else:
            param = None


        for child in self.children():
            child.to(*args, **kwargs)

        if param == None: # call: to()
            self._dev['id'] = 0
            self._dev['device'] = "cuda"
            self.op = cp
            # cp.cuda.Device(self._dev['id']).use()
            cp.cuda.runtime.setDevice(self._dev['id'])
        elif "cpu" in param: # call: to("cpu") or to(device="cpu")
            self._dev['id'] = -1
            self._dev['device'] = "cpu"
            self.op = np
        elif "cuda" in param:
            gpu_id = int(param.split(":")[-1])

            self._dev['id'] = gpu_id
            self._dev['device'] = "cuda"
            self.op = cp
            # cp.cuda.Device(self._dev['id']).use()
            cp.cuda.runtime.setDevice(self._dev['id'])
        return self

    # def _ret_lib(self):
    #     if self._dev['device'] == "cpu":
    #         return np
    #     else:
    #         cp.cuda.Device(self._dev['id']).use()
    #         return cp

    def __setattr__(self, key, value):
        if isinstance(value, myModule):
            modules = self.__dict__.get("_module")
            modules[key] = value
            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)

    def __getattr__(self, item):
        return

    def add_module(self, key, module):
        self._module[key] = module


class mySequential(myModule):
    def __init__(self, *args):
        super(mySequential, self).__init__()

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x):
        module = self.__dict__['_module']
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
