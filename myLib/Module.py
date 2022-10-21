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

    def to(self, *args, **kargs):
        '''
        to("cuda:0") or to("cpu")
        '''

        print()
        if len(args) > 1 or len(kargs) > 1:
            raise "too many parameters"

        if len(args) == 0 and len(kargs) == 0:
            self._dev['id'] = 0
            self._dev['device'] = "cuda"
            self.op = cp
            cp.cuda.Device(self._dev['id']).use()
        # elif if "cuda" in args or "cuda" in kargs['device']:
        #     self._dev['id'] = int(device.split(":")[-1])
        #     self._dev['device'] = device.split(":")[0]
        #     self.op = cp
        #     cp.cuda.Device(self._dev['id']).use()
        # elif device == None:
        #     self._dev['id'] = 0
        #     self._dev['device'] = "cuda"
        #     self.op = cp
        #     cp.cuda.Device(self._dev['id']).use()
        # else:
        #     self._dev['id'] = -1
        #     self._dev['device'] = device
        #     self.op = np

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
