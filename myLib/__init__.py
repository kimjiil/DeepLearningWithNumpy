from .Module import *
import sys
module_op = op()



def _Warpper(name, *args, **kwargs):
    if isinstance(args[0], myTensor):
        return getattr(args[0].op, name)(*args, **kwargs)
    else:
        return getattr(module_op, name)(*args, **kwargs)

def arange(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def eye(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def argmax(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def zeros(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def zeros_like(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

# ↓ numpy and cupy function wrapper
def maximum(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def max(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def where(self, *args, **kwargs):
    ...

def sqrt(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def dot(*args,  **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def matmul(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def mean(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def sum(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def exp(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def reshape(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def log(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def transpose(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)

def tile(*args, **kwargs):
    cur_func_name = sys._getframe().f_code.co_name
    return _Warpper(cur_func_name, *args, **kwargs)