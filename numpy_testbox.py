import numpy as np
from typing import Callable, Any

class testclass():
    def _init(self):
        self.text = "hello"

    __init__ : Callable[..., Any] = _init

    def _say_hello(self, text: str):
        print(text)

    __call__ : Callable[..., Any] = _say_hello



model = testclass()

model((11, 1))

from typing import List, Dict

a: List[int] = (1, 2, 3)

print(a)

b = [1,2,3,4,5, ... , 9999999999]

print(b)
