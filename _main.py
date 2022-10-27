def print_memory_usage():
    print("used - ",cp._default_memory_pool.used_bytes() / 1e9, 'gb')
    print("total - ",cp._default_memory_pool.total_bytes() / 1e9, 'gb')


'''
Model을 module에서 선언하고 외부에서 __call__ 할경우 자동으로 forward함수를 타게됨
이때, 각각의 레이어에 대해서도 똑같이 __call__ 할경우 -> forward()

model에서 .backward()를 호출할경우 gradient 계산

optimizer에서 .step() 호출하경우 gradient update
optimizer 선언부에서 model.paprameters()


sequential class는 module class에서 받아와서 module의 add_module을 실행해서 Layer를 추가함.

'''

from myLib.Module import myModule, mySequential, cupyTensor
from myLib.Layer import ReLU, Linear
from myLib.LossFunc import testloss

import numpy as np
import cupy as cp

from functools import wraps

def function_control(active):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if active:
                func(*args, **kwargs)
            return
        return wrapper
    return decorator

test00_act = True
test01_act = False

# @function_control(test00_act)
def test00():
    class my_model(myModule):
        def __init__(self):
            super(my_model, self).__init__()

            self.ReLU_seq = mySequential(
                Linear(in_features=20, out_features=10, bias=True),
                ReLU(),
            )
            self.Linear_layer = Linear(in_features=10, out_features=10, bias=True)
            print()

        def forward(self, x):
            x = self.ReLU_seq(x)
            x = self.Linear_layer(x)
            return x

    model = my_model()
    model.to(device="cuda:0")

    input = np.random.randn(5, 20)
    # cp.cuda.runtime.setDevice()
    # input_cuda_test = cp.asarray(input)
    # mul_test = 2 * input_cuda_test
    input_cuda = cupyTensor(input).to('cuda:0')
    label = np.zeros((5, 10))
    label[0, 5] = 1
    label[1, 6] = 1
    label[2, 3] = 1
    label[3, 4] = 1
    label[4, 8] = 1
    label_cuda = cupyTensor(label).to("cuda:0")
    pred = model(input_cuda)


    criterion = testloss()
    loss = criterion(pred, label_cuda)
    loss.backward()

    print()

@function_control(test01_act)
def test01():
    # 연산자 테스트 함수
    from myLib.Layer import operator_test_layer
    class testModel(myModule):
        def __init__(self):
            super(testModel, self).__init__()

            self.layer1 = mySequential(
                operator_test_layer()
            )

        def forward(self, x):
            x = self.layer1(x)
            return x

    a = np.array([[1,2], [4,6]], dtype=np.float32)
    b = np.array([[2,1], [2,3]], dtype=np.float32)
    a /= b
    temp = cupyTensor(a)#.to('cuda:0')
    # temp = temp[:, :, np.newaxis]
    temp[0,0] = 3.5
    test = testModel()#.to('cuda:0')
    test(temp)

test00()
test01()
