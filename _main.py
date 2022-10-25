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

class my_model(myModule):
    def __init__(self):
        super(my_model, self).__init__()

        self.ReLU_seq = mySequential(
            Linear(in_features=20, out_features=10, bias=True),
            ReLU(),
        )

        print()

    def forward(self, x):
        x = self.ReLU_seq(x)
        return x

model = my_model()
model.to(device="cuda:0")

############
a = np.array([[1,2,3,4,5]])
b = np.transpose(a)

cp_a = cupyTensor(a)
cp_b = cupyTensor(b)
cp_a[:,:]
ddd = cp_a + cp_a

mul_cp_c = cp_a * cp_b
mul_c = a * b
mul_cp_d = cp_b * cp_a
mul_d = b * a
mul_cp_e = cp_a * cp_a
mul_e = a * a

div_cp_c = cp_a / cp_b
div_cp_d = cp_b / cp_a
div_cp_e = cp_a / cp_a
print()

##########


input = np.random.randn(5, 20)
input_cuda = cp.asarray(input)


label = np.zeros((5, 10))
label[0, 5] = 1
label[1, 6] = 1
label[2, 3] = 1
label[3, 4] = 1
label[4, 8] = 1
label_cuda = cp.asarray(label)
pred = model(input_cuda)


criterion = testloss()
loss = criterion(pred, label_cuda)
loss.backward()

print("main!!!")





model.to("cpu")
c = model(b)
print("cpu - " , c)