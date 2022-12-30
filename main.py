def print_memory_usage():
    print("used - ",cp._default_memory_pool.used_bytes() / 1e9, 'gb')
    print("total - ",cp._default_memory_pool.total_bytes() / 1e9, 'gb')

from myLib.Module import myModule, mySequential, myTensor
from myLib.Layer import *
from myLib.LossFunc import *
from myLib.Optimizer import *
import myLib


import numpy as np
import cupy as cp

from functools import wraps

from torchvision.datasets import MNIST
import numpy as np

download_path = "./MNIST_Datset"
train_dataset = MNIST(download_path, train=True, download=True)
valid_dataset = MNIST(download_path, train=False, download=True)

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
    import time

    class my_model(myModule):
        def __init__(self):
            super(my_model, self).__init__()
            self.maxpool = MaxPool2d(3, 2)
            self.flatten = Flatten()
            self.hidden_layers = mySequential(
                Linear(in_features=169, out_features=312, bias=True),
                ReLU(),
                Linear(in_features=312, out_features=128, bias=True),
                ReLU()
            )
            self.classifier = Linear(in_features=128, out_features=10, bias=True)
            self.sigmoid = Sigmoid()
            print()

        def forward(self, x):
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.hidden_layers(x)
            x = self.classifier(x)
            x = self.sigmoid(x)
            return x

    model = my_model()
    model.to(device="cuda:0")

    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = MSELoss()
    epoch_size = 100
    batch_size = 144
    total_size = len(train_dataset)
    for epoch_i in range(epoch_size):
        start_time = time.time()
        step_size = int(total_size / batch_size)
        loss_sum = []
        for step_i in range(step_size):
            input_data = train_dataset.data[step_i*batch_size:(step_i+1)*batch_size].numpy().reshape(batch_size, 1, 28, 28)
            targets = train_dataset.targets[step_i*batch_size:(step_i+1)*batch_size].numpy()
            targets_one_hot = np.eye(10)[targets]

            input_data = myTensor(input_data).to(device="cuda:0")
            targets = myTensor(targets_one_hot).to(device="cuda:0")

            optimizer.zero_grad()
            pred = model(input_data)

            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

            # print(step_i, loss)
            loss_sum.append(loss.data)
        end_time = time.time()
        print(epoch_i, sum(loss_sum) / len(loss_sum), end_time - start_time)
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
    temp = myTensor(a)#.to('cuda:0')
    # temp = temp[:, :, np.newaxis]
    temp[0,0] = 3.5
    test = testModel()#.to('cuda:0')
    test(temp)

def test02():
    class my_model(myModule):
        def __init__(self):
            super(my_model, self).__init__()

            self.convlayers = mySequential(
                Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False),
                ReLU(),
                MaxPool2d(kernel_size=2, stride=2, padding=0),
                Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
                ReLU(),
                MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
            self.flatten = Flatten()
            self.hidden_layers = mySequential(
                Linear(in_features=1152, out_features=256, bias=True),
                ReLU(),
            )
            self.classifier = Linear(in_features=256, out_features=10, bias=True)
            self.sigmoid = Sigmoid()


        def forward(self, x):
            x = self.convlayers(x)
            x = self.flatten(x)
            x = self.hidden_layers(x)
            x = self.classifier(x)
            x = self.sigmoid(x)
            return x

    model = my_model()
    model.to(device="cuda:0")

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = MSELoss()
    epoch_size = 100
    batch_size = 144
    total_size = len(train_dataset)
    for epoch_i in range(epoch_size):
        start_time = time.time()
        step_size = int(total_size / batch_size)
        loss_sum = []
        for step_i in range(step_size):
            input_data = train_dataset.data[step_i * batch_size:(step_i + 1) * batch_size].numpy().reshape(batch_size,
                                                                                                           1, 28, 28) /255.
            targets = train_dataset.targets[step_i * batch_size:(step_i + 1) * batch_size].numpy()
            targets_one_hot = np.eye(10)[targets]

            input_data = myTensor(input_data).to(device="cuda:0")
            targets = myTensor(targets_one_hot).to(device="cuda:0")

            optimizer.zero_grad()
            pred = model(input_data)

            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

            # print(step_i, loss)
            loss_sum.append(loss.data)
        end_time = time.time()
        print(epoch_i, "loss", sum(loss_sum) / len(loss_sum), "runtime", end_time - start_time)

        step_size = int(len(valid_dataset.data) / batch_size)

        acc_list = []
        for step_i in range(step_size):
            input_data = valid_dataset.data[step_i * batch_size:(step_i + 1) * batch_size].numpy().reshape(batch_size,1, 28, 28) / 255.
            targets = valid_dataset.targets[step_i * batch_size:(step_i + 1) * batch_size].numpy()

            input_data = myTensor(input_data).to(device="cuda:0")
            targets = myTensor(targets).to(device="cuda:0")
            pred = model(input_data)
            pred_label = myLib.argmax(pred, 1)

            correct = (targets == pred_label).numpy()
            acc_list.extend(correct)

        print(f"accuracy {np.mean(acc_list)}")
test02()

# test00()
# test01()
