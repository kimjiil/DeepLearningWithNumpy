# My Deep Learning Code

## 기본적인 기능

### create tensor 
```python
import numpy as np
from myLib.Module import myTensor

# create tensor from numpy array
np_data_a = np.array([1,2,3,4])
np_data_b = np.array([2,1,4,2])
# set device
a = myTensor(np_data_a).to(device='cuda:0')
b = myTensor(np_data_b).to(device='cuda:0')

c = a + b

# tensor to numpy
a_np = a.numpy()
b_np = b.numpy()
```

### create Model & training

```python
from myLib.Module import myModule, mySequential, myTensor
from myLib.Layer import *
from myLib.LossFunc import *
from myLib.Optimizer import *

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
```


## 구현 목록

### [Module](./Module.py)

[[구현 상세]](https://kimjiil.github.io/ai/ml%20study/DeepLearningWithNumpy/#mymodule-class)

- Sequential
- Parameter
- Tensor

### [Layer](./Layer.py)

[[구현 상세]](https://kimjiil.github.io/ai/ml%20study/DeepLearningWithNumpy/#layer)

- Sigmoid
- ReLU
- Linear
- MaxPool2d
- Conv2d 
- Flatten
- BatchNorm2d (* 미구현)
- AvgPool (* 미구현)
- Dropout (* 미구현)

### [Optimizer](./Optimizer.py)

[[구현 상세]](https://kimjiil.github.io/ai/ml%20study/DeepLearningWithNumpy/#opimizer)

- Adam
- AdamW (* 미구현)
- Stochatic Gradient Descent (* 미구현)

## Refrence
- https://github.com/pytorch/pytorch