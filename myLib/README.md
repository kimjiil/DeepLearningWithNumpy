# My Deep Learning Code

## 기본적인 기능

### Tensor 
```python
import numpy as np
from myLib.Module import myTensor

# create tensor from numpy array
np_data_a = np.array([1,2,3,4])
np_data_b = np.array([2,1,4,2])
# set device
a = myTensor(np_data_a).to(device='cuda:0')
b = myTensor(np_data_b).to(device='cuda:0')
```