```python
from torchvision.datasets import MNIST
import numpy as np

download_path = "./MNIST_Datset"
train_dataset = MNIST(download_path, train=True, download=True)
valid_dataset = MNIST(download_path, train=False, download=True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./MNIST_Datset/MNIST/raw/train-images-idx3-ubyte.gz



      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ./MNIST_Datset/MNIST/raw/train-images-idx3-ubyte.gz to ./MNIST_Datset/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./MNIST_Datset/MNIST/raw/train-labels-idx1-ubyte.gz



      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ./MNIST_Datset/MNIST/raw/train-labels-idx1-ubyte.gz to ./MNIST_Datset/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./MNIST_Datset/MNIST/raw/t10k-images-idx3-ubyte.gz



      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ./MNIST_Datset/MNIST/raw/t10k-images-idx3-ubyte.gz to ./MNIST_Datset/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST_Datset/MNIST/raw/t10k-labels-idx1-ubyte.gz



      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ./MNIST_Datset/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST_Datset/MNIST/raw
    


    /home/kji/anaconda3/envs/py39_0/lib/python3.9/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448255797/work/torch/csrc/utils/tensor_numpy.cpp:180.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)



```python
def _forward(x, layers):
    out = x
    for idx, layer in enumerate(layers):
        out = layer.forward(out)

    return out

def _backward(back_propagation, layers):
    back = back_propagation
    for layer in reversed(layers):
        back = layer.backward(back)

def one_hot(y):
    n = len(y)

    one_hot_y = np.zeros((n, 10))
    idx = [i for i in range(n)]
    one_hot_y[idx, y] = 1
    return one_hot_y

def MSE_Loss(output, y):
    return np.sum(((output - y) ** 2) / 2)

def Cross_Entropy_Loss(output, y):
    return np.sum(-(y * np.log(output)))

def softmax(output):
    exp_output = np.exp(output)
    exp_sum = np.sum(exp_output, axis=1)
    return exp_output / exp_sum[:, np.newaxis]
```


```python
class ReLULayer_np():
    '''
        들어온 신호가 있는 pixel만 backprop를 전달함
    '''
    def __init__(self):
        self._prev_input = None

    def forward(self, x):
        self._prev_input = np.array(x, copy=True)
        return np.maximum(0, x)

    def backward(self, _back_gradient):
        mask = self._prev_input > 0
        _back = _back_gradient * mask
        return _back

    def get_weight(self):
        return None

class LinearLayer_np():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self._linear_weight = np.random.uniform(low=(-np.sqrt(6/input_dim)),high=np.sqrt(6/input_dim),size=(input_dim, output_dim)) #np.random.randn(input_dim, output_dim) * 0.1
        self._linear_bias = np.random.randn(output_dim) * 0.01

        self._prev_input = None
        self._update_w = None
        self._update_b = None

    def get_weight(self):
        return self._update_w, self._linear_weight, self._update_b, self._linear_bias

    def set_weight(self, w, b):
        self._linear_weight = w
        self._linear_bias = b

    def forward(self, x):
        self._prev_input = np.array(x, copy=True)

        return np.dot(x, self._linear_weight) + self._linear_bias


    def backward(self, _back_gradient):
        output = np.zeros((self._prev_input.shape))

        self._update_w = np.zeros((self._linear_weight.shape))
        self._update_b = np.zeros((self._linear_bias.shape))
        self._update_w += np.mean(self._prev_input[:, :, np.newaxis] * _back_gradient[:, np.newaxis, :], axis=0)

        self._update_b += np.mean(_back_gradient, axis=0)
        output += np.sum(self._linear_weight[np.newaxis, :, :] * _back_gradient[:, np.newaxis, :], axis=-1)
        return output


class MaxPoolLayer_np():
    def __init__(self, kernel_size=2, stride=1, padding=0):

        self._prev_input = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._mask_coord = dict()

    def forward(self, x):
        self._prev_input = np.array(x, copy=True)
        N, H, W, C = x.shape

        padding_x = np.zeros((N, H + 2 * self.padding, W + 2 * self.padding, C))
        padding_x[:, self.padding:(self.padding + H), self.padding:(self.padding + W), :] = x

        self.h_stride = (H+2*self.padding - self.kernel_size) // self.stride + 1
        self.w_stride = (W+2*self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((N, self.h_stride, self.w_stride, C))
        self.max_output_mask = np.zeros_like(padding_x)

        for j in range(self.h_stride):
            start_j_index = j * self.stride
            end_j_index = j * self.stride + self.kernel_size
            for i in range(self.w_stride):
                start_i_index = i * self.stride
                end_i_index = i * self.stride + self.kernel_size
                mask = padding_x[:, start_j_index:end_j_index, start_i_index:end_i_index, :]
                max_value = np.max(mask, axis=(1, 2))
                output[:, j, i, :] = max_value
                self.max_output_mask[:, start_j_index:end_j_index, start_i_index:end_i_index, :] = max_value[:, np.newaxis, np.newaxis, :]
                self._mask_coord[(j, i)] = np.array(self.max_output_mask[:, start_j_index:end_j_index, start_i_index:end_i_index, :], copy=True)
        return output

    def backward(self, _back_gradient):
        output = np.zeros((self._prev_input.shape))
        for j in range(self.h_stride):
            start_j_index = j * self.stride
            end_j_index = j * self.stride + self.kernel_size
            for i in range(self.w_stride):
                start_i_index = i * self.stride
                end_i_index = i * self.stride + self.kernel_size
                mask = self._mask_coord[(j, i)]
                temp = self._prev_input[:, start_j_index:end_j_index, start_i_index:end_i_index, :] >= mask
                order_mask = np.arange(1, self.kernel_size * self.kernel_size + 1).reshape(self.kernel_size, self.kernel_size)[np.newaxis, :, :, np.newaxis]
                temp_mask = order_mask * temp
                max_value = np.max(temp_mask, axis=(1, 2))
                ori_temp_mask = np.array(temp_mask, copy=True)
                temp_mask[:, :, :, :] = max_value[:, np.newaxis, np.newaxis, :]
                real_mask = temp_mask <= ori_temp_mask
                output[:, start_j_index:end_j_index, start_i_index:end_i_index, :] += mask * real_mask
        return output

    def get_weight(self):
        return None


class FlattenLayer_np():
    # input N H W C
    # output N x (H * W * C)
    def __init__(self):
        self._prev_input = None

    def forward(self, x):
        N, H, W, C = x.shape
        self._prev_input = np.array(x, copy=True)
        output = x.reshape(N, H * W * C)
        return output

    def backward(self, _back_gradient):
        N, H, W, C = self._prev_input.shape
        return _back_gradient.reshape(N, H, W, C)

    def get_weight(self):
        return None



class Adam():
    def __init__(self, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8):
        self.step = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps

        self._m = dict()
        self._v = dict()


    def update(self, layers):
        if len(self._m) == 0 or len(self._v) == 0:
            self._init_mv(layers)

        for idx, layer in enumerate(layers):
            gradient = layer.get_weight
            if gradient() is None:
                continue
            dw, w, db, b = gradient()
            dw_key = f"dw{idx}"
            db_key = f"db{idx}"
            self._m[dw_key] = self.beta1 * self._m[dw_key] + (1 - self.beta1) * dw
            self._m[db_key] = self.beta1 * self._m[db_key] + (1 - self.beta1) * db

            self._v[dw_key] = self.beta2 * self._v[dw_key] + (1 - self.beta2) * (dw ** 2)
            self._v[db_key] = self.beta2 * self._v[db_key] + (1 - self.beta2) * (db ** 2)

            bias_correction1_w = self._m[dw_key] / (1 - (self.beta1 ** self.step))
            bias_correction1_b = self._m[db_key] / (1 - (self.beta1 ** self.step))

            bias_correction2_w = self._v[dw_key] / (1 - (self.beta2 ** self.step))
            bias_correction2_b = self._v[db_key] / (1 - (self.beta2 ** self.step))

            next_w = w - self.lr * bias_correction1_w / (np.sqrt(bias_correction2_w) + self.eps)
            next_b = b - self.lr * bias_correction1_b / (np.sqrt(bias_correction2_b) + self.eps)

            layer.set_weight(w=next_w, b=next_b)


    def _init_mv(self, layers):
        for idx, layer in enumerate(layers):

            gradient = layer.get_weight
            if gradient() is None:
                continue

            dw, w, db, b = gradient()
            dw_key = f"dw{idx}"
            db_key = f"db{idx}"

            self._m[dw_key] = np.zeros_like(dw)
            self._m[db_key] = np.zeros_like(db)

            self._v[dw_key] = np.zeros_like(dw)
            self._v[db_key] = np.zeros_like(db)

class ConvLayer_np():
    '''
        input : N x C(input_dim) x H x W
        output : N x C(output_dim) x h x w

        kernel_weight : input_dim x output_dim x kernel_size x kernel_size
    '''
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._init_weight()

        self._prev_input = None
        self._update_w = None
        self._update_b = None

    def get_weight(self):
        return self._update_w, self.kernel_weight, self._update_b, self.kernel_bias

    def set_weight(self, w, b):
        self.kernel_weight = w
        self.kernel_bias = b

    def _init_weight(self):
        self.kernel_weight = np.random.uniform(low=(-np.sqrt(6/self.input_dim)),
                                               high=np.sqrt(6/self.input_dim),
                                               size=(self.kernel_size, self.kernel_size, self.input_dim, self.output_dim))
        # self.kernel_weight = np.random.randn(self.kernel_size, self.kernel_size,  self.input_dim, self.output_dim) * 0.1
        self.kernel_bias = np.random.randn(self.output_dim) * 0.01

    def forward(self, x):
        N, H, W, C = x.shape
        self._prev_input = np.array(x, copy=True) #deep copy

        w_stride = (W + self.padding * 2 - self.kernel_size) // self.stride + 1
        h_stride = (H + self.padding * 2 - self.kernel_size) // self.stride + 1

        output = np.zeros((N,  h_stride, w_stride, self.output_dim))
        for idx in range(N):
            for h_i in range(h_stride):
                for w_i in range(w_stride):
                    padding_x = np.zeros((H + self.padding * 2, W + self.padding * 2, C))
                    padding_x[self.padding:self.padding+H, self.padding:self.padding+W, :] = x[idx, :, :]
                    conv_img = padding_x[ h_i * self.stride:(h_i * self.stride + self.kernel_size), w_i*self.stride:(w_i * self.stride + self.kernel_size), :]
                    for out_dim_i in range(self.output_dim):
                        conv_kernel = self.kernel_weight[:, :, :, out_dim_i]
                        _conv = conv_img * conv_kernel
                        _conv_weight = np.sum(_conv)
                        output[idx, h_i, w_i, out_dim_i] = _conv_weight

        return output + self.kernel_bias

    def backward(self, _back_gradient):
        # 들어온 gradient에 현재 픽셀에 관련된 값을 다음 gradient에 더함
        # batch size만큼 평균을 내줌

        # backpropagation gradient
        N, H, W, C = self._prev_input.shape
        output = np.zeros((N, H+self.padding*2, W+self.padding*2, C))
        padding_prev_input = np.zeros(output.shape)
        padding_prev_input[:, self.padding:self.padding+H, self.padding:self.padding+W, :] = self._prev_input[:, :, :, :]
        self._update_w = np.zeros(self.kernel_weight.shape)
        self._update_b = np.zeros(self.kernel_bias.shape)
        '''
            커널에 backgradient를 곱한것을 뒤로 전달
        '''
        batch_n, height_out, width_out, out_dim = _back_gradient.shape
        weight_h, weight_w, input_dim, output_dim = self.kernel_weight.shape

        for col_i in range(width_out): # column
            output_i_start = col_i * self.stride
            output_i_end = output_i_start + weight_w
            for row_j in range(height_out): # row
                output_j_start = row_j * self.stride
                output_j_end = output_j_start + weight_h

                for dim_k in range(out_dim):
                    # output dim [256,3,3,2]
                    output[:, output_i_start:output_i_end, output_j_start:output_j_end, :] += (_back_gradient[:, np.newaxis, col_i:col_i+1, row_j:row_j+1, dim_k:dim_k+1] * \
                                                                                           self.kernel_weight[np.newaxis, :, :, :, dim_k:dim_k+1]).squeeze(axis=-1) # [256,1(new),1,1,1] * [1(new),3,3,2,1] => [256,3,3,2,1]

                    # update_w dim [3,3,2,4] / back_gradient dim [256,3,3,4] / _prev_input dim [256, 12, 12, 2] => [256, 3, 3, 2]
                    self._update_w[:, :, :, dim_k:dim_k+1] += (_back_gradient[:, col_i:col_i+1, row_j:row_j+1, np.newaxis, dim_k:dim_k+1]*
                                                       padding_prev_input[:, output_i_start:output_i_end, output_j_start:output_j_end, :, np.newaxis]).sum(axis=0) / batch_n
                    # [3,3,2,4] += [256,3,3,1(new),4] * [256,3,3,2,1(new)] => [256,3,3,2,4]

                    self._update_b[dim_k] += np.sum(_back_gradient[:, col_i:col_i+1, row_j:row_j+1, np.newaxis, dim_k:dim_k+1]) / batch_n
        return output

class LogSoftMax_np():
    def __init__(self):
        self._prev_input = None

    def forward(self, x):
        # x : N x class_num

        self._prev_input = np.array(x, copy=True)
        _exp_input = np.exp(x)
        _exp_sum = np.sum(_exp_input, axis=1)[:, np.newaxis]

        output = _exp_input / _exp_sum
        # output = np.log(output)
        # self._output = np.array(output, copy=True)
        return output

    def backward(self, _back_gradient):
        return _back_gradient

    def get_weight(self):
        return None
```


```python
layers = [
    # 28 x 28 x 1(Dim)
    ConvLayer_np(input_dim=1, output_dim=16, kernel_size=5, stride=1, padding=2),
    ReLULayer_np(),
    # 28 x 28 x 8
    MaxPoolLayer_np(kernel_size=2, stride=2, padding=0),
    # 14 x 14 x 8
    ConvLayer_np(input_dim=16, output_dim=32, kernel_size=3, stride=1, padding=0),
    ReLULayer_np(),
    # 12 x 12 x 8
    MaxPoolLayer_np(kernel_size=2, stride=2, padding=0),
    # 6 x 6 x 8
    FlattenLayer_np(),
    LinearLayer_np(input_dim=1152, output_dim=256),
    ReLULayer_np(),
    LinearLayer_np(input_dim=256, output_dim=10),
    LogSoftMax_np()
]
```


```python
optimizer = Adam(lr=0.0001)
LossFunc = None

trainX = train_dataset.data.numpy()
trainY = train_dataset.targets.numpy()

testX = valid_dataset.data.numpy()
testY = valid_dataset.targets.numpy()

total_epoch = 20
batch_size = 100

trainX = trainX[:60000] / 255.0 
trainY = trainY[:60000]

testX = testX[:10000]
testY = testY[:10000]
```


```python
import tqdm

for epoch in range(total_epoch):
    step = int(len(trainX) / batch_size)
    loss_list = []
    for step_i in tqdm.tqdm(range(step), ncols=100, desc=f"epoch {epoch}"):

        X = trainX[step_i*batch_size:(step_i+1)*batch_size]
        Y = trainY[step_i*batch_size:(step_i+1)*batch_size]
        one_hot_Y = one_hot(Y)
        X = X[:, :, :, np.newaxis]
        output = _forward(X, layers)
        back_propagation = output - one_hot_Y
        # loss = MSE_Loss(output, one_hot_Y)
        loss = Cross_Entropy_Loss(output, one_hot_Y)
        loss_list.append(loss)
        # print(loss)
        _backward(back_propagation, layers)

        optimizer.update(layers=layers)

    TEST_X = testX[:, :, :, np.newaxis] / 255.0
    # TEST_Y = one_hot(testY)

    output = _forward(TEST_X, layers)
    # softmax_output = softmax(output)
    acc = np.mean(testY == np.argmax(output, axis=1))
    Text = f'epoch:{epoch} / loss:{np.mean(loss_list)} / Acc:{acc*100.0}%'
    print(f'epoch:{epoch} / loss:{np.mean(loss_list)} / Acc:{acc*100.0}%')
```

    epoch 0: 100%|██████████████████████████████████████████████████| 600/600 [2:13:37<00:00, 13.36s/it]


    epoch:0 / loss:63.33228858164571 / Acc:91.58%


    epoch 1: 100%|██████████████████████████████████████████████████| 600/600 [2:08:28<00:00, 12.85s/it]


    epoch:1 / loss:25.012045140303083 / Acc:93.31%


    epoch 2: 100%|██████████████████████████████████████████████████| 600/600 [1:45:56<00:00, 10.59s/it]


    epoch:2 / loss:20.828275472384306 / Acc:94.27%


    epoch 3: 100%|██████████████████████████████████████████████████| 600/600 [1:47:27<00:00, 10.75s/it]


    epoch:3 / loss:18.80353689706 / Acc:94.56%


    epoch 4: 100%|██████████████████████████████████████████████████| 600/600 [1:46:28<00:00, 10.65s/it]


    epoch:4 / loss:17.656414802759347 / Acc:94.89999999999999%


    epoch 5: 100%|██████████████████████████████████████████████████| 600/600 [1:48:33<00:00, 10.86s/it]


    epoch:5 / loss:16.967753436408792 / Acc:95.23%


    epoch 6: 100%|██████████████████████████████████████████████████| 600/600 [1:45:06<00:00, 10.51s/it]


    epoch:6 / loss:16.55541912486767 / Acc:95.46%


    epoch 7: 100%|██████████████████████████████████████████████████| 600/600 [1:44:46<00:00, 10.48s/it]


    epoch:7 / loss:16.35387544988638 / Acc:95.75%


    epoch 8: 100%|██████████████████████████████████████████████████| 600/600 [1:37:04<00:00,  9.71s/it]


    epoch:8 / loss:16.361360650187343 / Acc:95.67999999999999%


    epoch 9: 100%|██████████████████████████████████████████████████| 600/600 [1:39:56<00:00,  9.99s/it]


    epoch:9 / loss:16.593731036675795 / Acc:95.81%


    epoch 10: 100%|█████████████████████████████████████████████████| 600/600 [1:40:14<00:00, 10.02s/it]


    epoch:10 / loss:17.052510929713826 / Acc:95.66%


    epoch 11: 100%|█████████████████████████████████████████████████| 600/600 [1:45:57<00:00, 10.60s/it]


    epoch:11 / loss:17.73684521510844 / Acc:95.55%


    epoch 12: 100%|█████████████████████████████████████████████████| 600/600 [1:46:31<00:00, 10.65s/it]


    epoch:12 / loss:18.661012213820477 / Acc:95.37%


    epoch 13: 100%|█████████████████████████████████████████████████| 600/600 [1:45:19<00:00, 10.53s/it]


    epoch:13 / loss:19.796590694821546 / Acc:95.15%


    epoch 14: 100%|█████████████████████████████████████████████████| 600/600 [1:39:34<00:00,  9.96s/it]


    epoch:14 / loss:21.14038522085117 / Acc:94.78%


    epoch 15: 100%|█████████████████████████████████████████████████| 600/600 [1:54:47<00:00, 11.48s/it]


    epoch:15 / loss:22.714402585108903 / Acc:94.35%


    epoch 16: 100%|█████████████████████████████████████████████████| 600/600 [2:07:26<00:00, 12.74s/it]


    epoch:16 / loss:24.516768552776213 / Acc:93.92%


    epoch 17: 100%|█████████████████████████████████████████████████| 600/600 [2:08:54<00:00, 12.89s/it]


    epoch:17 / loss:26.54560838901244 / Acc:93.30000000000001%


    epoch 18: 100%|█████████████████████████████████████████████████| 600/600 [2:10:07<00:00, 13.01s/it]


    epoch:18 / loss:28.79938493905757 / Acc:92.91%


    epoch 19: 100%|█████████████████████████████████████████████████| 600/600 [2:08:54<00:00, 12.89s/it]


    epoch:19 / loss:31.295395751504195 / Acc:92.4%



```python

```


```python

```
