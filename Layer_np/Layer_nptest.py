import numpy as np

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
# a = np.ones((3,3,3))
# b = np.ones((3,3,3))
#
# c = np.array([
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ],
#     [
#         [3, 2, 1],
#         [6, 5, 4],
#         [9, 8, 7]
#     ],
#     [
#         [1, 4, 7],
#         [2, 5, 8],
#         [3, 6, 9]
#     ]
# ])
#
# d = np.array([
#     [
#         [1, 0, 1],
#         [0, 2, 0],
#         [1, 1, 1]
#     ],
#     [
#         [1, 1, 1],
#         [0, 1, 0],
#         [2, 2, 2]
#     ],
#     [
#         [1, 2, 1],
#         [2, 0, 2],
#         [2, 1, 2]
#     ]
# ])
# e = c * d
# print(np.sum(e))
#
# c_temp = []
# for c in range(1,4):
#     h = []
#     for i in range(1,6):
#         w = []
#         for j in range(1,6):
#             w.append(((i-1)*5 + j) * c)
#         h.append(w)
#     c_temp.append(h)
#
# _img = []
# _img.append(c_temp)
# _img = np.array(_img)
# image = np.ones((1, 3, 32, 32))
# # input 3-dim / output 10-dim / kernel size - 3 / stride - 3 / padding - 1
# conv = ConvLayer_np(3, 10, 3, 3, 1)
#
# out = conv.forward(_img)
# print()
# a = np.zeros((1,10))
# b = np.zeros((1,192))
# print(a)
# n = b.shape[0]
# c = np.dot(a.T, b)
# print(c.shape, n )

# back_propagation = np.ones((256,3,3,4))
# conv = ConvLayer_np(2,4,3,4,0)
# img = np.ones((256,12,12,2))
# # b = np.ones((1,3,3,2,1))
# # c = a * b
# # print(c.squeeze(axis=-1).sum(axis=0).shape)
# out = conv.forward(img)
# conv.backward(_back_gradient=back_propagation)

def conv_test():
    img = np.ones((256, 12, 12, 2))
    conv = ConvLayer_np(input_dim=2, output_dim=4, kernel_size=3, stride=4, padding=0)
    out = conv.forward(img)
    back = conv.backward(_back_gradient=out)
    print()

# conv_test()

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
def ReLU_Test():
    # N H W C
    x = np.random.randn(2, 3, 3, 2)
    ReLU_Layer = ReLULayer_np()
    out = ReLU_Layer.forward(x)
    back_prop = ReLU_Layer.backward(x)

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

def Linear_Layer_test():
    x = np.random.randn(2, 10) # 2 batch
    dense = LinearLayer_np(10, 5)
    out = dense.forward(x)
    dense.backward(out)
    print()

if __name__ == "__main__":
    Linear_Layer_test()

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
def maxpool_test():
    # N H W C
    input = np.random.randn(2, 4, 4, 2)
    MP_layer = MaxPoolLayer_np(2, 2, 0)
    output = MP_layer.forward(input)
    back = MP_layer.backward(output)
    print()

# maxpool_test()
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
def FlattenLayer_test():
    # N H W C
    x = np.random.randn(2, 4, 4, 2)
    FlattenLayer = FlattenLayer_np()
    output = FlattenLayer.forward(x)
    grad = FlattenLayer.backward(output)
    print()

# FlattenLayer_test()