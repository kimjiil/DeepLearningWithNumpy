import numpy as np
from torchvision.datasets import MNIST
from Layer_np.Layer import ConvLayer_np, LinearLayer_np, ReLULayer_np, FlattenLayer_np, MaxPoolLayer_np, LogSoftMax_np
from Layer_np.optimizer import Adam


download_path = "./MNIST_Datset"
train_dataset = MNIST(download_path, train=True, download=True)
valid_dataset = MNIST(download_path, train=False, download=True)


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

optimizer = Adam(lr=0.0001)
LossFunc = None

trainX = train_dataset.data.numpy()
trainY = train_dataset.targets.numpy()

testX = valid_dataset.data.numpy()
testY = valid_dataset.targets.numpy()
#
total_epoch = 1000
batch_size = 12

trainX = trainX[:60000] / 255.0
trainY = trainY[:60000]

testX = testX[:10000]
testY = testY[:10000]

for epoch in range(total_epoch):
    step = int(len(trainX) / batch_size)
    loss_list = []
    for step_i in range(step):

        X = trainX[step_i*batch_size:(step_i+1)*batch_size]
        Y = trainY[step_i*batch_size:(step_i+1)*batch_size]
        one_hot_Y = one_hot(Y)
        X = X[:, :, :, np.newaxis]
        output = _forward(X, layers)
        back_propagation = output - one_hot_Y
        # loss = MSE_Loss(output, one_hot_Y)
        loss = Cross_Entropy_Loss(output, one_hot_Y)
        loss_list.append(loss)
        print(loss)
        _backward(back_propagation, layers)

        optimizer.update(layers=layers)

    TEST_X = testX[:, :, :, np.newaxis] / 255.0
    # TEST_Y = one_hot(testY)

    output = _forward(TEST_X, layers)
    # softmax_output = softmax(output)
    acc = np.mean(testY == np.argmax(output, axis=1))
    Text = f'epoch:{epoch} / loss:{np.mean(loss_list)} / Acc:{acc*100.0}%'
    print(f'epoch:{epoch} / loss:{np.mean(loss_list)} / Acc:{acc*100.0}%')
    with open('./Test.txt', 'a+') as file:
        file.write(Text + "\n")
if __name__ == "__main__":
    pass
