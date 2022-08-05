import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import numpy as np

download_path = "./MNIST_Datset"
train_dataset = MNIST(download_path, train=True, download=True)
valid_dataset = MNIST(download_path, train=False, download=True)

#git test
class Simple_CNN(nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.fc1 = nn.Linear(4 * 4 * 128, 625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":

    model = Simple_CNN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_len = len(train_dataset)
    batch_size = 32
    epoch = 10
    for epoch_i in range(epoch):
        step_size = int(train_len / batch_size)

        for step_i in range(step_size):
            data = train_dataset.data[step_i * batch_size: batch_size * (step_i+1)]
            label = train_dataset.targets[step_i * batch_size: batch_size * (step_i+1)]
            data = torch.unsqueeze(data, 0)
            data = data.permute(1, 0, 2, 3)
            data = data.type(torch.FloatTensor)

            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            print(loss)

        with torch.no_grad():
            valid_data = valid_dataset.data.view(len(valid_dataset), 1, 28, 28).float()
            valid_label = valid_dataset.targets
            torch.save()
            prediction = model(valid_data)
            torch.argmax(prediction, dim=1)
            acc = (torch.argmax(prediction, dim=1) == valid_label).float().mean()
            print('acc : ', acc.item())