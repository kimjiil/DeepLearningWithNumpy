import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import numpy as np

download_path = "./MNIST_Datset"
train_dataset = MNIST(download_path, train=True, download=True)
valid_dataset = MNIST(download_path, train=False, download=True)

#git test
#git test2
class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()


        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer2(x)


class my_model(nn.Module):
        def __init__(self):
            super(my_model, self).__init__()
            self.maxfool = nn.MaxPool2d(3, 2)
            self.flatten = nn.Flatten()
            self.hidden_layers = nn.Sequential(
                nn.Linear(in_features=169, out_features=312, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=312, out_features=128, bias=True),
                nn.ReLU()
            )
            self.classifier = nn.Linear(in_features=128, out_features=10, bias=True)
            self.sigmoid = nn.Sigmoid()
            print()

        def forward(self, x):
            x = self.maxfool(x)
            x = self.flatten(x)
            x = self.hidden_layers(x)
            x = self.classifier(x)
            x = self.sigmoid(x)
            return x

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
    import time

    # model = Simple_CNN()
    model = my_model()
    model.to("cuda:0")
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_len = len(train_dataset)
    batch_size = 144
    epoch = 100
    for epoch_i in range(epoch):
        step_size = int(train_len / batch_size)
        start_time = time.time()
        loss_sum = []
        for step_i in range(step_size):
            data = train_dataset.data[step_i * batch_size: batch_size * (step_i+1)]
            label = train_dataset.targets[step_i * batch_size: batch_size * (step_i+1)]
            data = torch.unsqueeze(data, 0)
            data = data.permute(1, 0, 2, 3)
            data = data.type(torch.FloatTensor).to("cuda:0")

            optimizer.zero_grad()
            outputs = model(data)

            target = torch.eye(10)[label].to("cuda:0")

            loss = criterion(outputs, target)
            loss.backward()
            loss_sum.append(loss.data)
            optimizer.step()

        end_time = time.time()
        print(epoch_i, sum(loss_sum) / len(loss_sum), end_time - start_time)

        # with torch.no_grad():
        #     valid_data = valid_dataset.data.view(len(valid_dataset), 1, 28, 28).float()
        #     valid_label = valid_dataset.targets
        #     torch.save()
        #     prediction = model(valid_data)
        #     torch.argmax(prediction, dim=1)
        #     acc = (torch.argmax(prediction, dim=1) == valid_label).float().mean()
        #     print('acc : ', acc.item())