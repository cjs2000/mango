import torch
from torch import nn
from torch.nn import Sequential


class LeNet_5(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # input_size = (1, 28, 28)
        self.conv1 = Sequential(
            # 由于input_size为28×28 所以conv1的padding就采用same 即填充两层
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # input_size = (14, 14, 6)
        self.conv2 = Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # input_size = (5, 5, 16)
        self.flatten1 = Sequential(
            nn.Flatten(),
        )

        # input_size = 400
        self.fc1 = Sequential(
            nn.Linear(in_features=5 * 5 * 16, out_features=120),
            nn.ReLU()
        )

        # input_size = 120
        self.fc2 = Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU()
        )

        # input_size = 84
        self.fc3 = Sequential(
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def LeNet():
    return LeNet_5()

if __name__ == '__main__':
    model = LeNet_5()
    print(model)
    input = torch.ones((64, 1, 28, 28))
    output = model(input)
    print(output.shape)
