import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=7)
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=11)
        self.dense1 = nn.Linear(in_features=80, out_features=120)
        self.dense2 = nn.Linear(in_features=120, out_features=number_of_classes)

    def forward(self, x, **kwargs):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))

        x = x.view(-1, 80)

        x = F.relu(self.dense1(x))
        x = self.dense2(x)

        return F.softmax(x, dim=1)
