import torch
import numpy as np
from ignite.metrics import Metric
from matplotlib import pyplot as plt


class SimLoss(torch.nn.Module):
    def __init__(self, number_of_classes, reduction_factor, device="cpu", epsilon=1e-8):
        super().__init__()

        assert number_of_classes > 0

        self.__number_of_classes = number_of_classes
        self.__device = device
        self.epsilon = epsilon
        self.r = reduction_factor

    def forward(self, x, y):
        w = self.__w[y, :]
        return torch.mean(-torch.log(torch.sum(w * x, dim=1) + self.epsilon))

    @property
    def r(self):
        return self.__r
    
    @r.setter
    def r(self, r):
        assert r >= 0.0
        assert r < 1.0

        self.__r = r
        self.__w = self.__generate_w(self.__number_of_classes, self.__r, self.__device)

    def __generate_w(self, number_of_classes, reduction_factor, device):
        w = torch.zeros((number_of_classes, number_of_classes)).to(device)
        for j in range(number_of_classes):
            for i in range(number_of_classes):
                w[j, i] = reduction_factor ** np.abs(i - j)

        return w

    def __repr__(self):
        return "SimilarityBasedCrossEntropy"


class PlotMetric(Metric):
    def __init__(self, number_of_classes, plot_output="img"):
        self.number_of_classes = number_of_classes
        self.mean_prob = np.zeros((self.number_of_classes,))
        self.length = 0
        self.plot_output = plot_output
        self.index = 0

        super().__init__()

    def compute(self):
        mean_prob = self.mean_prob / self.length
        x = np.arange(len(mean_prob)) + 1

        plt.figure(figsize=(7, 3))
        plt.plot(x, mean_prob)
        plt.savefig(f"{self.plot_output}/{self.index:03d}.png")

        # the metric has to return something
        return 0.0

    def update(self, output):
        out, target = output
        self.length += out.shape[0]

        mean_prob = np.mean(out.data.numpy(), axis=0)
        self.mean_prob += mean_prob
            
    def reset(self):
        self.mean_prob = np.zeros((self.number_of_classes,))
        self.length = 0
        self.index += 1
