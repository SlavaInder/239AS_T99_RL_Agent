import torch.nn as nn


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # input 20 x 10 x 1, output 18 x 8 x 3
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        # input 18 x 8 x 3, output 14 x 4 x 6
        self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
        # input 14 x 4 x 9, output 11 x 1 x 9
        self.conv3 = nn.Conv2d(6, 9, kernel_size=4)
        # input 99, output 1
        self.linear1 = nn.Sequential(nn.Linear(99, 1))
        # init mode
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            nn.Flatten(),
            self.linear1
        )

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.model(x)

        return q
