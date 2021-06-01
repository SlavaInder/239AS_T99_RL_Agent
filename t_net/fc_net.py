import torch.nn as nn


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class FCBoardNet(nn.Module):
    def __init__(self):
        super(FCBoardNet, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Sequential(nn.Linear(201, 100), nn.ReLU(inplace=True)),
            nn.ReLU(),
            nn.Sequential(nn.Linear(100, 100), nn.ReLU(inplace=True)),
            nn.ReLU(),
            nn.Sequential(nn.Linear(100, 100), nn.ReLU(inplace=True)),
            nn.ReLU(),
            nn.Sequential(nn.Linear(100, 100), nn.ReLU(inplace=True)),
            nn.ReLU(),
            nn.Sequential(nn.Linear(100, 1), nn.ReLU(inplace=True)),
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
