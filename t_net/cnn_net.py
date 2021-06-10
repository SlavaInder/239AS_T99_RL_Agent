import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input 20 x 10 x 1, output 18 x 8 x 3
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        # input 18 x 8 x 3, output 14 x 4 x 6
        self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
        # input 14 x 4 x 6, output 11 x 1 x 9
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
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.model(x)

        return q


class ExtendedCNN(nn.Module):
    def __init__(self):
        super(ExtendedCNN, self).__init__()
        # input 20 x 10 x 1, output 18 x 8 x 3
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        # input 18 x 8 x 3, output 14 x 4 x 6
        self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
        # input 14 x 4 x 6, output 11 x 1 x 9
        self.conv3 = nn.Conv2d(6, 9, kernel_size=4)
        # init model for board
        self.board_model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            nn.Flatten()
        )

        # input 100, output 100
        self.linear1 = nn.Sequential(nn.Linear(100, 100))
        self.linear2 = nn.Sequential(nn.Linear(100, 100))
        # final layer
        self.linear3 = nn.Sequential(nn.Linear(100, 1))
        # init model for features
        self.feature_model = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3
        )

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # process board to get synthetic features
        feats = self.board_model(x[:, :, :, :, 0])
        # print(feats.shape)
        # print(feats)
        # append number of cleared lines to the synthetic features
        lines = torch.reshape(x[:, 0, 0, 0, 1], (-1, 1))
        # print(x[:, 0, 0, 0, 1].shape)
        # print(x[:, 0, 0, 0, 1])
        feats_and_lines = torch.cat((feats, lines), dim=1)
        # calculate q using feature model
        q = self.feature_model(feats_and_lines)

        return q