import torch
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


class FCNetMultiplayer(nn.Module):
    def __init__(self, num_players):
        super(FCNetMultiplayer, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))

        self.attack_layer1 = nn.Sequential(nn.Linear(4*(num_players-1), 64), nn.ReLU(inplace=True))
        self.attack_layer2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))

        self.final_layer1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
        self.final_layer2 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        
        x2 = self.attack_layer1(x2)
        x2 = self.attack_layer2(x2)

        combined = torch.cat((x1, x2), dim=1)
        x = self.final_layer1(combined)
        x = self.final_layer2(x)
    
        return x

class FCNetTransfer(nn.Module):
    def __init__(self, num_players,checkpoint):
        super(FCNetTransfer, self).__init__()
        # checkpoint should be a checkpoint with at least:
            # "primary_net_state"
        #You can get a checkpoint by doing:  torch.load('model_path.pth')
        # See how states are saved in Agent.py functions "save_state()" if confused  

        self.layer1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))

        self.attack_layer1 = nn.Sequential(nn.Linear(4*(num_players-1), 64), nn.ReLU(inplace=True))
        self.attack_layer2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))

        self.final_layer1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
        self.final_layer2 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()
        
        #Overwrite weights where applicable
        model_dict = self.state_dict()
        pretrained_dict = checkpoint['primary_net_state']

        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
   
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        
        x2 = self.attack_layer1(x2)
        x2 = self.attack_layer2(x2)

        combined = torch.cat((x1, x2), dim=1)
        x = self.final_layer1(combined)
        x = self.final_layer2(x)
    
        return x