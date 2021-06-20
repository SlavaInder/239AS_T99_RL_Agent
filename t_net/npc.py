import torch
import torch.nn as nn
from copy import deepcopy


class StandardNPC(object):
    """
    A standard player to compete with the Agent
    public API consists of 2 functions:
        - act()
        - update()
    """
    def __init__(self, net, device, features):
        # copy the example net to init an enemy
        self.net = deepcopy(net)
        # record the name of the device
        self.device = device
        # send net to device
        self.net.to(device)
        # choose a function to extract features
        self.get_features = features

    def action(self, options, player_id):
        # standard npc always chooses a greedy actions from the options it has
        # get features for all next states
        feats = []
        for i in range(len(options)):
            feats.append(torch.from_numpy(self.get_features(options[i],player_id)))
        # then stack all possible net states into one tensor and send it to the GPU
        next_states = torch.stack(feats).type(torch.FloatTensor).to(self.device)
        # calculate predictions on the whole stack using primary net (see algorithm)
        predictions = self.net(next_states)[:, 0]
        # choose greedy action
        index = torch.argmax(predictions).item()
        # create an action from greedy option
        action = options[index]

        return action

    def update(self, weights_dict):
        # funciton that updates weights of the net
        self.net.load_state_dict(weights_dict)
