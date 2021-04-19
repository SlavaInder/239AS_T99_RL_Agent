import torch
from torch import nn

class SimpleDQN(nn.Module):
# this is a class set up for test only
# needs

    def __init__(self, input_size, width, height, output_size):
        # this function has to be modified later to account for different
        # possible architectures we might want to test.
        super(SimpleDQN, self).__init__()


        assert width >= 2


    def forward(self):
        pass