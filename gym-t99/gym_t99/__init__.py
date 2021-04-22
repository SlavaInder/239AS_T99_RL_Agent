from gym.envs.registration import register
from gym_t99.envs.state import *

register(
    id='basic-v0',
    entry_point='gym_t99.envs:Basic',
)

register(
    id='t99-v0',
    entry_point='gym_t99.envs:T99',
)

