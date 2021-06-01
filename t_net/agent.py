from collections import deque
from copy import deepcopy
from .fc_net import FCNet
import numpy as np
import torch
import random


class AgentSC(object):
    '''
    A classic deep Q Learning Agent for simplified tetris 99 environment. Implements DQN with fixed target and
    experience replay learning strategy
    Args:
        env (T99SC):                    environment to train on
        discount (float):               how important is the future rewards compared to the immediate ones [0,1]
        net (nn.module):                NN architecture and weights that will be copied to both target and policy nets
        learning rate (float):          speed of optimization
        criterion (nn.module):          loss function
        device (object):                device that will handle NN computations
        features (object):              function used to extract NN-acceptable features form env.state
        epsilon (float):                exploration (probability of random values given) value at the start
        mem_size (int):                 size of the cyclic buffer that records actions
    '''

    def __init__(self, env, discount, net, learning_rate, criterion, device, features, exploration_rate=0.1, mem_size=10000):

        # memory is a cyclic buffer
        self.memory = deque(maxlen=mem_size)
        # tetris environment
        self.env = env
        # gamma
        self.discount = discount
        # specifies a threshold, after which the amount of collected data is sufficient for starting training
        self.replay_start_size = 2000
        # set up nets
        self.policy_net = deepcopy(net)
        self.target_net = deepcopy(net)
        self.optimizer = torch.optim.Adam(self.target_net.parameters(), lr=learning_rate)
        self.criterion = criterion
        # choose a function to extract features
        self.get_features = features
        # set up strategy hyper parameters
        self.exploration_rate = exploration_rate

        # record the device for future use
        self.device = device
        # send the nets to the device
        self.policy_net.to(device)
        self.target_net.to(device)
        # initialize memory
        self.cumulative_rewards = [0]
        self.steps_per_episode = [0]
        # initialize episode
        self.episode = 0

    def add_to_memory(self, current_state, next_state, reward, done):
        # Adds a play to the replay memory buffer
        self.memory.append((current_state, next_state, reward, done))

    def act(self):
        '''
        Makes a single-step update in accordance with epsilon-greedy strategy
        '''
        # observe the options we have for reward and next states of the player controlled by the agent
        options, rewards = self.env._observe(0)
        # check if we are exploring on this step
        if np.random.random_sample() <= self.exploration_rate:
            # if so, choose an action on random
            index = np.random.randint(0, high=len(rewards))
        # if we are exploiting on this step
        else:
            # get features for all next states
            feats = []
            for i in range(len(rewards)):
                feats.append(torch.from_numpy(self.get_features(options[i])))
            # then stack all possible net states into one tensor and send it to the GPU
            next_states = torch.stack(feats).type(torch.FloatTensor).to(self.device)
            # calculate predictions on the whole stack using target net (see algorithm)
            predictions = self.target_net(next_states)[:, 0]
            # choose greedy action
            index = torch.argmax(predictions).item()
        # now make a step according with a selected action, and record the reward
        action = {
            "reward": rewards[index],
            "state": options[index]
        }
        _, reward, done, _ = self.env.step(action)

        return reward, done

    def train(self, batch_size=128, update_freq=2000, steps=10000):
        '''
        Trains the agent by following DQN-learning algorithm with experience replay and fixed target
        '''
        # repeats the algorithm steps times
        for i in range(steps):
            # record the step
            if i % 1000 == 0: print("calculating step", i)
            # get features for the current state
            current_state_features = self.get_features(self.env.state)
            # make an action, record the reward and check whether environment is done
            reward, done = self.act()
            # get next state features
            next_state_features = self.get_features(self.env.state)
            # record the current state, reward, next state, done in the cyclic buffer for future replay
            self.add_to_memory(current_state_features, reward, next_state_features, done)
            # record cumulative reward and steps done in the current episode
            self.cumulative_rewards[self.episode] += reward
            self.steps_per_episode[self.episode] += 1
            # if the environment is done, reboot it and reset counters
            if done:
                self.env.reset()
                self.episode += 1
                self.cumulative_rewards.append(0)
                self.steps_per_episode.append(0)

            # check if there is enough data to start training
            if len(self.memory) > self.replay_start_size:
                # sample a batch of transitions
                batch = random.sample(self.memory, batch_size)
                # init arrays to hold states and rewards
                batch_states = []
                batch_rewards = []
                batch_next_states = []
                # store features in arrays
                for j in range(len(batch)):
                    batch_states.append(torch.from_numpy(batch[j][0]))
                    batch_rewards.append(torch.tensor(batch[j][1]))
                    batch_next_states.append(torch.from_numpy(batch[j][2]))
                # stack tensors and send them to GPU
                torch_states = torch.stack(batch_states).type(torch.FloatTensor).to(self.device)
                torch_rewards = torch.stack(batch_rewards).type(torch.FloatTensor).to(self.device)
                torch_next_states = torch.stack(batch_next_states).type(torch.FloatTensor).to(self.device)
                # get the expected score for the next state using policy net
                q_next = self.policy_net(torch_next_states)[:, 0]
                # calculate target
                y_i = q_next + torch.tensor(self.discount) * torch_rewards
                # get the expected score for the current state using target net
                q_current = self.target_net(torch_states)[:, 0]

                # Fit the model to the given values
                loss = self.criterion(y_i, q_current)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # if it's time to update policy net
            if i % update_freq == 0:
                # pop up a message
                print("policy net updated")
                # update weights
                self.policy_net.load_state_dict(self.target_net.state_dict())

    def save_state(self,path):     
        torch.save({
            'target_net_state': self.target_net.state_dict(),
            'policy_net_state': self.policy_net.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'discount' : self.discount,
            'replay_start_size' : self.replay_start_size,
            'exploration_rate' : self.exploration_rate,
            'mem_size' : self.memory.maxlen,
            'cumulative_rewards' : self.cumulative_rewards,
            'steps_per_episode' : self.steps_per_episode
            }, path)

    def load_state(self,path):
        # Don't forget to do .eval() or .train() now!
        checkpoint = torch.load(path)
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.discount = checkpoint['discount']
        self.replay_start_size = checkpoint['replay_start_size']
        self.exploration_rate = checkpoint['exploration_rate']
        self.memory = deque(maxlen=checkpoint['mem_size'])
        self.cumulative_rewards = checkpoint['cumulative_rewards']
        self.steps_per_episode = checkpoint['steps_per_episode']


class AgentDoubleSC(object):
    '''
    A classic deep Q Learning Agent for simplified tetris 99 environment. Implements DQN with fixed target and
    experience replay learning strategy
    Args:
        env (T99SC):                    environment to train on
        discount (float):               how important is the future rewards compared to the immediate ones [0,1]
        net (nn.module):                NN architecture and weights that will be copied to both target and policy nets
        learning rate (float):          speed of optimization
        criterion (nn.module):          loss function
        device (object):                device that will handle NN computations
        features (object):              function used to extract NN-acceptable features form env.state
        epsilon (float):                exploration (probability of random values given) value at the start
        mem_size (int):                 size of the cyclic buffer that records actions
    '''

    def __init__(self, env, discount, net, learning_rate, criterion, device, features, exploration_rate=0.1, mem_size=10000):

        # memory is a cyclic buffer
        self.memory = deque(maxlen=mem_size)
        # tetris environment
        self.env = env
        # gamma
        self.discount = discount
        # specifies a threshold, after which the amount of collected data is sufficient for starting training
        self.replay_start_size = 2000
        # set up nets
        self.primary_net = deepcopy(net)
        self.target_net = deepcopy(net)
        self.optimizer = torch.optim.Adam(self.primary_net.parameters(), lr=learning_rate)
        self.criterion = criterion
        # choose a function to extract features
        self.get_features = features
        # set up strategy hyper parameters
        self.exploration_rate = exploration_rate

        # record the device for future use
        self.device = device
        # send the nets to the device
        self.primary_net.to(device)
        self.target_net.to(device)
        # initialize memory
        self.cumulative_rewards = [0]
        self.steps_per_episode = [0]
        # initialize episode
        self.episode = 0

    def add_to_memory(self, s_t, s_t_1, s_t_2, reward, done):
        # Adds a play to the replay memory buffer
        self.memory.append((s_t, s_t_1, s_t_2, reward, done))

    def act(self):
        '''
        Makes a single-step update in accordance with epsilon-greedy strategy
        '''
        # observe the options we have for reward and next states of the player controlled by the agent
        options, rewards = self.env._observe(0)
        # check if we are exploring on this step
        if np.random.random_sample() <= self.exploration_rate:
            # if so, choose an action on random
            index = np.random.randint(0, high=len(rewards))
        # if we are exploiting on this step
        else:
            # get features for all next states
            feats = []
            for i in range(len(rewards)):
                feats.append(torch.from_numpy(self.get_features(options[i])))
            # then stack all possible net states into one tensor and send it to the GPU
            next_states = torch.stack(feats).type(torch.FloatTensor).to(self.device)
            # calculate predictions on the whole stack using primary net (see algorithm)
            predictions = self.primary_net(next_states)[:, 0]
            # choose greedy action
            index = torch.argmax(predictions).item()
        # now make a step according with a selected action, and record the reward
        action = {
            "reward": rewards[index],
            "state": options[index]
        }
        _, reward, done, _ = self.env.step(action)

        return reward, done

    def optimal_action(self):
        # finds optimal action using target net
        # observe the options we have for reward and next states of the player controlled by the agent
        options, rewards = self.env._observe(0)
        # get features for all next states
        feats = []
        for i in range(len(rewards)):
            feats.append(torch.from_numpy(self.get_features(options[i])))
        # then stack all possible next states into one tensor and send it to the GPU
        next_states = torch.stack(feats).type(torch.FloatTensor).to(self.device)
        # calculate predictions on the whole stack using target net (see algorithm)
        predictions = self.target_net(next_states)[:, 0]
        # choose greedy action
        index = torch.argmax(predictions).item()

        return options[index]

    def train(self, batch_size=128, update_freq=2000, steps=10000):
        '''
        Trains the agent by following Double DQN-learning algorithm
        '''
        # repeats the algorithm steps times
        for i in range(steps):
            # record the step
            if i % 1000 == 0: print("calculating step", i)
            # get features for the s(t)
            s_t_features = self.get_features(self.env.state)
            # make an action, record the reward and check whether environment is done
            reward, done = self.act()
            # get features for the s(t+1)
            s_t_1_features = self.get_features(self.env.state)
            # initialize empty s(t+2)
            s_t_2_features = None
            # if the environment has not finished
            if not done:
                # find and record optimal action a*=s*(t+2)=argmax{Q'[s(t+1), a=s(t+2)]} according to the target net
                s_t_2 = self.optimal_action()
                s_t_2_features = self.get_features(s_t_2)
            # record the current state, reward, next state, done in the cyclic buffer for future replay
            self.add_to_memory(s_t_features, s_t_1_features, s_t_2_features, reward, done)
            # record cumulative reward and steps done in the current episode
            self.cumulative_rewards[self.episode] += reward
            self.steps_per_episode[self.episode] += 1
            # if the environment is done, reboot it and reset counters
            if done:
                self.env.reset()
                self.episode += 1
                self.cumulative_rewards.append(0)
                self.steps_per_episode.append(0)

            # check if there is enough data to start training
            if len(self.memory) > self.replay_start_size:
                # sample a batch of transitions
                batch = random.sample(self.memory, batch_size)
                # init arrays to hold states and rewards
                batch_rewards = []
                # we need s(t+1) for getting the value of Q[s(t), a=s(t+1)] = Q[ . , s(t+1)]
                batch_s_t_1 = []
                # we need s(t+2) for getting the value of Q[s(t+1), s*(t+2)=argmax{Q'[s(t+1), a=s(t+2)]}] = \
                # Q[ . , s(t+2)]
                batch_s_t_2 = []
                # we need to keep track of non-zero indexes of batch_s_t_2
                batch_s_t_2_idx = []
                # store features in arrays
                for j in range(len(batch)):
                    batch_rewards.append(torch.tensor(batch[j][3]))
                    batch_s_t_1.append(torch.from_numpy(batch[j][1]))
                    # if the environment was not done and the next state exists
                    if not batch[j][4]:
                        batch_s_t_2.append(batch[j][2])
                        batch_s_t_2_idx.append[j]
                # stack tensors and send them to GPU
                torch_s_t_1 = torch.stack(batch_s_t_1).type(torch.FloatTensor).to(self.device)
                torch_rewards = torch.stack(batch_rewards).type(torch.FloatTensor).to(self.device)
                torch_s_t_2 = torch.stack(batch_s_t_2).type(torch.FloatTensor).to(self.device)
                # get the expected score for the s(t+2) using primary net
                q_s_t_2_dense = self.primary_net(torch_s_t_2)[:, 0]
                # their order is not the same as in the batch, so we need to rearrange it
                q_s_t_2_sparse = torch.zeros(batch_size)
                for j in range(len(batch_s_t_2_idx)):
                    q_s_t_2_sparse[batch_s_t_2_idx[j]] = q_s_t_2_dense[j]
                # send this new tensor to the device
                q_s_t_2_sparse.type(torch.FloatTensor).to(self.device)
                # calculate target
                y_i = q_s_t_2_sparse + torch.tensor(self.discount) * torch_rewards
                # get the expected score for the s(t+1) using primary net
                q_current = self.target_net(torch_s_t_1)[:, 0]

                # Fit the model to the given values
                loss = self.criterion(y_i, q_current)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # if it's time to update policy net
            if i % update_freq == 0:
                # pop up a message
                print("policy net updated")
                # update weights
                self.policy_net.load_state_dict(self.target_net.state_dict())

    def save_state(self,path):
        torch.save({
            'target_net_state': self.target_net.state_dict(),
            'policy_net_state': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'discount' : self.discount,
            'replay_start_size' : self.replay_start_size,
            'exploration_rate' : self.exploration_rate,
            'mem_size' : self.memory.maxlen,
            'cumulative_rewards' : self.cumulative_rewards,
            'steps_per_episode' : self.steps_per_episode
            }, path)

    def load_state(self,path):
        # Don't forget to do .eval() or .train() now!
        checkpoint = torch.load(path)
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.discount = checkpoint['discount']
        self.replay_start_size = checkpoint['replay_start_size']
        self.exploration_rate = checkpoint['exploration_rate']
        self.memory = deque(maxlen=checkpoint['mem_size'])
        self.cumulative_rewards = checkpoint['cumulative_rewards']
        self.steps_per_episode = checkpoint['steps_per_episode']



