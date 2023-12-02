import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helpers import letter_dict, move_probabilities, move_to_matrix, device
from collections import namedtuple, deque
import random

import numpy as np

# https://www.youtube.com/watch?v=aOwvRvTPQrs


class Network(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200, lr=0.003):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList(
            [SubNet(hidden_size).to(device) for _ in range(hidden_layers)]
        )
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = self.input_layer(state)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)
        return x


class SubNet(nn.Module):
    def __init__(self, hidden_size):
        super(SubNet, self).__init__()

        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, state):
        state_input = torch.clone(state)
        x = self.conv1(state)
        x = self.activation1(x)
        x = self.conv2(x)
        x = x + state_input
        x = self.activation2(x)
        return x


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        batch_size,
        mem_size=100_000,
        eps_end=0.01,
        eps_dec=5e-4,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.batch_size = batch_size

        self.network = Network(
            lr=self.lr,
        ).to(device)

        self.memory = ReplayMemory(mem_size)

    def store_transition(self, state, action, next_state, reward):
        self.memory.push(state.to(device), action.to(device), next_state, reward)

    def choose_action(self, observation, action_space):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float32).to(device)
            actions = self.network.forward(state)

            vals = []
            froms = [str(legal_move)[:2] for legal_move in action_space]
            froms = list(set(froms))

            for from_ in froms:
                val = actions[0, :, :][8 - int(from_[1]), letter_dict[from_[0]]]
                vals.append(val.cpu())

            probs = move_probabilities(torch.tensor(vals, device=device))

            try:
                chosen_from = str(np.random.choice(froms, size=1, p=probs)[0])[:2]
            except Exception:
                chosen_from = str(np.random.choice(froms, size=1)[0])[:2]

            vals = []
            froms = [str(legal_move)[:2] for legal_move in action_space]
            froms = list(set(froms))

            for legal_move in action_space:
                from_ = str(legal_move)[:2]
                if from_ == chosen_from:
                    to = str(legal_move)[2:]
                    val = actions[1, :, :][8 - int(to[1]), letter_dict[to[0]]]
                    vals.append(val.cpu())
                else:
                    vals.append(0)

            vals = torch.tensor(vals, device=device)
            action = action_space[np.argmax(vals.cpu())]
        else:
            action = np.random.choice(action_space)

        to = action.uci()[2:]
        fr = action.uci()[:2]

        from_mat = np.zeros((8, 8))
        from_mat[8 - int(fr[1]), letter_dict[fr[0]]] = 1

        to_mat = np.zeros((8, 8))
        to_mat[8 - int(to[1]), letter_dict[to[0]]] = 1

        return action, move_to_matrix(action)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_next_states = [s for s in batch.next_state if s is not None]

        count_states = len(non_final_next_states)

        non_final_next_states = torch.cat(non_final_next_states).reshape(
            count_states, 6, 8, 8
        )

        state_batch = torch.cat(batch.state).reshape(self.batch_size, 6, 8, 8)
        action_batch = torch.cat(batch.action).reshape(self.batch_size, 2, 8, 8)
        reward_batch = torch.cat(batch.reward)

        rewards = []
        for reward in reward_batch[:count_states]:
            sub_reward = []
            for _ in range(2):
                sub_sub_reward = []
                for _ in range(8):
                    sub_sub_reward.append(np.full(8, reward.cpu()))
                sub_reward.append(sub_sub_reward)
            rewards.append(sub_reward)

        reward_batch = torch.tensor(rewards).to(device)

        state_action_values = self.network(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, 2, 8, 8).to(device)
        with torch.no_grad():
            maxes = self.network(non_final_next_states).max(1)
            next_state_values = torch.cat([maxes[0], maxes[1]]).reshape(
                count_states, 2, 8, 8
            )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.network.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.network.optimizer.step()

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )
