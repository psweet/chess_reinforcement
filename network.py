import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class Network(nn.Module):
    def __init__(self, lr):
        super(Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(390, 1, 1)
        self.conv2 = nn.Conv2d(1, 9, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        self.to(self.device)

    def forward(self, state, available_actions):
        n_actions = len(available_actions)
        fc1 = nn.Linear(1, n_actions)

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = fc1(x)
        return x
    
class Agent():
    def __init__(
            self,
            gamma,
            epsilon,
            lr,
            input_dims,
            batch_size,
            max_me_size = 100_000,
            eps_end=0.01,
            eps_dec=5e-4
        ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_me_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = Network(
            lr = self.lr,
        )

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, 1), dtype=np.int32)
        self.terminal_memory = np.zeros((self.mem_size, 1), dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation, action_space):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state, action_space)
            action = torch.argmax(actions).item()
            print("chosen", action, action_space)
        else:
            action = np.random.choice(action_space)

        return action
    
    def learn(self, action_space, next_action_space):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        # terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]
        q_eval = self.Q_eval.forward(state_batch, action_space)
        print(q_eval.shape, batch_index.shape, action_batch.shape)
        print(action_batch[0], action_batch[0].shape)
        print(q_eval[batch_index, action_batch])
        # q_eval = q_eval[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch, next_action_space)
        # print(terminal_batch.shape)
        # q_next[terminal_batch] = 0
        print(torch.max(q_next, dim =1))

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=2)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backwards()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min



