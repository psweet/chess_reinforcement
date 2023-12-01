import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helpers import letter_2_num, dist_over_moves

import numpy as np

# https://www.youtube.com/watch?v=aOwvRvTPQrs

class ChessNetwork(nn.Module):
    def __init__(self, hidden_layers = 4, hidden_size = 200, lr = 0.003):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([Network(hidden_size) for _ in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)


    def forward(self, state):
        x = self.input_layer(state)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)
        
        x = self.output_layer(x)
        return x


class Network(nn.Module):
    def __init__(self, hidden_size):
        super(Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

        self.loss_from = nn.CrossEntropyLoss()
        self.loss_to = nn.CrossEntropyLoss()
        
        self.to(self.device)

    def forward(self, state):
        state_input = torch.clone(state)
        x = self.conv1(state)
        # x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = x + state_input
        x = self.activation2(x)
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

        self.Q_eval = ChessNetwork(
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
            state = torch.tensor(observation).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            # action = torch.argmax(actions).item()
            # print("chosen", action, action_space)

            vals = []
            froms = [str(legal_move)[:2] for legal_move in action_space]
            froms = list(set(froms))

            for from_ in froms:
                val = actions[0,:,:][8 - int(from_[1]), letter_2_num[from_[0]]]
                vals.append(val.detach().numpy())

            probs = dist_over_moves(vals)
            chosen_from = str(np.random.choice(froms, size=1, p=probs)[0])[:2]

            vals = []
            froms = [str(legal_move)[:2] for legal_move in action_space]
            froms = list(set(froms))

            for legal_move in action_space:
                from_ = str(legal_move)[:2]
                if from_ == chosen_from:
                    to = str(legal_move)[2:]
                    val = actions[1,:,:][8 - int(to[1]), letter_2_num[to[0]]]
                    vals.append(val.detach().numpy())
                else:
                    vals.append(0)

            action = action_space[np.argmax(vals)]
        else:
            action = np.random.choice(action_space)

        return action
    
    def learn(self):
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
        q_eval = self.Q_eval.forward(state_batch)
        print(q_eval.shape, batch_index.shape, action_batch.shape)
        print(action_batch[0], action_batch[0].shape)
        print(q_eval[batch_index, action_batch])
        # q_eval = q_eval[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        # print(terminal_batch.shape)
        # q_next[terminal_batch] = 0
        print(torch.max(q_next, dim =1))

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=2)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backwards()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min



