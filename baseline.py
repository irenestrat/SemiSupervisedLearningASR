import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.lstm1 import LSTM


class Baseline(nn.Module):

    # Todo
    # Loss
    # Dropout or add noise - read how

    # Mean Teacher algorithm:
    # 1. Take a supervised architecture and make a copy of it. Let's call the original model the student and the new one the teacher.
    # 2. At each training step, use the same minibatch as inputs to both the student and the teacher but add random augmentation or noise to the inputs separately.
    # 3. Add an additional consistency cost between the student and teacher outputs (after softmax).
    # 4. Let the optimizer update the student weights normally.
    # 5. Let the teacher weights be an exponential moving average (EMA) of the student weights. That is, after each training step, update the teacher weights a little bit toward the student weights.

    def __init__(self, loss, mfccs, output_phonemes, size_hidden_layers):
        super(Baseline, self).__init__()

        self.name = 'Baseline'

        self.loss = loss

        self.model = LSTM(mfccs, output_phonemes, size_hidden_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def to(self, device):
        self.model = self.model.to(device)

    def get_optimizer(self):
        return self.optimizer

    def forward(self, x):
        return torch.squeeze(self.model(x), dim=1)

    def loss_fn(self, device, sample, targets):
        sample, target = sample.to(device), target.to(device)
        return self.loss(self.model(sample), targets)

    def train_step(self, loss_val):
        
        loss_val.backward()
        self.optimizer.step()
