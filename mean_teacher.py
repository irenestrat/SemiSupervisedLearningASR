import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from optimizers.ema import ExponetialMovingAverage
from models.lstm1 import LSTM
from torch.autograd import Variable


class MeanTeacher(nn.Module):

    # Problems?
    # 1. Random aug. or noise
    # 2. No softmax is used
    # 3. Batch size = 1. In the paper they have for example 40 unlabeled samples and 20 labeled samples in the batch

    # Mean Teacher algorithm:
    # 1. Take a supervised architecture and make a copy of it. Let's call the original model the student and the new one the teacher.
    # 2. At each training step, use the same minibatch as inputs to both the student and the teacher but add random augmentation or noise to the inputs separately.
    # 3. Add an additional consistency cost between the student and teacher outputs (after softmax).
    # 4. Let the optimizer update the student weights normally.
    # 5. Let the teacher weights be an exponential moving average (EMA) of the student weights. That is, after each training step, update the teacher weights a little bit toward the student weights.

    def __init__(self, mfccs, output_phonemes, size_hidden_layers, max_steps=10000, ema_decay=0.999, consistency_weight=1.0):
        super(MeanTeacher, self).__init__()

        self.name = 'MeanTeacher'
        self.consistency_weight = consistency_weight
        self.max_steps = max_steps
        self.step = 0

        self.std = 0.15  # Need to check
        self.mean = 0.0

        self.loss_consistency = nn.MSELoss()
        self.loss_class = nn.CrossEntropyLoss()

        self.student = LSTM(mfccs, output_phonemes, size_hidden_layers)
        self.teacher = LSTM(mfccs, output_phonemes, size_hidden_layers)

        self.teacher.load_state_dict(self.student.state_dict())

        self.ema_optimizer = ExponetialMovingAverage(
            model=self.student, ema_model=self.teacher, alpha=ema_decay)

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=0.001)

    def to(self, device):
        self.student = self.student.to(device)
        self.teacher = self.teacher.to(device)

    def get_optimizer(self):
        return self.optimizer

    def forward_student(self, x):
        return torch.squeeze(self.student(x), dim=1)

    def forward_teacher(self, x):
        return torch.squeeze(self.teacher(x), dim=1)

    def forward(self, x):
        return self.forward_student(x)

    def loss_fn(self, device, sample, targets):
        loss = 0

        sample = sample.to(device)

        if not(targets is None):
            targets = targets.to(device)
            loss += self.loss_class(self.forward_student(sample +
                                    torch.randn(sample.size()).to(device) * self.std), targets)

        loss += self.consistency_weight * self.loss_consistency(self.forward_student(sample + torch.randn(sample.size()).to(device) * self.std),
                                                                self.forward_teacher(sample + torch.randn(sample.size()).to(device) * self.std))

        return loss

    def train_step(self, loss_val):
        loss_val.backward()
        self.optimizer.step()
        self.ema_optimizer.step()

    def linear_ramp_up(self):
        return min(float(self.step) / self.max_steps, 1.0)

    def sigmoid_rampup(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
