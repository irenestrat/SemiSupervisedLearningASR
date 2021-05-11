import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

inputs = [torch.randn(1, 13) for _ in range(500)]        # 500 x 13
targets = [torch.randn(1, 48) for _ in range(500)]       # 500 x 48
outputs = [torch.randn(1, 48) for _ in range(500)]       # 500 x 48

# print(inputs)

# Setting Parameters of The Network
hidden_dimension = 500                                        # 500 hidden layers
hidden_size = 48
number_of_batch = 1
number_of_epochs = 300
isCTCLoss = False
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('DEVICE')
print(device)

class LSTMClass(nn.Module):
    def __init__(self, input, hidden_dim, hidden_size):
        super(LSTMClass, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_dim, hidden_size)
    def forward(self, input):
        input = input.to(device)
        # lstm_out, _ = self.lstm(input.view(len(input), number_of_batch, -1))
        lstm_out, _ = self.lstm(input)
        return lstm_out

model = LSTMClass(inputs, hidden_dimension, hidden_size).to(device)
a = model
if isCTCLoss:
    loss_function = nn.CTCLoss()
else:
    loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

loss_array = []
for epoch in range(number_of_epochs):
    # for phoneme, target in zip(outputs, targets):
    model.zero_grad()
    # targets = targets.to(device)
    # targets = torch.Tensor(targets[:]) # check here

    targets = targets.to(device)
    # outputs = torch.Tensor(outputs[:]) # check here

    trained_data = model(outputs)

    if isCTCLoss:
        # loss = loss_function(probs_here, targets, input_lengths, target_lengths)
        print('uncomment here')
    else:
        loss = loss_function(trained_data, targets)

    loss.backward()
    optimizer.step()

    # print('loss')
    # print(loss)
    loss_array.append(loss)


print('FINAL LOSS')
print(loss_array)
plt.plot(loss_array)
plt.show()
