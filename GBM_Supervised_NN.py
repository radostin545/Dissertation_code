import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import anderson
import matplotlib.pylab as plt


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

class QuantileEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QuantileEstimator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.fc6(x)
        return x

data = pd.read_csv('training_data_GBM.csv', header=None)

t_values = data.iloc[0, 1:].values
log_prices = data.iloc[1:, 1:].values

uniform_samples = np.linspace(0, 1, log_prices.shape[0]+1)
uniform_samples = uniform_samples[1:]

sorted_log_prices = np.sort(log_prices, axis=0)
input_data = [[],[]]

for i in t_values:
    input_data[0].extend([i]*len(uniform_samples))
    for j in uniform_samples:
        input_data[1].extend([round(j, 3)])

input_data = torch.tensor(input_data, dtype=torch.float32).T
input_data  = input_data[:, [1, 0]]


sorted_log_prices = np.array(sorted_log_prices)
target_data = sorted_log_prices.flatten('F')

target_data = torch.tensor(target_data, dtype=torch.float32)
target_data = target_data.unsqueeze(1)

input_size = input_data.shape[1]

output_size = target_data.shape[1]

model = QuantileEstimator(input_size=input_size, hidden_size=256, output_size=output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):

    # Forward pass
    outputs = model(input_data.to(device))

    # Calculate loss
    loss = criterion(outputs, target_data.to(device))

    # Zero grad optimizer
    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Step the optimizera
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

model.eval()

