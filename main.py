# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# import necessary libraries: Import the necessary libraries, including NumPy, Matplotlib, and PyTorch.
import numpy as np
import matplotlib.pyplot as plt
import torch

# Prepare the data: Load your data into NumPy arrays and convert them to PyTorch tensors.
# In this example, we'll generate some dummy data using NumPy.
X = np.linspace(0, 10, 100)
Y = X + np.random.randn(X.shape[0])
X_tensor = torch.from_numpy(X).float().reshape(-1, 1)
Y_tensor = torch.from_numpy(Y).float().reshape(-1, 1)

# Define the model: Define a simple linear regression model using PyTorch's nn module.
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Train the model: Train the model on the data using PyTorch's optim module.

model = LinearRegression(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(X_tensor)
    loss = criterion(y_pred, Y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Evaluate the model: Evaluate the model's performance on the data by comparing the predicted values to the actual values.

with torch.no_grad():
    y_pred = model(X_tensor)
    loss = criterion(y_pred, Y_tensor)
    print(f'Loss: {loss:.4f}')


#  Plot the results using Matplotlib.

plt.scatter(X, Y)
plt.plot(X, y_pred.numpy(), 'r')
plt.show()
