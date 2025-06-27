import torch
import matplotlib.pyplot as plt

x = torch.linspace(-20, 20, 2000)
x_derivative = torch.linspace(-20, 20, 2000, requires_grad=True)
torch.relu(x_derivative).sum().backward()
y = torch.relu(x)

_, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title("sigmoid function")

axes[1].plot(x_derivative.detach(), x_derivative.grad)
axes[1].grid()
axes[1].set_title("sigmoid derivative function")

plt.show()