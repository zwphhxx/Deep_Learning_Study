'''
Backpropagation is an algorithm that efficiently computes the gradients (partial derivatives) of the loss function with
respect to all parameters in a neural network. These gradients are then used by optimizers (such as gradient descent)
to update the parameters, minimizing the loss function.
'''

# Implementation of Automatic Differentiation
import torch


def autograd_scalar():
    x = torch.tensor(10)
    # Target Value
    y = torch.tensor(0.)
    # Initial Value for weight and intercept
    w = torch.tensor(1., requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3., requires_grad=True, dtype=torch.float32)
    # Output Value
    z = w * x + b
    # Loss Function
    loss = torch.nn.MSELoss()
    loss = loss(z, y)
    # Automatic Differentiation
    loss.backward()
    # Gradients computed by the backward function are stored in the tensor's .grad attribute.
    print("w's grad:", w.grad)
    print("b's grad:", b.grad)


def autograd_tensor():
    x = torch.ones(3, 4)
    # Target Value
    y = torch.zeros(3, 6)
    # Initial Value for weight and intercept
    w = torch.randn(4, 6, requires_grad=True)
    b = torch.randn(6, requires_grad=True) # Broadcasting Mechanism->4*6
    # Output Value
    z = x @ w + b
    # Loss Function
    loss = torch.nn.MSELoss()
    loss = loss(z, y)
    # Automatic Differentiation
    loss.backward()
    # Gradients computed by the backward function are stored in the tensor's .grad attribute.
    print("w's grad:", w.grad)
    print("b's grad:", b.grad)


if __name__ == '__main__':
    autograd_scalar()
    autograd_tensor()
