import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression  # For generating synthetic regression datasets
import torch
from torch import optim  # For optimization algorithms (e.g., SGD)
from torch.utils.data import DataLoader # For batching and shuffling data
from torch.utils.data import TensorDataset # For creating PyTorch datasets from tensors


def create_regression_datasets():
    """
      Generates a synthetic linear regression dataset using scikit-learn.
      :returns:
        x: Feature data (independent variable) as PyTorch tensor
        y: Target values (dependent variable) as PyTorch tensor
        coef: True slope coefficient of the underlying linear relationship
    """
    # Generate synthetic regression data with known parameters
    x, y, coef = make_regression(n_samples=300,
                                 n_features=1,
                                 noise=15,
                                 coef=True,
                                 bias=1.5,
                                 random_state=42)
    # Convert numpy arrays to PyTorch tensors
    x = torch.tensor(x)
    y = torch.tensor(y)
    print(f"x.shape:{x.shape}, y.shape:{y.shape}")
    return x, y, coef


def linear_regression_model(x, y, coef):
    """
   Trains a linear regression model using PyTorch's torch.nn module.
    :param x:Feature data (X)
    :param y:Target values (Y)
    :param coef:True coefficient
    :return:
        epochs: Total number of training iterations
        loss_epoch: List of average loss per epoch
        linear_regression_model: Trained PyTorch model
    """

    # Create TensorDataset and DataLoader for efficient batching
    dateset = TensorDataset(x, y)

    dataLoader = DataLoader(dateset,
                            batch_size=6,   # Mini-batch size
                            shuffle=True,   # Shuffle data each epoch
                            drop_last=False # Keep last batch even if smaller
                            )

    # Define linear regression model (single input, single output)
    linear_regression_model = torch.nn.Linear(in_features=1, out_features=1)

    # Mean Squared Error loss function
    criterion = torch.nn.MSELoss()

    # Stochastic Gradient Descent optimizer
    optimizer = torch.optim.SGD(linear_regression_model.parameters(), lr=0.01)

    # Training configuration
    epochs = 100        # Number of complete passes through the dataset
    loss_epoch = []     # Store average loss per epoch
    total_loss = 0.0    # Accumulate loss within epoch
    train_sample = 0.0  # Count of samples processed in epoch

    # Training loop
    for _ in range(epochs):
        # Iterate through data in batches
        for train_x, train_y in dataLoader:

            # Forward pass: compute predicted y
            y_pred = linear_regression_model(train_x.type(torch.float32))

            # Compute loss between predictions and true values
            # Reshape train_y to match prediction shape [batch_size, 1]
            loss = criterion(y_pred, train_y.reshape(-1, 1).type(torch.float32))

            # Accumulate loss and sample count
            total_loss += loss.item()
            train_sample += len(train_x)

            # Reset gradients to zero before backward pass
            optimizer.zero_grad()

            # Automatic Differentiation
            loss.backward()

            # Backward pass: compute gradients
            optimizer.step()

        # Calculate average loss for this epoch
        loss_epoch.append(total_loss / train_sample)
    return epochs, loss_epoch, linear_regression_model


if __name__ == '__main__':
    # Create dataset and train model
    x, y, coef = create_regression_datasets()
    epochs, loss_epoch, linear_regression_model = linear_regression_model(x, y, coef)

    # Plotting the loss change curve and fitted linear regression plot using matplotlib
    plt.plot(range(epochs), loss_epoch)
    plt.title("Loss Change Curve")
    plt.grid()
    plt.show()
    x = torch.linspace(x.min(), x.max(), 300)

    # True Linear Regression Equation
    y_linear = torch.tensor([coef * xi.numpy() + 1.5
                             for xi in x])

    # Fitted Linear Regression Equation
    y_model = torch.tensor([linear_regression_model.weight.detach() * xi
                            + linear_regression_model.bias
                            for xi in x])
    plt.scatter(x, y)
    plt.plot(x, y_linear, label="True Linear Regression")
    plt.plot(x, y_model, label="Fitted Linear Regression")
    plt.legend()
    plt.grid()
    plt.show()
