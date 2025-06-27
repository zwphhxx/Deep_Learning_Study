import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression  # make linear regression datasets
import torch
from torch import optim  # optimizer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def create_regression_datasets():
    x, y, coef = make_regression(n_samples=300,
                                 n_features=1,
                                 noise=15,
                                 coef=True,
                                 bias=1.5,
                                 random_state=42)
    x = torch.tensor(x)
    y = torch.tensor(y)
    print(f"x.shape:{x.shape}, y.shape:{y.shape}")
    return x, y, coef


def linear_regression_model(x, y, coef):
    dateset = TensorDataset(x, y)
    dataLoader = DataLoader(dateset, batch_size=2, shuffle=True)

    # create linear regression model
    linear_regression_model = torch.nn.Linear(in_features=1, out_features=1)

    # MSE function
    criterion = torch.nn.MSELoss()

    # optimizer function -> SGD
    optimizer = torch.optim.SGD(linear_regression_model.parameters(), lr=0.01)

    epochs = 100
    # loss parameters
    loss_epoch = []
    total_loss = 0.0
    train_sample = 0.0
    for _ in range(epochs):
        for train_x, train_y in dataLoader:
            # Use model to predict the train data
            y_pred = linear_regression_model(train_x.type(torch.float32))

            # Loss calculation
            loss = criterion(y_pred, train_y.reshape(-1, 1).type(torch.float32))
            total_loss += loss.item()
            train_sample += len(train_x)

            # gradient initialize
            optimizer.zero_grad()

            # Automatic Differentiation
            loss.backward()

            # SGD
            optimizer.step()

        # update average loss of each epoch
        loss_epoch.append(total_loss / train_sample)
    return epochs, loss_epoch, linear_regression_model


if __name__ == '__main__':
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
