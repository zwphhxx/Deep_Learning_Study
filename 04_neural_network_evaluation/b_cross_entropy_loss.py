import torch.nn as nn
import torch


def cross_entropy_loss():
    y_true = torch.tensor([1, 2], dtype=torch.int64)
    y_true = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    y_pred = torch.tensor([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]], dtype=torch.float32)
    loss = nn.CrossEntropyLoss()
    my_loss = loss(y_pred, y_true).numpy()
    print(f"cross_entropy_loss->{my_loss}")


def binary_cross_entropy_loss():
    y_true = torch.tensor([0, 1, 0], dtype=torch.float32)
    y_pred = torch.tensor([0.621, 0.342, 0.593], requires_grad=True)
    criterion = nn.BCELoss()
    loss = criterion(y_pred, y_true).detach().numpy()
    print(f"binary_cross_entropy_loss->{loss}")

def mean_absolute_error():
    y_true = torch.tensor([2.0,3.0,1.0], dtype=torch.float32)
    y_pred = torch.tensor([5.0,4.0,1.0], dtype=torch.float32)
    loss = nn.L1Loss()
    print(f"mean_absolute_error->{loss(y_pred, y_true).numpy()}")

def mean_square_error():
    y_true = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)
    y_pred = torch.tensor([5.0, 4.0, 1.0], dtype=torch.float32)
    loss = nn.MSELoss()
    print(f"mean_square_error->{loss(y_pred, y_true).numpy()}")

def smooth_l1():
    y_true = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)
    y_pred = torch.tensor([5.0, 4.0, 1.0], dtype=torch.float32)
    loss = nn.SmoothL1Loss()
    print(f"smooth_l1->{loss(y_pred, y_true).numpy()}")

if __name__ == '__main__':
    cross_entropy_loss()
    binary_cross_entropy_loss()
    mean_absolute_error()
    mean_square_error()
    smooth_l1()
