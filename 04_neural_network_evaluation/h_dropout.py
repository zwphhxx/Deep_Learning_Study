import torch.nn as nn
import torch


def dropout():
    dropout = nn.Dropout(p=0.75)
    inputs = torch.randint(0, 10, size=[1, 4]).float()
    layer = nn.Linear(4, 5)
    y = layer(inputs)
    print(f"FC without dropout: \n{y}")

    y = dropout(y)
    print(f"FC with dropout: \n{y}")


if __name__ == '__main__':
    dropout()
