import torch
import torch.nn.functional as F
import torch.nn as nn

#Default = Uniform Random Initialization
def init_default():
    linear = nn.Linear(in_features=5, out_features=3)
    print(f"{'-' * 30}Default Initialization Weight{'-' * 30}\n{linear.weight.data}\n")
    print(f"{'-' * 30}Default Initialization Bias{'-' * 30}\n{linear.bias.data}\n")

def init_uniform():
    # 5 in features 3 out features
    linear = nn.Linear(in_features=5, out_features=3)
    nn.init.uniform_(linear.weight)
    print(f"{'-' * 30}Uniform Random Initialization{'-' * 30}\n{linear.weight.data}\n")


def init_constant():
    linear = nn.Linear(in_features=5, out_features=3)
    nn.init.constant_(linear.weight, 2)
    print(f"{'-' * 30}Fixed Initialization{'-' * 30}\n{linear.weight.data}\n")


def init_zero():
    linear = nn.Linear(in_features=5, out_features=3)
    nn.init.zeros_(linear.weight)
    print(f"{'-' * 30}Zero Initialization{'-' * 30}\n{linear.weight.data}\n")


def init_ones():
    linear = nn.Linear(in_features=5, out_features=3)
    nn.init.ones_(linear.weight)
    print(f"{'-' * 30}Ones Initialization{'-' * 30}\n{linear.weight.data}\n")


def init_normal():
    linear = nn.Linear(in_features=5, out_features=3)
    nn.init.normal_(linear.weight, mean=0, std=1)
    print(f"{'-' * 30}Normal Distribution Random Initialization{'-' * 30}\n{linear.weight.data}\n")


def init_kaiming():
    linear = nn.Linear(in_features=5, out_features=3)
    nn.init.kaiming_normal_(linear.weight)
    print(f"{'-' * 30}Kaiming Normal Distribution Random Initialization{'-' * 30}\n{linear.weight.data}\n")

    linear = nn.Linear(in_features=5, out_features=3)
    nn.init.kaiming_uniform_(linear.weight)
    print(f"{'-' * 30}Kaiming Uniform Random Initialization{'-' * 30}\n{linear.weight.data}\n")

def init_xavier():
    linear = nn.Linear(in_features=5, out_features=3)
    nn.init.xavier_uniform_(linear.weight)
    print(f"{'-' * 30}Xavier Random Initialization{'-' * 30}\n{linear.weight.data}\n")

    linear = nn.Linear(in_features=5, out_features=3)
    nn.init.xavier_normal_(linear.weight)
    print(f"{'-' * 30}Xavier Normal Distribution Initialization{'-' * 30}\n{linear.weight.data}\n")


if __name__ == '__main__':
    init_default()
    init_uniform()
    init_constant()
    init_zero()
    init_ones()
    init_normal()
    init_kaiming()
    init_xavier()
