import torch
import matplotlib.pyplot as plt

def stepLR():
    LR = 0.1
    iteration = 10
    max_epoch = 200

    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0],requires_grad=True)

    optimizer = torch.optim.SGD([w], lr=LR,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    lr_list, epoch_list = list(), list()

    for epoch in range(max_epoch):
        lr_list.append(scheduler.get_last_lr())
        epoch_list.append(epoch)
        for i in range(iteration):
            loss = ((w*x-y_true)**2)/2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    plt.plot(epoch_list, lr_list,label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.show()

def multi_stepLR():
    LR = 0.1
    iteration = 10
    max_epoch = 200

    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0],requires_grad=True)

    optimizer = torch.optim.SGD([w], lr=LR,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,90,130,180], gamma=0.5)

    lr_list, epoch_list = list(), list()

    for epoch in range(max_epoch):
        lr_list.append(scheduler.get_last_lr())
        epoch_list.append(epoch)
        for i in range(iteration):
            loss = ((w*x-y_true)**2)/2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    plt.plot(epoch_list, lr_list,label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.show()

def exp_stepLR():
    LR = 0.1
    iteration = 10
    max_epoch = 200

    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0],requires_grad=True)

    optimizer = torch.optim.SGD([w], lr=LR,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    lr_list, epoch_list = list(), list()

    for epoch in range(max_epoch):
        lr_list.append(scheduler.get_last_lr())
        epoch_list.append(epoch)
        for i in range(iteration):
            loss = ((w*x-y_true)**2)/2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    plt.plot(epoch_list, lr_list,label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    stepLR()
    multi_stepLR()
    exp_stepLR()