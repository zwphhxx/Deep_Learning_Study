import torch
import matplotlib.pyplot as plt

DAYS_NUMBER = 30
torch.manual_seed(42)
temperature = torch.randn(size=[DAYS_NUMBER, ]) * 10


def create_weather_data():
    print(f"temperature->{temperature}")
    days = torch.arange(1, DAYS_NUMBER + 1, 1)
    plt.plot(days, temperature, color='r')
    plt.scatter(days, temperature)
    plt.show()


def exponent_weight_average(beta=0.9):
    exp_weight_avg = []
    for idx, temp in enumerate(temperature,1):
        print(idx, temp)
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)
    days = torch.arange(1, DAYS_NUMBER + 1, 1)
    plt.plot(days, exp_weight_avg, color='r')
    plt.scatter(days, temperature)
    plt.show()


if __name__ == '__main__':
    create_weather_data()
    exponent_weight_average()
    exponent_weight_average(beta=0.5)
