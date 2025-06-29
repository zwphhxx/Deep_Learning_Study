import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary


def create_dataset():
    # read dataset
    cell_phone_dataset = pd.read_csv("./data/cell_phone_datasets.csv")
    x, y = cell_phone_dataset.iloc[:, :-1], cell_phone_dataset.iloc[:, -1]

    zscore_scaler = preprocessing.StandardScaler()
    x = zscore_scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)

    # Convert to numpy arrays first
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.int64)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.int64)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # create dataloader to iter data
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return train_dataloader, test_dataloader, test_dataset


class CellPhonePricePredictionModel(nn.Module):
    def __init__(self):
        super(CellPhonePricePredictionModel, self).__init__()
        self.layer1 = nn.Linear(in_features=20, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=4)
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        out = self.out(x)
        return out


def model_train(train_dataloader):
    torch.manual_seed(42)
    model = CellPhonePricePredictionModel()

    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        loss_sum = 0
        sample = 0.01
        for x, y in train_dataloader:
            y_predict = model(x)
            loss = criterion(y_predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            sample += 1
        print(loss_sum / sample)
    torch.save(model.state_dict(), './model/cell_phone_price_prediction_model.pth')


def model_predict(test_dataloader, test_dataset):
    model = CellPhonePricePredictionModel()
    model.load_state_dict(torch.load('./model/cell_phone_price_prediction_model.pth'))
    correct = 0
    for x, y in test_dataloader:
        y_predict = model(x)
        y_index = torch.argmax(y_predict, dim=1)
        correct += (y_index == y).sum()
    acc = correct.item() / len(test_dataset)
    print(acc)


if __name__ == '__main__':
    train_dataloader, test_dataloader, test_dataset = create_dataset()
    price_prediction_model = CellPhonePricePredictionModel()
    summary(price_prediction_model, input_size=(20,), batch_size=10)
    model_train(train_dataloader)
    model_predict(test_dataloader,test_dataset)
