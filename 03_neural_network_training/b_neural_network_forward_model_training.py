import torch.nn
from torch import nn  # Neural network modules
from torchsummary import summary  # For model summary visualization


class NeuralNetwork(nn.Module):
    def __init__(self):
        """Initialize neural network layers and weights"""
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(in_features=3, out_features=3)  # First linear layer
        self.layer2 = nn.Linear(in_features=3, out_features=2)  # Second linear layer
        self.out = nn.Linear(in_features=2, out_features=2)  # OUTPUT layer

        # Initialize weights with different methods
        # NOTE: Using different initialization methods per layer is unusual in practice
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.kaiming_normal_(self.layer2.weight)
        nn.init.uniform_(self.out.weight)

    def forward(self, x):
        """Define forward pass through network"""
        # Pass through first linear layer
        x_layer1 = self.layer1(x)
        x_layer1 = torch.sigmoid(x_layer1)

        # Pass through second linear layer
        x_layer2 = self.layer2(x_layer1)
        x_layer2 = torch.relu(x_layer2)

        # Pass through output layer
        x_out = self.out(x_layer2)

        # Convert to probabilities via softmax
        x_out = torch.softmax(x_out, dim=-1)
        return x_out


if __name__ == '__main__':
    # Instantiate model
    neural_network_model = NeuralNetwork()

    # Create random input tensor (batch_size=5, features=3)
    input_value = torch.randn(5, 3)

    print(f"{'-' * 30}input_value{'-' * 30}\n{input_value}\n")

    # Forward pass through model
    output_value = neural_network_model(input_value)

    print(f"{'-' * 30}output_value{'-' * 30}\n{output_value}\n")

    # Display model architecture summary
    # input_size: (input_features,) batch_size: number of samples per batch
    print(f"{'-' * 30}Torchsummary Report{'-' * 30}\n")
    summary(neural_network_model, input_size=(3,), batch_size=5)

    # Print all parameter
    print(f"{'-' * 30}Layer Parameters{'-' * 30}")
    for name, param in neural_network_model.named_parameters():
        print(f"{name} : {param}\n")
