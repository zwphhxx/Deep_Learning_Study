import torch
torch.manual_seed(42)
torch_data = torch.randint(0,10,[2,3,6])

print(f"{'-' * 30}shape size() reshape(){'-' * 30}")

print(f"torch_data->\n{torch_data}")
print(f"torch_data shape->{torch_data.shape}")
print(f"torch_data shape0->{torch_data.shape[0]}")
print(f"torch_data shape1->{torch_data.shape[1]}")
print(f"torch_data shape2->{torch_data.shape[2]}\n")

print(f"torch_data size->{torch_data.size()}")
print(f"torch_data size0->{torch_data.size(0)}")
print(f"torch_data size1->{torch_data.size(1)}")
print(f"torch_data size2->{torch_data.size(2)}\n")

torch_data_reshape = torch_data.reshape(3,4,3)
print(f"torch_data_reshape->\n{torch_data_reshape}")
print(f"torch_data_reshape shape->{torch_data_reshape.shape}")

torch_data_reshape = torch_data.reshape(4,3,-1)
print(f"torch_data_reshape->\n{torch_data_reshape}")
print(f"torch_data_reshape shape->{torch_data_reshape.shape}")