import torch

data1 = torch.randint(0,10,[2,2,3])
data2 = torch.randint(0,10,[2,2,3])

# Concatenate along dimension 0
data_concat_0 = torch.cat([data1,data2],dim=0)
print(f"data_concat_0->\n{data_concat_0}")
print(f"data_concat_0_shape->{data_concat_0.shape}")

data1 = torch.randint(0,10,[2,1,3])
data2 = torch.randint(0,10,[2,2,3])

# Concatenate along dimension 1
# Concatenated tensors must have matching dimensions except along the specified axis.
data_concat_0 = torch.cat([data1,data2],dim=1)
print(f"data_concat_1->\n{data_concat_0}")
print(f"data_concat_1_shape->{data_concat_0.shape}")

data1 = torch.randint(0,10,[2,3,2])
data2 = torch.randint(0,10,[2,3,4])

# Concatenate along dimension 2
# Concatenated tensors must have matching dimensions except along the specified axis.
data_concat_0 = torch.cat([data1,data2],dim=2)
print(f"data_concat_2->\n{data_concat_0}")
print(f"data_concat_2_shape->{data_concat_0.shape}")