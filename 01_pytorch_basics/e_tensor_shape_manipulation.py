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

print(f"{'-' * 30}squeeze unsqueeze{'-' * 30}")

torch_data_unsqueeze_dim0 = torch_data.unsqueeze(dim=0)
torch_data_unsqueeze_dim1 = torch_data.unsqueeze(dim=1)
torch_data_unsqueeze_dim_1 = torch_data.unsqueeze(dim=-1)
print(f"torch_data_shape->\n{torch_data.shape}")
print(f"torch_data_unsqueeze_dim0_shape->\n{torch_data_unsqueeze_dim0.shape}")
print(f"torch_data_unsqueeze_dim1_shape->\n{torch_data_unsqueeze_dim1.shape}")
print(f"torch_data_unsqueeze_dim_1_shape->\n{torch_data_unsqueeze_dim_1.shape}")
print(f"torch_data->\n{torch_data}")
print(f"torch_data_unsqueeze_dim0->\n{torch_data_unsqueeze_dim0}")
print(f"torch_data_unsqueeze_dim1->\n{torch_data_unsqueeze_dim1}")
print(f"torch_data_unsqueeze_dim_1->\n{torch_data_unsqueeze_dim_1}")

print(f"torch_data_unsqueeze_dim_1_squeeze_shape->\n{torch_data_unsqueeze_dim_1.squeeze().shape}")
print(f"torch_data_unsqueeze_dim_1_squeeze->\n{torch_data_unsqueeze_dim_1.squeeze()}")

print(f"{'-' * 30}transpose permute{'-' * 30}")
print(f"torch_data->\n{torch_data}")
print(f"torch_data_shape->\n{torch_data.shape}")
print(f"torch_data_transpose->\n{torch.transpose(torch_data,0,1)}")
print(f"torch_data_transpose_shape->\n{torch.transpose(torch_data,0,1).shape}")
print(f"torch_data_transpose->\n{torch.transpose(torch_data,0,2)}")
print(f"torch_data_transpose_shape->\n{torch.transpose(torch_data,0,2).shape}")

print(f"torch_data_permute->\n{torch.permute(torch_data,[1,0,2])}")
print(f"torch_data_permute_shape->\n{torch.permute(torch_data,[1,0,2]).shape}")

print(f"{'-' * 30}view contiguous{'-' * 30}")
# is_contiguous-> whether the tensor data is in one memory
print(f"torch_data_contiguous->{torch_data.is_contiguous()}")
print(f"torch_data_transpose_contiguous->{torch.transpose(torch_data,0,1).is_contiguous()}")
# contiguous-> the tensor data is transformed to contiguous
print(f"torch_data_transpose_contiguous->{torch.transpose(torch_data,0,1).contiguous().is_contiguous()}")

# view cannot transform uncontiguous tensor data
print(f"torch_data_view->\n{torch_data.view(-1)}")
print(f"torch_data_view_shape->{torch_data.view(-1).shape}")
print(f"torch_data_view_contiguous->{torch_data.view(-1).is_contiguous()}")