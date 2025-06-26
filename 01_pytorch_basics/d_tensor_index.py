import torch

print(f"{'-' * 30}original data{'-' * 30}")
torch.random.manual_seed(42)
tensor_mat = torch.randint(1, 10, [5, 6])

print(f"tensor_mat->\n{tensor_mat}")

print(f"{'-' * 30}Row and column indexing of tensors{'-' * 30}")
# Return the tensor data of the second row.
tensor_row2 = tensor_mat[1]
print(f"tensor_row2->{tensor_row2}")

# Return the tensor data of the fourth column.
tensor_col4 = tensor_mat[:, 3]
print(f"tensor_col4->{tensor_col4}")

# Return the tensor data at coordinates (1, 3).
tensor_row2_col4 = tensor_mat[1, 3]
print(f"tensor_row2_col4->{tensor_row2_col4}")

# Return the tensor data at coordinates (1, 2) and (1, 4).
tensor_coordinates = tensor_mat[[1, 1], [2, 4]]
print(f"tensor_coordinates->{tensor_coordinates}")

# Return rows 2 and 3 of columns 1 and 4.
tensor_cross = tensor_mat[[[1], [2]], [0, 3]]
print(f"tensor_cross->\n{tensor_cross}")

print(f"{'-' * 30}Range indexing of tensors{'-' * 30}")
# The first 2 columns of the first 3 rows of data.
tensor_cols_rows = tensor_mat[:3, :2]
print(f"tensor_cols_rows->\n{tensor_cols_rows}")
# The first 2 columns of the last 2 rows of data.
tensor_cols_rows = tensor_mat[2:, :2]
print(f"tensor_cols_rows->\n{tensor_cols_rows}")
tensor_cols_rows = tensor_mat[2:, 1:6:2]
print(f"tensor_cols_rows->\n{tensor_cols_rows}")

print(f"{'-' * 30}Boolean indexing of tensors{'-' * 30}")

# Rows where the scalar value in the third column is greater than 3.
tensor_bool = tensor_mat[tensor_mat[:, 2] > 3]
print(f"tensor_bool->\n{tensor_bool}")

# Columns where the scalar value in the second row is less than 6.
tensor_bool = tensor_mat[:, tensor_mat[1] < 6]
print(f"tensor_bool->\n{tensor_bool}")

print(f"{'-' * 30}Multidimensional indexing{'-' * 30}")
multidim_tensor = torch.randint(1, 10, [3, 5, 6])
print(f"multidim_tensor->\n{multidim_tensor}")

# Get the first element along axis 0.
dim0_tensor = multidim_tensor[0, :, :]
# Get the first element along axis 1.
dim1_tensor = multidim_tensor[:, 0, :]
# Get the first element along axis 2.
dim2_tensor = multidim_tensor[:, :, 0]
print(f"dim0_tensor->\n{dim0_tensor}")
print(f"dim1_tensor->\n{dim1_tensor}")
print(f"dim2_tensor->\n{dim2_tensor}")