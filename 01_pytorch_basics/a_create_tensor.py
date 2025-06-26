import torch
import numpy as np

# Create a tensor from a scalar
data = torch.tensor(1234)
print(f"torch.tensor(1234)->{data}")
print(f"type->{type(data)}\n")

# Create a tensor from a NumPy array.
ndarray_rand = np.random.rand(2,4)
print(f"ndarray_rand->\n{ndarray_rand}")
print(f"torch_ndarray->\n{torch.tensor(ndarray_rand)}\n")

# Create a tensor from a List.
list_float = [[1.2,2.,4.5],[3.,2.3,5.3]]
print(f"list_float->{list_float}")
print(f"torch_list_float->\n{torch.tensor(list_float)}\n")

# Create a tensor with a specified number of rows and columns.
specified_tensor = torch.Tensor(2,4) # 2 rows and 4 columns
print(f"specified_tensor->\n{specified_tensor}")
# Attention: If a list is passed in, a tensor corresponding to the list will be created,
# rather than a tensor with the specified number of rows and columns.
list_tensor = torch.Tensor([2,4])
print(f"list_tensor->{list_tensor}\n")

# Tensor APIs for different data types.
int_matrix_tensor = torch.IntTensor(3,2)
print(f"int_matrix_tensor->{int_matrix_tensor}")
int_tensor = torch.IntTensor([2.3,4.2])
print(f"int_tensor->{int_tensor}") # DeprecationWarning: an integer is required (got type float).

print(f"torch.ShortTensor()->{torch.ShortTensor([3,2])}") # tensor([], dtype=torch.int16)
print(f"torch.LongTensor()->{torch.LongTensor([3,2])}") # tensor([], dtype=torch.int64)
print(f"torch.FloatTensor()->{torch.FloatTensor([3,2])}") # tensor([], dtype=torch.float32)
print(f"torch.DoubleTensor()->{torch.DoubleTensor([3,2])}") # tensor([], dtype=torch.float64)