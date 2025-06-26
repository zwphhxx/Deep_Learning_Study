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
print(f"torch.DoubleTensor()->{torch.DoubleTensor([3,2])}\n") # tensor([], dtype=torch.float64)

# Create a linear tensor.
linear_tensor_arange = torch.arange(1,15,3)
print(f"linear_tensor_arange->\n{linear_tensor_arange}")
linear_tensor_linspace = torch.linspace(13,22,3) # Evenly split
print(f"linear_tensor_linspace->\n{linear_tensor_linspace}\n")

# Create a random tensor.
randn_tensor = torch.randn(2,4)
print(f"randn_tensor->\n{randn_tensor}")
seed = torch.random.initial_seed()
print(f"torch random seed->\n{seed}") # 15646168311400

torch.manual_seed(seed)
specified_randn_tensor = torch.randn(2,4)
print(f"specified_randn_tensor->\n{specified_randn_tensor}\n")

# Create an zeros / ones / full tensor.
ones_tensor = torch.ones(2,4)
print(f"ones_tensor->\n{ones_tensor}")
ones_like_torch = torch.ones_like(ones_tensor)
print(f"ones_like_torch->\n{ones_like_torch}")

zeros_tensor = torch.zeros(2,4)
print(f"zeros_tensor->\n{zeros_tensor}")
zeros_like_torch = torch.zeros_like(int_matrix_tensor)
print(f"zeros_like_torch->\n{zeros_like_torch}")

full_tensor = torch.full((4,5),30)
print(f"full_tensor->\n{full_tensor}")
full_like_tensor = torch.full_like(zeros_tensor,50)
print(f"full_like_tensor->\n{full_like_tensor}")