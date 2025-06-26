import torch
import numpy as np

# Tensor type Short Int Float Double
randn_tensor = torch.randn(2, 4)
print(f"randn_tensor.dtype->{randn_tensor.dtype}\n")

# Tensor type conversion.
print(f"Int.dtype->{randn_tensor.type(torch.IntTensor).dtype}")
print(f"Int.dtype->{randn_tensor.int().dtype}\n")

print(f"{'-'*30}tensor2ndarray{'-'*30}")
torch.manual_seed(42)

# Conversion from tensor to ndarray.
tensor_mat = torch.randint(0,100,[2,4])
numpy_mat = tensor_mat.numpy()
print(f"tensor_mat->\n{tensor_mat}")
print(f"numpy_mat->\n{numpy_mat}\n")
print(f"tensor_mat->\n{type(tensor_mat)}")
print(f"numpy_mat->\n{type(numpy_mat)}\n")

print(f"{'-'*30}tensor2ndarray without copy{'-'*30}")

numpy_mat[0][0]=2
print(f"numpy_mat->\n{numpy_mat}\n")
print(f"numpy_mat->\n{tensor_mat}\n")

print(f"{'-'*30}tensor2ndarray copy{'-'*30}")

# Note: After conversion, the tensor and ndarray share the same memory, which can lead to synchronized modifications.
# To avoid this issue, you can use the copy() function to create a new memory block.
tensor_mat = torch.randint(0,100,[2,4])
numpy_mat = tensor_mat.numpy().copy()
print(f"tensor_mat->\n{type(tensor_mat)}")
print(f"numpy_mat->\n{type(numpy_mat)}\n")

numpy_mat[0][0]=2
print(f"numpy_mat->\n{numpy_mat}\n")
print(f"numpy_mat->\n{tensor_mat}\n")

print(f"{'-'*30}ndarray2tensor{'-'*30}")

# Conversion from ndarray to tensor.
# 'from_numpy' converts an ndarray to a tensor with shared memory,
# while 'torch.tensor' converts an ndarray to a tensor without shared memory.
np.random.seed(42)
numpy_mat = np.random.randint(0,100,[2,4])
tensor_mat = torch.from_numpy(numpy_mat)
print(f"tensor_mat->\n{tensor_mat}")
print(f"numpy_mat->\n{numpy_mat}\n")

print(f"numpy_mat->\n{type(numpy_mat)}")
print(f"tensor_mat->\n{type(tensor_mat)}\n")

print(f"{'-'*30}tensor2ndarray from_numpy without copy{'-'*30}")

tensor_mat[0][0]=2
print(f"tensor_mat->\n{tensor_mat}")
print(f"numpy_mat->\n{numpy_mat}\n")

print(f"{'-'*30}tensor2ndarray from_numpy copy{'-'*30}")

tensor_mat = torch.from_numpy(numpy_mat.copy())
tensor_mat[0][0]=20
print(f"tensor_mat->\n{tensor_mat}")
print(f"numpy_mat->\n{numpy_mat}\n")

print(f"{'-'*30}tensor2ndarray torch.tensor{'-'*30}")

tensor_mat = torch.tensor(numpy_mat)
tensor_mat[0][0]=10
print(f"tensor_mat->\n{tensor_mat}")
print(f"numpy_mat->\n{numpy_mat}\n")

print(f"{'-'*30}tensor2scalar{'-'*30}")

# For a tensor with only one element, use item() to extract its value.
scalar_tensor = torch.tensor(10)
print(f"scalar_tensor->{scalar_tensor}")
print(f"scalar->{scalar_tensor.item()}\n")