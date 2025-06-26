import torch

print(f"{'-' * 30}add sub mul div neg{'-' * 30}")

torch.random.manual_seed(42)
operand = torch.randint(1, 10, [3, 4])
print(f"operand->\n{operand}\n")

result = operand.add(4)
print(f"result add 4->\n{result}\n")

result = operand.sub(2)
print(f"result sub 2->\n{result}\n")

result = operand.mul(1.5)
print(f"result mul 1.5->\n{result}\n")

result = operand.div(2, rounding_mode="trunc")
print(f"result div 2 trunc->\n{result}\n")

result = operand.neg()
print(f"result neg->\n{result}\n")

print(f"{'-' * 30}add_ sub_ mul_ div_ neg_{'-' * 30}")

operand = torch.randint(1, 10, [3, 4])
print(f"operand->\n{operand}\n")

operand.add_(4)
print(f"operand add_ 4->\n{operand}\n")

operand.sub_(2)
print(f"operand sub_ 2->\n{operand}\n")

operand.mul_(3)
print(f"operand mul_ 3->\n{operand}\n")

operand.div_(2, rounding_mode="trunc")
print(f"operand div_ 2 trunc->\n{operand}\n")

operand.neg_()
print(f"operand neg_->\n{operand}\n")

print(f"{'-' * 30}Hadamard{'-' * 30}")

A = torch.tensor([[3, 4], [9, 10]])
B = torch.tensor([[2, 2], [1, 2]])

result1 = torch.mul(A, B)
result2 = A * B
print(f"result Hadamard \n A:\n{A} \nB\n{B}\nmul(A,B)\n{result1}\nA*B\n{result1}\n")

print(f"{'-' * 30}matmul{'-' * 30}")
torch.random.manual_seed(42)
A = torch.randint(1, 10, [3, 4])
B = torch.randint(1, 10, [4, 5])
result3= torch.matmul(A, B)
result4 = A@B

print(f"result matmul \n A:\n{A} \nB\n{B}\nmatmul(A,B)\n{result3}\nA@B\n{result4}\n")

print(f"{'-' * 30}mean sum sqrt pow log{'-' * 30}")

torch.random.manual_seed(42)
operand = torch.randint(1, 10, [3, 4],dtype=torch.float64)

print(f"operand->\n{operand}\n")

operand_mean = operand.mean()
operand_mean_dim0 = operand.mean(dim=0) # column
operand_mean_dim1 = operand.mean(dim=1) # row
print(f"operand_mean->{operand_mean}")
print(f"operand_mean_dim0->{operand_mean_dim0}")
print(f"operand_mean_dim1->{operand_mean_dim1}\n")

operand_sum = operand.sum()
operand_sum_dim0 = operand.sum(dim=0) # column
operand_sum_dim1 = operand.sum(dim=1) # row
print(f"operand_sum->{operand_sum}")
print(f"operand_sum_dim0->{operand_sum_dim0}")
print(f"operand_sum_dim1->{operand_sum_dim1}\n")

operand_pow2 = torch.pow(operand, 2)
print(f"operand_pow2->\n{operand_pow2}\n")
operand_2pow = torch.pow(2,operand)
print(f"operand_2pow->\n{operand_2pow}\n")
operand_exp = operand.exp()
print(f"operand_exp->\n{operand_exp}\n")
operand_log = operand.log()
print(f"operand_log->\n{operand_log}\n")
operand_log2 = operand.log2()
print(f"operand_log2->\n{operand_log2}\n")
operand_log10 = operand.log10()
print(f"operand_log10->\n{operand_log10}\n")

operand_sqrt = operand.sqrt()
print(f"operand_sqrt->\n{operand_sqrt}\n")


