import numpy as np
import torch

params = torch.load("results/params.pth")
params = {name: value.numpy() for (name, value) in params.items()}

print("Order parameter at lower temperature:")
print(np.round(params["aligner_2.mean"], 5))
print()

print("Coefficients of quadratic observable:")
print("A_0:", np.round(params["observable.linear.bias"].sum(), 5))
print("A_1:", np.round(params["observable.linear.weight"].sum(), 5))
print("||A_2||:", np.round(np.linalg.norm(params["observable.linear.bias"], ord=2), 5))
