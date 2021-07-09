import numpy as np
import torch

def inverse_softplus(y):
	return y + torch.log(1.0 - torch.exp(-y))
