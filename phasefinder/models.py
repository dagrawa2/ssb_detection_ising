import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim):
		super(MLP, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.activation = nn.LeakyReLU()

		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, X):
		out = self.linear2(self.activation(self.linear1(X)))
		if out.ndim == 1:
			out = torch.reshape(out, [-1, 1])
		return out
