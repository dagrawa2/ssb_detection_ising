import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim):
		super(Encoder, self).__init__()
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


class Decoder(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim):
		super(Decoder, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.activation = nn.LeakyReLU()
		self.linear1 = nn.Linear(input_dim+1, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, X, T):
		return self.linear2(self.activation(self.linear1(torch.cat([X, T], 1))))
