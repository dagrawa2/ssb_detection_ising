import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim, symmetric=False):
		super(Encoder, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.symmetric = symmetric

		self.activation = nn.LeakyReLU()

		if symmetric:
			self.L = input_dim[0]
			self.conv = nn.Conv2d(1, hidden_dim, kernel_size=self.L, stride=2, padding=(self.L//2-1, self.L//2-1), padding_mode="circular")
			self.pool = nn.AvgPool2d(self.L//2)
			self.linear = nn.Linear(hidden_dim, output_dim)
		else:
			self.linear1 = nn.Linear(input_dim, hidden_dim)
			self.linear2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, X):
		if self.symmetric:
			out = self.linear( self.pool(self.activation(self.conv(X))).squeeze(-1).squeeze(-1) )
		else:
			out = self.linear2(self.activation(self.linear1(X)))
		if out.ndim == 1:
			out = torch.reshape(out, [-1, 1])
		return out


class Decoder(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim, symmetric=False):
		super(Decoder, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.symmetric = symmetric

		self.activation = nn.LeakyReLU()
		self.linear1 = nn.Linear(input_dim+1, hidden_dim)

		if symmetric:
			self.L = output_dim[0]
			self.linear2 = nn.Linear(hidden_dim, 2)
		else:
			self.linear2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, X, T):
		out = self.linear2(self.activation(self.linear1(torch.cat([X, T], 1))))
		if self.symmetric:
			out = torch.tile(torch.stack([out, torch.flip(out, [1])], 2), [1, self.L//2, self.L//2]).unsqueeze(1)
		return out
