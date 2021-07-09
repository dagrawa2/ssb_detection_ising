import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sympy.combinatorics import Permutation
from sympy.combinatorics import PermutationGroup


class Group(object):

	def __init__(self, degree, generators):
		self.degree = degree
		self.generators = generators
		for g in generators:
			assert isinstance(g, list), "Each group generator must be a list of tuples."
			for h in g:
				assert isinstance(h, tuple), "Each group generator must be a list of tuples."

		self.group = PermutationGroup(*[Permutation(g, size=degree) for g in generators])
		self.elements = [g.list() for g in self.group._elements]
		self.order = self.group.order()

	def orbit(self, X):
		return torch.stack([X[..., g] for g in self.elements], 0)


class GMM(nn.Module):

	def __init__(self, input_dim, group):
		super(GMM, self).__init__()
		self.input_dim = input_dim
		self.group = group

	def initialize(self, X):
		with torch.no_grad():
			self.mean = X.mean(0)
			self.variance = (X - self.mean.unsqueeze(0)).pow(2).sum()/X.size(0)
			self.second_moment = X.pow(2).sum()/X.size(0)
			self.GX = self.group.orbit(X)

	def update(self):
		with torch.no_grad():
			weights = torch.exp( torch.einsum("gij,j->gi", self.GX, self.mean)/self.variance )
			mean_new = ( (self.GX*weights.unsqueeze(-1)).sum(0)/weights.unsqueeze(-1).sum(0) ).mean(0)
			self.variance = ( self.second_moment - 2*self.mean.dot(mean_new) + self.mean.pow(2).sum() )/self.input_dim
			self.mean = mean_new

	def loss(self):
		with torch.no_grad():
			return -torch.logsumexp(-(self.GX - self.mean.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)/(2*self.variance), 0).mean() + self.input_dim/2*torch.log(2*np.pi*self.variance) + np.log(self.group.order)


class Encoder(nn.Module):

	def __init__(self, input_dims, hidden_dim, output_dim):
		super(Encoder, self).__init__()
		self.input_dims = list(input_dims)
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.conv = nn.Conv2d(1, hidden_dim, kernel_size=input_dims, padding="same", padding_mode="circular", bias=False)
		self.pool = nn.MaxPool2d(input_dims)
		self.activation = nn.LeakyReLU()
		self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

	def forward(self, X):
		out_conv = self.conv(X)
		out_1 = self.linear(self.activation( self.pool(out_conv).squeeze(-1).squeeze(-1) ))
		out_2 = self.linear(self.activation( self.pool(-out_conv).squeeze(-1).squeeze(-1) ))
		return torch.cat([out_1, -out_2], 1)


class Decoder(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dims):
		super(Decoder, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dims = list(output_dims)

		self.activation = nn.LeakyReLU()
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, 1)

	def forward(self, X):
		out_single = self.linear2(self.activation(self.linear1(X)))
		return out_single.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, *self.output_dims])
