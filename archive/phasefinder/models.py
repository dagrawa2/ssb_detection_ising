import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sympy.combinatorics import Permutation
from sympy.combinatorics import PermutationGroup

from .utils import inverse_softplus


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


class MRA(nn.Module):

	def __init__(self, input_dim, group, broken=True):
		super(MRA, self).__init__()
		self.input_dim = input_dim
		self.group = group
		self.broken = broken

		self.variance_unconstrained = nn.Parameter(torch.tensor(0.541), requires_grad=False)
		if broken:
			with torch.no_grad():
				init_mean = torch.normal(torch.zeros(input_dim), torch.ones(input_dim))
				init_mean = 3*init_mean/torch.norm(init_mean)
			self.mean = nn.Parameter(init_mean, requires_grad=False)

	def EM(self, X, iters=1):
		with torch.no_grad():
			if not self.broken:
				variance = X.pow(2).sum()/X.size(0)
				self.variance_unconstrained.copy_(inverse_softplus(variance))
			else:
				mean = X.mean(0)
				variance = (X - mean.unsqueeze(0)).pow(2).sum()/X.size(0)
				GX = self.group.orbit(X)
				for _ in range(iters):
					weights = torch.exp( torch.einsum("gij,j->gi", GX, mean)/variance )
					mean_new = ( (GX*weights.unsqueeze(-1)).sum(0)/weights.unsqueeze(-1).sum(0) ).mean(0)
					variance = ( mean.pow(2).sum() + X.pow(2).sum()/X.size(0) - 2*mean.dot(mean_new) )/self.input_dim
					mean = mean_new
				self.mean.copy_(mean)
				self.variance_unconstrained.copy_(inverse_softplus(variance))

	def forward(self, X):
		variance = F.softplus(self.variance_unconstrained)
		if not self.broken:
			return 1/(2*variance)*X.pow(2).sum()/X.size(0) + self.input_dim/2*torch.log(2*np.pi*variance) - np.log(self.group.order)
		GX = self.group.orbit(X)
		return -torch.logsumexp(-(GX - self.mean.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)/(2*variance), 0).mean() + self.input_dim/2*torch.log(2*np.pi*variance)


class PFNet(nn.Module):

	def __init__(self, observable, group, regularizer_weight=0.1):
		super(PFNet, self).__init__()
		self.observable = observable
		self.group = group
		self.regularizer_weight = regularizer_weight

		self.aligner_1 = MRA(observable.output_dim, group, broken=False)
		self.aligner_2 = MRA(observable.output_dim, group, broken=True)

	def regularizer(self):
#		return self.regularizer_weight *( F.relu( self.group.orbit(self.aligner_2.mean).mv(self.aligner_2.mean)/self.aligner_2.mean.pow(2).sum() ).mean() + 1/self.aligner_2.mean.pow(2).sum())
		return 2*self.regularizer_weight * 1/( self.aligner_2.mean.pow(2).sum() - self.group.orbit(self.aligner_2.mean).mv(self.aligner_2.mean).mean() )

	def forward(self, X, broken=True):
		if broken:
			return self.aligner_2(self.observable(X))
		return self.aligner_1(self.observable(X))


class IsingObservable(nn.Module):

	def __init__(self, input_dim):
		super(IsingObservable, self).__init__()
		self.input_dim = input_dim
		self.output_dim = 2

		self.linear = nn.Linear(1, 1)
		self.conv = nn.Conv1d(1, 1, kernel_size=input_dim, padding="same", padding_mode="circular", bias=False)

	def forward(self, X):
		Q = self.conv(X.unsqueeze(1)).squeeze(1).pow(2).sum(1, keepdim=True)
		Z = X.sum(1, keepdim=True)
		out = torch.cat([self.linear(Z), self.linear(-Z)], 1) + Q
		return out
