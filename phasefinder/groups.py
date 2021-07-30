import numpy as np

class Element(object):

	def __init__(self, a_1, a_2, m, q, s, rotation_order=4, translation_order=2):
		self.a_1 = a_1
		self.a_2 = a_2
		self.m = m
		self.q = q
		self.s = s

		self.rotation_order = rotation_order
		self.translation_order = translation_order
		self.value = [a_1, a_2, m, q, s]

	def rho(self, 		a_1, a_2, m):
		for _ in range(m%self.rotation_order):
			temp = -a_2
			a_2 = a_1
			a_1 = temp
		return a_1, a_2

	def tau(self, a_1, a_2, q):
		if q%2 == 0:
			return a_1, a_2
		else:
			return -a_1, a_2

	def __eq__(self, other):
		return self.value == other.value

	def __hash__(self):
		return hash((self.a_1, self.a_2, self.m, self.q, self.s, self.rotation_order, self.translation_order))

	def __mul__(self, other):
		a_1, a_2 = self.tau(other.a_1, other.a_2, self.q)
		a_1, a_2 = self.rho(a_1, a_2, self.m)
		m = other.m if self.q==0 else -other.m
		return Element((self.a_1+a_1)%self.translation_order, (self.a_2+a_2)%self.translation_order, (self.m+m)%self.rotation_order, (self.q+other.q)%2, (self.s+other.s)%2, rotation_order=self.rotation_order, translation_order=self.translation_order)

	def inverse(self):
		a_1, a_2 = self.tau(self.a_1, self.a_2, self.q)
		a_1, a_2 = self.rho(a_1, a_2, self.m)
		m = -self.m if self.q==0 else self.m
		return Element((-a_1)%self.translation_order, (-a_2)%self.translation_order, m%self.rotation_order, self.q, self.s)

	def conjugate(self, other):
		return other.inverse()*self.__mul__(other)

	def action(self, X):
		if self.s==1:
			X = -X
		if self.q==1:
			X = np.flip(X, -2)
		if self.m > 0:
			X = np.rot90(X, self.m, axes=(-2, -1))
		if self.a_1 > 0:
			X = np.roll(X, self.a_1, axis=-2)
		if self.a_2 > 0:
			X = np.roll(X, self.a_2, axis=-1)
		return X


def generate_group(rotation_order=4, translation_order=2):
	G = [Element(a_1, a_2, m, q, s, rotation_order=rotation_order, translation_order=translation_order) \
		for a_1 in range(translation_order) \
		for a_2 in range(translation_order) \
		for m in range(rotation_order) \
		for q in range(2) \
		for s in range(2) \
		]
	return set(G)

def generate_subgroup(generators):
	G = set(generators)
	order = len(G)
	while True:
		G = G.union([g*h for g in G for h in G])
		if len(G) == order:
			break
		else:
			order = len(G)
	return G

def generate_normal_subgroup(generators, G):
	GinvHG = set(generators)
	if len(GinvHG) == len(G):
		return GinvHG
	while True:
		H = generate_subgroup(GinvHG)
		GinvHG = H.union( set([g.inverse()*h*g for h in H for g in G-H]) )
		if len(GinvHG) == len(H):
			break
	return H
