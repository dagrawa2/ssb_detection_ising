import os
import torch
import numpy as np


def signed_f(x):
	if x < 0:
		return "- {:.3f}".format(np.abs(x))
	if x > 0:
		return "+ {:.3f}".format(np.abs(x))
	if x == 0:
		return "+ 0"


Js = ["ferromagnetic", "antiferromagnetic"]
Ls = [16, 32, 64, 128]

for J in Js:
	print("{}:".format(J))
	for L in Ls:
		params = torch.load(os.path.join("results_saved_models", J, "latent_equivariant", "L{:d}".format(L), "encoder.pth"))
		params = {n: p.numpy() for (n, p) in params.items()}
		norms = np.sqrt(np.sum(params["linear1.weight"]**2, 1))
		params["linear2.weight"] = params["linear2.weight"]*norms[None,:]
		params["linear1.weight"] = params["linear1.weight"]/norms[:,None]
		params["linear1.bias"] = params["linear1.bias"]/norms
		for i in range(4):
			print("y_{:d}[{:d}] = {:.3f} phi([{:.3f}, {:.3f}] x {})" \
				.format(L, i, params["linear2.weight"][0,i], params["linear1.weight"][i,0], params["linear1.weight"][i,1], signed_f(params["linear1.bias"][i])))
	print()


# ===
print("\n===\n")

import itertools
import polytope as pc

def linear_region_info(J, L):
	params = torch.load(os.path.join("results_saved_models", J, "latent_equivariant", "L{:d}".format(L), "encoder.pth"))
	params = {n: p.numpy() for (n, p) in params.items()}
	A = params["linear2.weight"].T*params["linear1.weight"]
	b = params["linear2.weight"].squeeze(0)*params["linear1.bias"]
	A_borders = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
	b_borders = np.array([1, 1, 1, 1])
	signatures = list(map(lambda s: np.array(list(s)), itertools.product([0, 1], repeat=4)))
	print("area, gradient_norm, gradient_angle")
	for s in signatures:
		signs = 1 - 2*s
		A_region = np.concatenate((signs[:,None]*A, A_borders), 0)
		b_region = np.concatenate((signs*b, b_borders), 0)
		area = pc.volume(pc.Polytope(A_region, b_region))
		area *= 100/4
		if area <= 1e-3:
			continue
		leaks = np.clip(s+0.01, 0, 1)
		gradient = np.sum(leaks[:,None]*A, 0)
		norm = np.sqrt(np.sum(gradient**2))
		angle = np.real( -1j*np.log((gradient[0]+gradient[1]*1j)/norm) )
		angle *= 180/np.pi
		print("{:.0f}%, {:.2f}, {:.0f}".format(area, norm, angle))
	print()

linear_region_info("ferromagnetic", 128)
