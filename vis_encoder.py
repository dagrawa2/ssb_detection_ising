import os
import random
import itertools
import numpy as np
import torch
import polytope as pc
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

from phasefinder import jackknife


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def lrelu(X):
	return np.where(X>=0, X, 0.01*X)

def load_encoder_params(results_dir, J, encoder_name, L, N, fold, seed):
	params = torch.load(os.path.join(results_dir, J, encoder_name, "L{:d}".format(L), "N{:d}".format(N), "fold{:d}".format(fold), "seed{:d}".format(seed), "encoder.pth"))
	params = {n: p.numpy() for (n, p) in params.items()}
	params["linear2.weight"] = params["linear2.weight"].squeeze(0)
	params["linear2.bias"] = params["linear2.bias"].item()
	return params

def make_encoder_func(params):
	def func(x):
		x = np.atleast_2d(x)
		params_linear1_bias = np.atleast_2d(params["linear1.bias"])
		y = lrelu(x.dot(params["linear1.weight"].T)+params_linear1_bias).dot(params["linear2.weight"]) + params["linear2.bias"]
		if y.size == 1:
			y = y.item()
		return y
	return func

def get_scale(encoder_func, J="ferromagnetic", L=None):
	if L is None:
		lattice = np.array([1, 1]) if J == "ferromagnetic" \
			else np.array([1, -1])
	else:
		lattice = np.ones((L**2)) if J == "ferromagnetic" \
				else np.tile(np.array([[1, -1], [-1, 1]]), [L//2, L//2]).reshape((-1))
	scale = np.max(np.abs(encoder_func( np.stack([lattice, -lattice], 0) )))
	return scale

def rescale(params, encoder_func, J="ferromagnetic", L=None):
	scale = get_scale(encoder_func, J=J, L=L)
	params["linear2.weight"] = params["linear2.weight"]/scale
	params["linear2.bias"] = params["linear2.bias"]/scale
	func = make_encoder_func(params)
	return params, func	


def plot_regions(results_dir, J, encoder_name, L, N, folds, seeds):
	x_norm, y_norm, x_angle, y_angle = [], [], [], []
	for (fold, seed) in itertools.product(folds, seeds):
		params = load_encoder_params(results_dir, J, encoder_name, L, N, fold, seed)
		encoder_func = make_encoder_func(params)
		params, encoder_func = rescale(params, encoder_func, J=J)
		A = params["linear2.weight"][:,None]*params["linear1.weight"]
		b = params["linear2.weight"]*params["linear1.bias"]
		A_borders = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
		b_borders = np.array([1, 1, 1, 1])
		signatures = list(map(lambda s: np.array(list(s)), itertools.product([0, 1], repeat=4)))
		for s in signatures:
			signs = 1 - 2*s
			A_region = np.concatenate((signs[:,None]*A, A_borders), 0)
			b_region = np.concatenate((signs*b, b_borders), 0)
			area = pc.volume(pc.Polytope(A_region, b_region))
			area *= 100/4
			leaks = np.clip(s+0.01, 0, 1)
			gradient = np.sum(leaks[:,None]*A, 0)
			norm = np.sqrt(np.sum(gradient**2))
			angle = np.real( -1j*np.log((gradient[0]+gradient[1]*1j)/norm) )
			angle *= 180/np.pi
			x_norm.append(norm)
			y_norm.append(area)
			x_angle.append(angle)
			y_angle.append(area)
	output_dir = os.path.join(results_dir, "plots", J)
	os.makedirs(output_dir, exist_ok=True)
	# gradient norms
	plt.figure()
	plt.scatter(x_norm, y_norm, color="black")
	plt.xlabel("Gradient norm")
	plt.ylabel("Area (%)")
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "region_norm.png"))
	plt.close()
	# gradient angles
	plt.figure()
	plt.scatter(x_angle, y_angle, color="black")
	plt.xlabel("Gradient angle")
	plt.ylabel("Area (%)")
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "region_angle.png"))
	plt.close()


def plot_magnitude(results_dir, J, L, N, fold, seed, observable_names, observable_labels, observable_colors, n_points=100):
	x = np.linspace(-1, 1, n_points, endpoint=True)
	plt.figure()
	for (name, label, color) in zip(observable_names, observable_labels, observable_colors):
		if name == "magnetization":
			plt.plot(x, x, color=color, label=label)
			continue
		params = load_encoder_params(results_dir, J, name, L, N, seed, fold)
		encoder_func = make_encoder_func(params)
		if name == "latent_equivariant":
			params, encoder_func = rescale(params, encoder_func, J=J)
			lattice = np.array([1, 1]) if J == "ferromagnetic" \
				else np.array([1, -1])
		else:
			params, encoder_func = rescale(params, encoder_func, J=J, L=L)
			lattice = np.ones((L**2)) if J == "ferromagnetic" \
				else np.tile(np.array([[1, -1], [-1, 1]]), [L//2, L//2]).reshape((-1))
		y = encoder_func(np.outer(x, lattice))
		plt.plot(x, y, color=color, label=label)
	plt.xlabel("Magnetization")
	plt.ylabel("Encoder")
	plt.tight_layout()
	output_dir = os.path.join(results_dir, "plots", J)
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, "magnitude.png"))
	plt.close()


if __name__ == "__main__":
	results_dir = "results6"
	Js = ["ferromagnetic", "antiferromagnetic"]
	observable_names = ["magnetization", "latent", "latent_equivariant"]
	observable_labels = ["Magnetization", "Baseline-encoder", "GE-encoder"]
	observable_colors = ["red", "green", "blue"]

	folds = [0, 1, 2, 3]
	seeds = [0, 1, 2]

	for J in Js:
		print(J)
		plot_regions(results_dir, J, "latent_equivariant", 128, 256, folds, seeds)
		plot_magnitude(results_dir, J, 128, 256, 0, 0, observable_names, observable_labels, observable_colors, n_points=100)

	print("Done!")
