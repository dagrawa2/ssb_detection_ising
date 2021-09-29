import os
import itertools
import numpy as np
import torch
import polytope as pc

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

np.random.seed(1)
torch.manual_seed(2)


def lrelu(X):
	return np.where(X>=0, X, 0.01*X)

def load_encoder_params(J, L):
	params = torch.load(os.path.join("results", J, "latent_equivariant", "L{:d}".format(L), "encoder.pth"))
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

def rescale(params, encoder_func, J="ferromagnetic"):
	lattice = np.array([1, 1]) if J == "ferromagnetic" \
		else np.array([1, -1])
	scale = np.max(np.abs(encoder_func( np.stack([lattice, -lattice], 0) )))
	params["linear2.weight"] = params["linear2.weight"]/scale
	params["linear2.bias"] = params["linear2.bias"]/scale
	func = make_encoder_func(params)
	return params, func	


def region_plot(J, L, N=1025):
	params = load_encoder_params(J, L)
	encoder_func = make_encoder_func(params)
	params, encoder_func = rescale(params, encoder_func, J=J)
	x = np.linspace(-1, 1, N, endpoint=True)
	y = np.linspace(-1, 1, N, endpoint=True)
	X, Y = np.meshgrid(x, y)
	Z = encoder_func( np.stack([X, Y], 2).reshape((-1, 2)) )
	Z = Z.reshape((N, N))
	y_boundaries = -( params["linear1.bias"][:,None] + np.outer(params["linear1.weight"][:,0], x) )/params["linear1.weight"][:,[1]]
	# figure
	fig, ax = plt.subplots()
	for y_boundary in y_boundaries:
		ax.plot(x, y_boundary, color="black", linestyle="dotted")
	im = ax.imshow(Z, interpolation="bilinear", origin="lower", cmap=cm.gray, extent=(-1, 1, -1, 1))
	CS = ax.contour(Z, origin="lower", cmap="flag", extend="both", extent=(-1, 1, -1, 1))
	# label contour lines
	ax.clabel(CS, inline=True, fontsize=10)
	# make a colorbar for the contour lines
	CB = fig.colorbar(CS, shrink=0.8)
	# We can still add a colorbar for the image, too.
	CBI = fig.colorbar(im, orientation="horizontal", shrink=0.8)
	# This makes the original colorbar look a bit out of place,
	# so let's improve its position.
	l, b, w, h = ax.get_position().bounds
	ll, bb, ww, hh = CB.ax.get_position().bounds
	CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])
	# titles
	ax.set_xlabel("Black magnetization")
	ax.set_ylabel("White magnetization")
	output_dir = "results/vis_encoder/{}/L{:d}".format(J, L)
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, "regions.png"))
	plt.close()


def region_table(J, L):
	params = load_encoder_params(J, L)
	encoder_func = make_encoder_func(params)
	params, encoder_func = rescale(params, encoder_func, J=J)
	A = params["linear2.weight"][:,None]*params["linear1.weight"]
	b = params["linear2.weight"]*params["linear1.bias"]
	A_borders = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
	b_borders = np.array([1, 1, 1, 1])
	signatures = list(map(lambda s: np.array(list(s)), itertools.product([0, 1], repeat=4)))
	output_dir = "results/vis_encoder/{}/L{:d}".format(J, L)
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join("results/vis_encoder", J, "L{:d}".format(L), "regions.tex"), "w") as fp:
		fp.write("\\begin{tabular}{ccc} \n")
		fp.write("\\toprule\n")
		fp.write("Area (%) & Grad norm & Grad angle (${}^\circ$) \\\\\n")
		fp.write("\\midrule\n")
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
			fp.write("{:.0f}% & {:.2f} & {:.0f} \\\\\n".format(area, norm, angle))
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


if __name__ == "__main__":

	print("Plotting . . . ")
	region_plot("ferromagnetic", 128, N=8)

	print("Tabulating . . . ")
	region_table("ferromagnetic", 128)

	print("Done!")
