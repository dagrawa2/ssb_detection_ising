import os
import json
import numpy as np
from scipy.linalg import lstsq

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)


def build_path(results_dir, J, observable_name, L, N=None, fold=None, seed=None, L_test=None, subdir=None):
	if subdir is not None:
		results_dir = os.path.join(results_dir, subdir)
	path = os.path.join(results_dir, J, observable_name)
	if L is not None:
		path = os.path.join(path, "L{:d}".format(L))
	if observable_name == "magnetization":
		return path
	path = os.path.join(path, "N{:d}".format(N), "fold{:d}".format(fold), "seed{:d}".format(seed))
	if L_test is None or L_test == L:
		return path
	path = os.path.join(path, "L{:d}".format(L_test))
	return path


### plot magnetization, order parameter curves, and U_4 Binder cumulant curves

def plot_stats(results_dir, J, observable_name, L, N=None, fold=None, seed=None, fit_color="black", tc_color="black"):
	dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed")
	with np.load(os.path.join(dir, "stats.npz")) as fp:
		stats = dict(fp)
	with np.load(os.path.join(dir, "tc.npz")) as fp:
		tc_estimate = fp["means"][-1]+fp["biases"][-1]
	output_dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="plots")
	os.makedirs(output_dir, exist_ok=True)
	# distributions
	stats["distribution_range"] = stats["distribution_range"].tolist() if "distribution_range" in stats else [-1, 1]
	plt.figure()
	plt.imshow(np.flip(stats["distributions"].T, 0), cmap="gray_r", vmin=0, vmax=1, extent=(stats["temperatures"].min(), stats["temperatures"].max(), *stats["distribution_range"]), aspect="auto")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color=tc_color, linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "distribution.png"))
	plt.close()
	# order parameter
	plt.figure()
	plt.plot(stats["temperatures"], stats["order_means"], color="black")
	plt.plot(stats["temperatures"], stats["order_means"]-stats["order_stds"], color="black", linestyle="dashed")
	plt.plot(stats["temperatures"], stats["order_means"]+stats["order_stds"], color="black", linestyle="dashed")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color=tc_color, linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "order.png"))
	plt.close()
	# U_4 Binder cumulant
	mask = stats["temperatures"] < tc_estimate
	not_mask = np.logical_not(mask)
	step_fit = np.array([stats["u4_means"][mask].mean()]*mask.sum() + [stats["u4_means"][not_mask].mean()]*not_mask.sum())
	plt.figure()
	plt.plot(stats["temperatures"], stats["u4_means"], color="black")
	plt.plot(stats["temperatures"], stats["u4_means"]-stats["u4_stds"], color="black", linestyle="dashed")
	plt.plot(stats["temperatures"], stats["u4_means"]+stats["u4_stds"], color="black", linestyle="dashed")
	plt.plot(stats["temperatures"], step_fit, color=fit_color, linestyle="dashed")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color=tc_color, linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "u4.png"))
	plt.close()


### plot error vs time data

def fit_error_vs_time(results_dir, J, observable_name, extrapolate=False, biased=False, jitter=0):
	tc = 2/np.log(1+np.sqrt(2))
	suffix = "_extrapolate" if extrapolate else ""
	with np.load(os.path.join(results_dir, "processed", J, observable_name, "tc_vs_time{}.npz".format(suffix))) as data:
		D = dict(data)
	if extrapolate:
		D = {key: value.reshape((-1)) for (key, value) in D.items()}
	if biased:
		D["tc_means"] = D["tc_means"] + D["tc_biases"]
	x = np.log(D["times"]/60 + jitter)
	X = np.stack([np.ones_like(x), x], 1)
	y = np.log(100*np.abs(D["tc_means"]/tc-1) + jitter)
	w = lstsq(X, y)[0]
	x = D["times"]/60
	y = 100*np.abs(D["tc_means"]/tc-1)
	w[0] = np.exp(w[0])
	return x, y, w


def plot_error_vs_time(results_dir, J, observable_names, observable_labels, observable_colors, extrapolate=False, biased=False):
	jitter = 1e-4 if extrapolate else 0
	plt.figure()
	for (name, label, color) in zip(observable_names, observable_labels, observable_colors):
		x, y, w = fit_error_vs_time(results_dir, J, name, extrapolate=extrapolate, biased=biased, jitter=jitter)
		x_pnts = np.linspace(x.min(), x.max(), 1000, endpoint=True)
		yhat = w[0]*x_pnts**w[1]
		plt.scatter(x, y, color=color, label=label)
		plt.plot(x_pnts, yhat, color=color)
	plt.legend(loc="upper right", bbox_to_anchor=(1, 1), fancybox=True, fontsize=10)
	plt.xscale("log")
	plt.yscale("log")
	plt.xlabel("Time (min)", fontsize=12)
	plt.ylabel("Abs error (%)", fontsize=12)
	suffix = "_extrapolate" if extrapolate else ""
	output_dir = os.path.join(results_dir, "plots", J)
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, "tc_vs_time{}.png".format(suffix)))
	plt.close()


### tabulate symmetry generators

def tabulate_generators(results_dir, Js, encoder_name):
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(results_dir, "processed", "generators.json"), "r") as fp:
		gens = json.load(fp)
	for J in Js:
		gens[J] = gens[J][encoder_name]
	generator_types = ["spatial", "internal"]
	stds = [gens[J][gen_type]["std"] for J in Js for gen_type in generator_types]
	max_precision = 1 + max([1-int(np.log10(s)) for s in stds	])
	S_columns = 2*"S[table-format=-1.{:d}(2),table-align-uncertainty=true]".format(max_precision)
	with open(os.path.join(output_dir, "generators.tex"), "w") as fp:
		fp.write("\\begin{{tabular}}{{c{}}}\n".format(S_columns))
		fp.write("\\toprule\n")
		fp.write("\\quad & Spatial & Internal \\\\\n")
		fp.write("\\midrule\n")
		for J in Js:
			fp.write(J.capitalize())
			for gen_type in generator_types:
				precision = 2-int(np.log10(gens[J][gen_type]["std"]))
				fp.write(" & {{:.{:d}f}}\\pm {{:.{:d}f}}".format(precision, precision).format(gens[J][gen_type]["mean"], gens[J][gen_type]["std"]))
			fp.write(" \\\\\n")
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


if __name__ == "__main__":
	results_dir = "results5"
	Js = ["ferromagnetic", "antiferromagnetic"]

	tc_color = "red"
	fit_color = "blue"

	observable_names = ["magnetization", "latent", "latent_equivariant", "latent_multiscale"]
	observable_labels = ["Magnetization", "Baseline-encoder", "GE-encoder", "GE-encoder (multiscale)"]
	observable_colors = ["red", "green", "blue", "purple"]

	print("Plotting statistics . . . ")
	for J in Js:
		for name in ["magnetization", "latent", "latent_equivariant"]:
			plot_stats(results_dir, J, name, 128, N=256, fold=0, seed=0, fit_color=fit_color, tc_color=tc_color)

	print("Plotting error vs time . . . ")
	for J in Js:
		plot_error_vs_time(results_dir, J, observable_names[:-1], observable_labels[:-1], observable_colors[:-1], extrapolate=False, biased=True)
		plot_error_vs_time(results_dir, J, observable_names, observable_labels, observable_colors, extrapolate=True, biased=True)

	print("Tabulating generators . . . ")
	tabulate_generators(results_dir, Js, "latent_equivariant")

	print("Done!")
