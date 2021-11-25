import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from phasefinder.utils import build_path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=8)
matplotlib.rc("ytick", labelsize=8)


### plot magnetization, order parameter curves, and U_4 Binder cumulant curves

def subplot_stat(results_dir, J, observable_name, L, N=None, fold=None, seed=None, colors=None, what="distribution", xlabel=True, ylabel=True, title=None):
	dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed")
	with np.load(os.path.join(dir, "stats.npz")) as fp:
		stats = dict(fp)
	with np.load(os.path.join(dir, "tc.npz")) as fp:
		tc_estimate = fp["mean"]+fp["bias"]
	tc_exact = 2/np.log(1+np.sqrt(2))
	if what == "distribution":
		stats["distribution_range"] = stats["distribution_range"].tolist() if "distribution_range" in stats else [-1, 1]
		plt.imshow(np.flip(stats["distributions"].T, 0), cmap="gray_r", vmin=0, vmax=1, extent=(stats["temperatures"].min(), stats["temperatures"].max(), *stats["distribution_range"]), aspect="auto")
		plt.axvline(x=tc_exact, linestyle="dashed", color=colors["tc"])
		if xlabel:
			plt.xlabel(r"Temperature ($T$)", fontsize=8)
		if ylabel:
			plt.ylabel(J.capitalize()+"\n"+r"Observable ($\mathcal{O}$)", fontsize=8)
		if title is not None:
			plt.title(title, fontsize=8)
	if what == "order":
		plt.plot(stats["temperatures"], stats["order_means"], color="black")
		plt.plot(stats["temperatures"], stats["order_means"]-stats["order_stds"], color="black", linestyle="dashed")
		plt.plot(stats["temperatures"], stats["order_means"]+stats["order_stds"], color="black", linestyle="dashed")
		plt.axvline(x=tc_exact, linestyle="dashed", color=colors["tc"])
		if xlabel:
			plt.xlabel(r"Temperature ($T$)", fontsize=8)
		if ylabel:
			plt.ylabel(J.capitalize()+"\n"+r"Mean Abs Obs ($\langle|\mathcal{O}|\rangle$)", fontsize=8)
		if title is not None:
			plt.title(title, fontsize=8)
	if what == "binder":
		mask = stats["temperatures"] < tc_estimate
		not_mask = np.logical_not(mask)
		step_fit = np.array([stats["u4_means"][mask].mean()]*mask.sum() + [stats["u4_means"][not_mask].mean()]*not_mask.sum())
		plt.plot(stats["temperatures"], stats["u4_means"], color="black")
		plt.plot(stats["temperatures"], stats["u4_means"]-stats["u4_stds"], color="black", linestyle="dashed")
		plt.plot(stats["temperatures"], stats["u4_means"]+stats["u4_stds"], color="black", linestyle="dashed")
		plt.plot(stats["temperatures"], step_fit, linestyle="dashed", color=colors["fit"])
		plt.axvline(x=tc_exact, linestyle="dashed", color=colors["tc"])
		plt.xlabel(r"$T$", fontsize=12)
		if xlabel:
			plt.xlabel(r"Temperature ($T$)", fontsize=8)
		if ylabel:
			plt.ylabel(J.capitalize()+"\n"+r"Binder ($U_4$)", fontsize=8)
		if title is not None:
			plt.title(title, fontsize=8)


def plot_stat(results_dir, Js, observable_names, L, N=None, fold=None, seed=None, colors=None, titles=None, what="distribution"):
	plt.figure()
	nrows, ncols = len(Js), len(observable_names)
	for (index, (J, name)) in enumerate(itertools.product(Js, observable_names)):
		plt.subplot(nrows, ncols, index+1)
		xlabel = index//ncols == nrows-1
		ylabel = index%ncols == 0
		title = titles[name] if index//ncols==0 else None
		subplot_stat(results_dir, J, name, L, N=N, fold=fold, seed=seed, colors=colors, what=what, xlabel=xlabel, ylabel=ylabel, title=title)
	plt.tight_layout()
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, "{}.png".format(what)))
	plt.close()


### plot critical temperature estimates

def y_minmax(means, stds, padding=0.05):
	y_min = (means-stds).min()
	y_max = (means+stds).max()
	y_range = y_max-y_min
	y_min = max(0, y_min - padding*y_range)
	y_max = y_max + padding*y_range
	return y_min, y_max


def bar_width_shifts(n_bars):
	total_width = 0.7
	width = total_width/n_bars
	shifts = np.array([-total_width/2 + total_width/(2*n_bars)*(2*m+1) for m in range(n_bars)])
	return width, shifts


def get_unique_legend_handles_labels(fig):
	tuples = [(h, l) for ax in fig.get_axes() for (h, l) in zip(*ax.get_legend_handles_labels())]
	handles, labels = zip(*tuples)
	unique = [(h, l) for (i, (h, l)) in enumerate(zip(handles, labels)) if l not in labels[:i]]
	handles, labels = zip(*unique)
	return list(handles), list(labels)


def subplot_tc(results_dir, J, L, Ns, encoder_names, labels, colors, xlabel=True, ylabel=True, title=None):
	data = pd.read_csv(os.path.join(results_dir, "processed", "gathered.csv"))
	data = data[data.J.eq(J) & data.L.eq(str(L))]
	y_min, y_max = y_minmax(data.tc_mean.values, data.tc_std.values)
	x = np.arange(len(Ns))
	augment = lambda x: np.stack([np.zeros_like(x), x], 0)
	width, shifts = bar_width_shifts(len(encoder_names))
	for (name, shift) in zip(encoder_names, shifts):
		plt.bar(x+shift, data[data.observable.eq(name)].tc_mean.values, width, yerr=augment(data[data.observable.eq(name)].tc_std.values), capsize=5, ecolor=colors[name], color=colors[name], label=labels[name])
	plt.axhline(y=data[data.observable.eq("magnetization")].tc_mean.values[0], linestyle="dashed", color=colors["magnetization"], label=labels["magnetization"])
	plt.xticks(x, Ns)
	plt.ylim(y_min, y_max)
	if xlabel:
		plt.xlabel(r"Samples per temperature ($N$)", fontsize=8)
	if ylabel:
		plt.ylabel("Error (%)", fontsize=8)
	if title is not None:
		plt.title(title, fontsize=8)


def plot_tc(results_dir, J, Ls, Ns, encoder_names, labels, colors, grid_dims=None):
	plt.figure()
	nrows, ncols = grid_dims
	for (index, L) in enumerate(Ls):
		plt.subplot(nrows, ncols, index+1)
		xlabel = index//ncols == nrows-1
		ylabel = index%ncols == 0
		title = r"$L = {:d}$".format(L)
		subplot_tc(results_dir, J, L, Ns, encoder_names, labels, colors, xlabel=xlabel, ylabel=ylabel, title=title)
	handles, labels = get_unique_legend_handles_labels(plt.gcf())
	plt.figlegend(handles, labels, ncol=2, loc="upper center", fancybox=True, fontsize=8)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, "tc_{}.png".format(J)))
	plt.close()


def plot_tc_extrapolate(results_dir, Js, Ns, encoder_names, labels, colors):
	plt.figure()
	nrows, ncols = 1, len(Js)
	for (index, J) in enumerate(Js):
		plt.subplot(nrows, ncols, index+1)
		xlabel = index//ncols == nrows-1
		ylabel = index%ncols == 0
		title = J.capitalize()
		subplot_tc(results_dir, J, None, Ns, encoder_names, labels, colors, xlabel=xlabel, ylabel=ylabel, title=title)
	handles, labels = get_unique_legend_handles_labels(plt.gcf())
	plt.figlegend(handles, labels, ncol=2, loc="upper center", fancybox=True, fontsize=8)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, "tc_extrapolate.png"))
	plt.close()


### plot error vs lattice size data

def subplot_tc_vs_lattice(results_dir, J, Ls, observable_names, labels, colors, N=None, fold=None, seed=None, xlabel=True, ylabel=True, title=None):
	tc_exact = 2/np.log(1+np.sqrt(2))
	tc2err = lambda tc: 100*(tc/tc_exact-1)
	x = 1/np.array(Ls)
	x_pnts = np.linspace(0, x.max(), 100, endpoint=True)
	for name in observable_names:
		y = []
		for L in Ls:
			with np.load(os.path.join(build_path(results_dir, J, name, L, N=N, fold=fold, seed=seed, subdir="processed"), "tc.npz")) as fp:
				y.append( fp["mean"]+fp["bias"] )
		with np.load(os.path.join(build_path(results_dir, J, name, None, N=N, fold=fold, seed=seed, subdir="processed"), "tc.npz")) as fp:
			slope, intercept = fp["slope_mean"]+fp["slope_bias"], fp["yintercept_mean"]+fp["yintercept_bias"]
		yhat = slope*x_pnts + intercept
		y, yhat = tc2err(y), tc2err(yhat)
		plt.scatter(x, y, alpha=0.7, color=colors[name], label=labels[name])
		plt.plot(x_pnts, yhat, alpha=0.7, color=colors[name])
	if xlabel:
		plt.xlabel(r"Inverse lattice size ($L^{-1}$)", fontsize=8)
	if ylabel:
		plt.ylabel("Error (%)", fontsize=8)
	if title is not None:
		plt.title(title, fontsize=8)


def plot_tc_vs_lattice(results_dir, Js, Ls, observable_names, labels, colors, N=None, fold=None, seed=None):
	plt.figure()
	nrows, ncols = 1, len(Js)
	for (index, J) in enumerate(Js):
		plt.subplot(nrows, ncols, index+1)
		xlabel = index//ncols == nrows-1
		ylabel = index%ncols == 0
		title = J.capitalize()
		subplot_tc_vs_lattice(results_dir, J, Ls, observable_names, labels, colors, N=N, fold=fold, seed=seed, xlabel=xlabel, ylabel=ylabel, title=title)
	handles, labels = get_unique_legend_handles_labels(plt.gcf())
	plt.figlegend(handles, labels, ncol=3, loc="upper center", fancybox=True, fontsize=8)
	plt.tight_layout()
	plt.subplots_adjust(top=0.9)
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, "tc_vs_L.png"))
	plt.close()


### plot execution times

def subplot_time(results_dir, J, Ls, N, encoder_names, labels, colors, xlabel=True, ylabel=True, title=None):
	encoder_names_singlescale = [name for name in encoder_names if "multiscale" not in name]
	data = pd.read_csv(os.path.join(results_dir, "processed", "gathered.csv"))
	data = data[data.N==N]
	total_times = (data.generation_time+data.preprocessing_time+data.training_time).values
	y_min, y_max = y_minmax(total_times, np.zeros_like(total_times))
	data_Ls = data[(data.J==J) & (data.L!="None")]
	data_inf = data[(data.J==J) & (data.L=="None")]
	x = np.arange(len(Ls))
	width, shifts = bar_width_shifts(len(encoder_names_singlescale))
	for (name, shift) in zip(encoder_names_singlescale, shifts):
		subdata = data_Ls[data_Ls.observable.eq(name)]
		plt.bar(x+shift, subdata.generation_time.values, width, color=colors["magnetization"])
		plt.bar(x+shift, subdata.preprocessing_time.values+subdata.training_time.values, width, bottom=subdata.generation_time.values, color=colors[name])
	width, shifts = bar_width_shifts(len(encoder_names))
	for (name, shift) in zip(encoder_names, shifts):
		subdata = data_inf[data_inf.observable.eq(name)]
		plt.bar([len(x)+shift], subdata.generation_time.values, width, color=colors["magnetization"], label=labels["magnetization"])
		plt.bar([len(x)+shift], subdata.preprocessing_time.values+subdata.training_time.values, width, bottom=subdata.generation_time.values, color=colors[name], label=labels[name])
	plt.xticks(list(x)+[len(x)], list(Ls)+[r"$\infty$"])
	plt.ylim(y_min, y_max)
#	plt.yscale("log")
	if xlabel:
		plt.xlabel(r"Lattice size ($L$)", fontsize=8)
	if ylabel:
		plt.ylabel("Time (min)", fontsize=8)
	if title is not None:
		plt.title(title, fontsize=8)


def plot_time(results_dir, Js, Ls, N, encoder_names, labels, colors):
	plt.figure()
	nrows, ncols = 1, len(Js)
	for (index, J) in enumerate(Js):
		plt.subplot(nrows, ncols, index+1)
		xlabel = index//ncols == nrows-1
		ylabel = index%ncols == 0
		title = J.capitalize()
		subplot_time(results_dir, J, Ls, N, encoder_names, labels, colors, xlabel=xlabel, ylabel=ylabel, title=title)
	handles, labels = get_unique_legend_handles_labels(plt.gcf())
	plt.figlegend(handles, labels, ncol=2, loc="upper center", fancybox=True, fontsize=8)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, "time.png"))
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
	results_dir = "results4"
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	Ns = [8, 16, 32, 64, 128, 256]

	print("Plotting statistics . . . ")
	observable_names = ["magnetization", "latent", "latent_equivariant"]
	titles = {"magnetization": "Magnetization", "latent": "Baseline-AE", "latent_equivariant": "GE-AE"}
	colors = {"tc": "red", "fit": "blue"}
	for what in ["distribution", "order", "binder"]:
		plot_stat(results_dir, Js, observable_names, 128, N=256, fold=0, seed=0, colors=colors, titles=titles, what=what)

	print("Plotting error . . . ")
	encoder_names = ["latent", "latent_equivariant", "latent_multiscale_4"]
	labels = {"magnetization": "Magnetization", "latent": "Baseline-AE", "latent_equivariant": "GE-AE", "latent_multiscale_4": "GE-AE (multiscale)"}
	colors = {"magnetization": "red", "latent": "green", "latent_equivariant": "blue", "latent_multiscale_4": "purple"}
	for J in Js:
		plot_tc(results_dir, J, Ls, Ns, encoder_names, labels, colors, grid_dims=(2, 2))
	plot_tc_extrapolate(results_dir, Js, Ns, encoder_names, labels, colors)
	plot_tc_vs_lattice(results_dir, Js, Ls, observable_names, labels, colors, N=256, fold=0, seed=0)

	print("Plotting time . . . ")
	plot_time(results_dir, Js, Ls, 256, encoder_names, labels, colors)

	print("Tabulating generators . . . ")
	tabulate_generators(results_dir, Js, "latent_equivariant")

	print("Done!")
