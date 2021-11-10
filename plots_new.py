import os
import json
import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

def plot_stats(results_dir, J, observable_name, L, N=None, tc_color="black"):
	if N is not None:
		stats = dict(np.load(os.path.join(results_dir, "processed", J, observable_name, "L{:d}".format(L), "N{:d}".format(N), "stats.npz")))
		output_dir = os.path.join(results_dir, "plots", J, observable_name, "L{:d}".format(L), "N{:d}".format(N))
	else:
		stats = dict(np.load(os.path.join(results_dir, "processed", J, observable_name, "L{:d}".format(L), "stats.npz")))
		output_dir = os.path.join(results_dir, "plots", J, observable_name, "L{:d}".format(L))
	os.makedirs(output_dir, exist_ok=True)
	stats["distribution_range"] = stats["distribution_range"].tolist() if "distribution_range" in stats else [-1, 1]
	# distributions
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
	plt.figure()
	plt.plot(stats["temperatures"], stats["u4_means"], color="black")
	plt.plot(stats["temperatures"], stats["u4_means"]-stats["u4_stds"], color="black", linestyle="dashed")
	plt.plot(stats["temperatures"], stats["u4_means"]+stats["u4_stds"], color="black", linestyle="dashed")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color=tc_color, linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "u4.png"))
	plt.close()


def plot_critical_temperatures(results_dir, J, L, encoder_names, encoder_labels, encoder_colors, magnetization_color="black", tc_color="black", std_threshold_perc=10, remove_bias=True):
	L_str = L if type(L) == str else "L{:d}".format(L)
	output_dir = os.path.join(results_dir, "plots", J, L_str)
	os.makedirs(output_dir, exist_ok=True)
	suffix = "_biased" if not remove_bias else ""
	with open(os.path.join(results_dir, "processed", J, "tc{}.json".format(suffix)), "r") as fp:
		data = json.load(fp)[L_str]
	data["Ns"] = np.array(data["Ns"])
	for name in encoder_names:
		for stats in ["means", "stds"]:
			data[name][stats] = np.array(data[name][stats])[data["Ns"]>0]
			data[name][stats][data[name][stats]=="nan"] = np.nan
	data["Ns"] = data["Ns"][data["Ns"]>0]
	tc = 2/np.log(1+np.sqrt(2))
	std_threshold_abs = std_threshold_perc/100*tc
	stars = {name:i for name in encoder_names for i in range(len(data["Ns"])) if data[name]["stds"][i] > std_threshold_abs}
	for (name, i) in stars.items():
		data[name]["stds"][i] = 0
	x_min = np.round( np.nanmin(np.concatenate([data[name]["means"] for name in encoder_names], 0))-0.1, 1)
	x_max = np.round( np.nanmax(np.concatenate([data[name]["means"]+data[name]["stds"] for name in encoder_names], 0))+0.1, 1)
	for name in encoder_names:
		data[name]["means"][data[name]["means"]==np.nan] = x_min-1
		data[name]["stds"][data[name]["stds"]==np.nan] = 0
	y = np.arange(len(data["Ns"]))
	total_width = 0.7
	width = total_width/len(encoder_names)
	shifts = [-total_width/2 + total_width/(2*len(encoder_names))*(2*m+1) for m in range(len(encoder_names))]
	plt.figure()
	for (name, label, color, shift) in zip(encoder_names, encoder_labels, encoder_colors, shifts):
		bar = plt.barh(y+shift, data[name]["means"], width, xerr=data[name]["stds"], color=color, label=label)
		if name in stars:
			rect = bar[stars[name]]
			plt.text(rect.get_height(), rect.get_y()+shift, "*", ha="left", va="center")
	plt.axvline(x=tc, color=tc_color, linestyle="dashed")
	plt.axvline(x=data["magnetization"]["mean"], color=magnetization_color, linestyle="dashed")
	plt.axvline(x=data["magnetization"]["mean"]-data["magnetization"]["std"], color=magnetization_color, linestyle="dotted")
	plt.axvline(x=data["magnetization"]["mean"]+data["magnetization"]["std"], color=magnetization_color, linestyle="dotted")
	plt.legend(loc="upper right", bbox_to_anchor=(1, 1), fancybox=True, fontsize=10)
	plt.xlim(x_min, x_max)
	plt.xlabel(r"$T$", fontsize=12)
	plt.ylabel(r"Samples per temperature ($N$)", fontsize=12)
	plt.yticks(y, data["Ns"])
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "tc{}.png".format(suffix)))
	plt.close()


def plot_times(results_dir, encoder_names, encoder_labels, encoder_colors, preprocessing_bottoms, preprocessing_label="", preprocessing_color="black"):
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	times = np.load(os.path.join(results_dir, "processed", "times.npz"))
	x = np.arange(len(times["Ls"]))
	total_width = 0.7
	width = total_width/len(encoder_names)
	shifts = [-total_width/2 + total_width/(2*len(encoder_names))*(2*m+1) for m in range(len(encoder_names))]
	plt.figure()
	for (name, label, color, shift) in zip(encoder_names, encoder_labels, encoder_colors, shifts):
		plt.bar(x+shift, times[name+"_means"], width, yerr=times[name+"_stds"], color=color, label=color)
		if name in preprocessing_bottoms:
			plt.bar(x+shift, times["preprocessing_means"], width, yerr=times["preprocessing_stds"], bottom=times[name+"_means"], color=preprocessing_color, label=preprocessing_label)
	plt.legend(loc="upper left", bbox_to_anchor=(0, 1), fancybox=True, fontsize=10)
	plt.xlabel(r"Lattice size ($L$)", fontsize=12)
	plt.xticks(x, times["Ls"].astype(np.int32))
	plt.ylabel("Time (min)", fontsize=12)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "times.png"))
	plt.close()


***

def tabulate_generators(results_dir, Js, encoder_names, encoder_headings):
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(results_dir, "processed", "generators.json"), "r") as fp:
		gens = json.load(fp)
	generator_types = ["spatial", "internal"]
	stds = [gens[J][generator_type][name]["std"] for J in Js for generator_type in generator_types for name in encoder_names]
	max_precision = 1 + max([1-int(np.log10(s)) for s in stds	])
	S_columns = "S[table-format=-1.{:d}(2),table-align-uncertainty=true]".format(max_precision)*(2*len(encoder_names))
	with open(os.path.join(output_dir, "generators.tex"), "w") as fp:
		fp.write("\\begin{{tabular}}{{c{}}}\n".format(S_columns))
		fp.write("\\toprule\n")
		fp.write("\\quad")
		for heading in generator_types:
			fp.write(" & \\multicolumn{{{:d}}}{{c}}{{{}}}".format(len(encoder_names), heading.capitalize()))
		fp.write(" \\\\\n")
		fp.write("\\midrule\n")
		fp.write("\\quad")
		for _ in range(2):
			for heading in encoder_headings:
				fp.write(" & {}".format(heading))
		fp.write(" \\\\\n")
		fp.write("\\midrule\n")
		for J in Js:
			fp.write(J.capitalize())
			for generator_type in generator_types:
				for name in encoder_names:
					precision = 2-int(np.log10(gens[J][generator_type][name]["std"]))
					fp.write(" & {{:.{:d}f}}\\pm {{:.{:d}f}}".format(precision, precision).format(gens[J][generator_type][name]["mean"], gens[J][generator_type][name]["std"]))
			fp.write(" \\\\\n")
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


if __name__ == "__main__":
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]

	tc_color = "red"
	magnetization_color= "orange"

	encoder_names = ["latent", "latent_equivariant", "latent_multiscale"]
	encoder_labels = ["Baseline-encoder", "GE-encoder", "GE-encoder (multiscale)"]
	encoder_colors = ["purple", "blue", "green"]

	preprocessing_bottoms = ["latent_equivariant", "latent_multiscale"]
	preprocessing_label = "Checkerboard-averaging"
	preprocessing_color = "lightblue"

	print("Plotting statistics . . . ")
	for J in Js:
		plot_stats("results", J, "magnetization", 128, tc_color=tc_color)
		for name in encoder_names:
			plot_stats("results", J, name, 128, N=2048, tc_color=tc_color)

	print("Plotting critical temperature estimates . . . ")
	for J in Js:
		for L in Ls:
			plot_critical_temperatures("results", J, L, encoder_names, encoder_labels, encoder_colors, magnetization_color=magnetization_color, tc_color=tc_color, remove_bias=False)

	print("Plotting times . . . ")
	plot_times("results", encoder_names, encoder_labels, encoder_colors, preprocessing_bottoms, preprocessing_label=preprocessing_label, preprocessing_color=preprocessing_color)

	print("Tabulating generators . . . ")
	GE_encoder_names = ["latent_equivariant", "latent_multiscale"]
	GE_encoder_headings = ["GE-encoder", "GE-encoder (multiscale)"]
	tabulate_generators("results", Js, GE_encoder_names, GE_encoder_headings)

	print("Done!")
