import os
import json
import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

def plot_stats(results_dir, J, observable_name, L, N=None):
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
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="green", linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "distribution.png"))
	plt.close()
	# order parameter
	plt.figure()
	plt.plot(stats["temperatures"], stats["order_means"], color="black")
	plt.plot(stats["temperatures"], stats["order_means"]-stats["order_stds"], color="black", linestyle="dashed")
	plt.plot(stats["temperatures"], stats["order_means"]+stats["order_stds"], color="black", linestyle="dashed")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="green", linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "order.png"))
	plt.close()
	# U_4 Binder cumulant
	plt.figure()
	plt.plot(stats["temperatures"], stats["u4_means"], color="black")
	plt.plot(stats["temperatures"], stats["u4_means"]-stats["u4_stds"], color="black", linestyle="dashed")
	plt.plot(stats["temperatures"], stats["u4_means"]+stats["u4_stds"], color="black", linestyle="dashed")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="green", linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "u4.png"))
	plt.close()


def plot_critical_temperatures(results_dir, J, L, remove_bias=True):
	output_dir = os.path.join(results_dir, "plots", J, "L{:d}".format(L))
	os.makedirs(output_dir, exist_ok=True)
	suffix = "_biased" if not remove_bias else ""
	with open(os.path.join(results_dir, "processed", J, "tc{}.json".format(suffix)), "r") as fp:
		data = json.load(fp)["L{:d}".format(L)]
	x_min = np.round(np.array(data["AE"]["means"]+data["GE"]["means"]).min()-0.1, 1)
	x_max = np.round(np.concatenate([np.array(data[obs]["means"])+np.array(data[obs]["stds"]) for obs in ["AE", "GE"]], 0).max()+0.1, 1)
	y = np.arange(len(data["Ns"]))
	width = 0.35
	plt.figure()
	plt.barh(y-width/2, data["AE"]["means"], width, xerr=data["AE"]["stds"], color="red", label="Baseline encoder")
	plt.barh(y+width/2, data["GE"]["means"], width, xerr=data["GE"]["stds"], color="blue", label="GE-encoder")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="green", linestyle="dashed")
	plt.axvline(x=data["M"]["mean"], color="orange", linestyle="dashed")
	plt.axvline(x=data["M"]["mean"]-data["M"]["std"], color="orange", linestyle="dotted")
	plt.axvline(x=data["M"]["mean"]+data["M"]["std"], color="orange", linestyle="dotted")
	plt.legend(loc="upper right", bbox_to_anchor=(1, 1), fancybox=True, fontsize=10)
	plt.xlim(x_min, x_max)
	plt.xlabel(r"$T$", fontsize=12)
	plt.ylabel(r"Samples per temperature ($N$)", fontsize=12)
	plt.yticks(y, data["Ns"])
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "tc{}.png".format(suffix)))
	plt.close()


def plot_times(results_dir):
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	times = np.load(os.path.join(results_dir, "processed", "times.npz"))
	x = np.arange(len(times["Ls"]))
	width = 0.35
	plt.figure()
	plt.bar(x-width/2, times["AE_means"], width, yerr=times["AE_stds"], color="red", label="Baseline encoder")
	plt.bar(x+width/2, times["GE_means"], width, yerr=times["GE_stds"], color="blue", label="GE-encoder")
	plt.bar(x+width/2, times["preprocessing_means"], width, yerr=times["preprocessing_stds"], bottom=times["GE_means"], color="lightblue", label="Checkerboard averaging")
	plt.legend(loc="upper left", bbox_to_anchor=(0, 1), fancybox=True, fontsize=10)
	plt.xlabel("Lattice size (L)", fontsize=12)
	plt.xticks(x, times["Ls"].astype(np.int32))
	plt.ylabel("Time (min)", fontsize=12)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "times.png"))
	plt.close()


def tabulate_generators(results_dir, Js):
	output_dir = os.path.join(results_dir, "plots")
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(results_dir, "processed", "generators.json"), "r") as fp:
		gens = json.load(fp)
	generator_types = ["spatial", "internal"]
	stds = [gens[J][generator_type]["std"] for J in Js for generator_type in generator_types]
	max_precision = 1 + max([1-int(np.log10(s)) for s in stds	])
	with open(os.path.join(output_dir, "generators.tex"), "w") as fp:
		fp.write("\\begin{{tabular}}{{cS[table-format=2.{:d}(2),table-align-uncertainty=true]S[table-format=2.{:d}(2),table-align-uncertainty=true]}}\n".format(max_precision, max_precision))
		fp.write("\\toprule\n")
		fp.write("\\quad & {Spatial} & {Internal} \\\\\n")
		fp.write("\\midrule\n")
		for J in Js:
			fp.write(J.capitalize())
			for generator_type in generator_types:
				precision = 2-int(np.log10(gens[J][generator_type]["std"]))
				fp.write(" & {{:.{:d}f}}\\pm {{:.{:d}f}}".format(precision, precision).format(gens[J][generator_type]["mean"], gens[J][generator_type]["std"]))
			fp.write(" \\\\\n")
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


if __name__ == "__main__":
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]

	print("Plotting statistics . . . ")
	for J in Js:
		plot_stats("results", J, "magnetization", 128)
		plot_stats("results", J, "latent", 128, N=2048)
		plot_stats("results", J, "latent_equivariant", 128, N=2048)

	print("Plotting critical temperature estimates . . . ")
	for J in Js:
		for L in Ls:
			plot_critical_temperatures("results", J, L, remove_bias=False)

	print("Plotting times . . . ")
	plot_times("results")

	print("Tabulating generators . . . ")
	tabulate_generators("results", Js)

	print("Done!")
