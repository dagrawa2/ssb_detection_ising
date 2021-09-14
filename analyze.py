import os
import copy
import json
import numpy as np
from scipy.integrate import quad as quad_integrate
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
from functools import reduce
from operator import mul

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

from phasefinder import groups
from phasefinder.datasets import Ising


def gather_Ms(data_dir, J, L):
	temperatures = []
	measurements = []
	L_dir = os.path.join(data_dir, J, "L{:d}".format(L))
	for temperature_dir in sorted(os.listdir(L_dir)):
		I = Ising()
		Ms = I.magnetization(os.path.join(L_dir, temperature_dir), per_spin=True, staggered=(J=="antiferromagnetic"))
		temperatures.append(I.T)
		measurements.append(Ms)
	temperatures = np.array(temperatures)
	measurements = np.stack(measurements, 0)
	output_dir = "results/{}/magnetization/L{:d}".format(J, L)
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "measurements.npz"), temperatures=temperatures, measurements=measurements)

def calculate_stats(results_dir, J, observable_name, L, bins=50):
	data = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "measurements.npz"))
	temperatures = data["temperatures"]
	measurements = data["measurements"]
	distributions = []
	distribution_range = (measurements.min(), measurements.max())
	order_means, order_stds = [], []
	binder_means, binder_stds = [], []
	for i in range(measurements.shape[0]):
		hist, _ = np.histogram(measurements[i], bins=bins, range=distribution_range, density=False)
		hist = hist/measurements.shape[1]
		distributions.append(hist)
		order_mean, order_std = Ising().jackknife(measurements[i], lambda x: np.mean(np.abs(x), 1))
		order_means.append(order_mean)
		order_stds.append(order_std)
		binder_mean, binder_std = Ising().jackknife(measurements[i], lambda x: 1 - np.mean(x**4, 1)/(3*np.mean(x**2, 1)**2))
		binder_means.append(binder_mean)
		binder_stds.append(binder_std)
	distributions = np.stack(distributions, 0)
	order_means = np.array(order_means)
	order_stds = np.array(order_stds)
	binder_means = np.array(binder_means)
	binder_stds = np.array(binder_stds)
	distribution_range = np.array(list(distribution_range))
	np.savez(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "stats.npz"), distributions=distributions, distribution_range=distribution_range, order_means=order_means, order_stds=order_stds, binder_means=binder_means, binder_stds=binder_stds)

def plot(results_dir, J, observable_name, L):
	temperatures = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "measurements.npz"))["temperatures"]
	stats = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "stats.npz"))
	distributions = stats["distributions"]
	distribution_range = stats["distribution_range"].tolist() if "distribution_range" in stats.keys() else [-1, 1]
	order_means, order_stds = stats["order_means"], stats["order_stds"]
	binder_means, binder_stds = stats["binder_means"], stats["binder_stds"]
	# distributions
	plt.figure()
	plt.imshow(np.flip(distributions.T, 0), cmap="gray_r", vmin=0, vmax=1, extent=(temperatures.min(), temperatures.max(), *distribution_range), aspect="auto")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="blue", linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.ylabel(r"$M$", fontsize=12)
	plt.title(r"$L="+str(int(L))+r"$", fontsize=16)
	plt.tight_layout()
	plots_dir = os.path.join(results_dir, "plots", J, observable_name, "L{:d}".format(L))
	os.makedirs(plots_dir, exist_ok=True)
	plt.savefig(os.path.join(plots_dir, "distribution.png"))
	plt.close()
	# order parameter
	plt.figure()
	plt.plot(temperatures, order_means, color="black")
	plt.plot(temperatures, order_means-order_stds, color="black", linestyle="dashed")
	plt.plot(temperatures, order_means+order_stds, color="black", linestyle="dashed")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="blue", linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.ylabel(r"$\langle |M|\rangle$", fontsize=12)
	plt.title(r"$L="+str(int(L))+r"$", fontsize=16)
	plt.tight_layout()
	plt.savefig(os.path.join(plots_dir, "order.png"))
	plt.close()
	# binder cumulant
	plt.figure()
	plt.plot(temperatures, binder_means, color="black")
	plt.plot(temperatures, binder_means-binder_stds, color="black", linestyle="dashed")
	plt.plot(temperatures, binder_means+binder_stds, color="black", linestyle="dashed")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="blue", linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.ylabel(r"$U_4$", fontsize=12)
	plt.title(r"$L="+str(int(L))+r"$", fontsize=16)
	plt.tight_layout()
	plt.savefig(os.path.join(plots_dir, "binder.png"))
	plt.close()


def calculate_symmetries(results_dir, J, observable_name, L, z_critical=1):
	data = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "symmetry_scores.npz"))
	temperatures = data["temperatures"]
	group_elements = [groups.Element(*g) for g in data["group_elements"].tolist()]
	symmetry_scores = data["symmetry_scores"]
	unbroken_groups = []
	for i in range(len(temperatures)):
		elements = list( map(lambda x: x[1], sorted(zip(symmetry_scores[i], group_elements), key=lambda tup: tup[0])) )
		scores = {g: score for (g, score) in zip(group_elements, symmetry_scores[i])}
		subgroup = set([])
		subgroups = []
		cluster_separations = []
		while len(elements) > 0:
			subgroup.add( elements.pop(0) )
			subgroup = groups.generate_normal_subgroup(subgroup, set(group_elements))
			elements = [g for g in elements if g not in subgroup]
			subgroups.append(copy.deepcopy(subgroup))
			if len(elements) > 0:
				cluster_separations.append( scores[elements[0]] - max([scores[g] for g in subgroup]) )
		cluster_separations = np.array(cluster_separations)
		i_max = np.argmax(cluster_separations)
		score_diffs = np.diff(symmetry_scores[i])
		if cluster_separations[i_max] > np.mean(score_diffs) + z_critical*np.std(score_diffs):
			subgroup = subgroups[i_max]
		else:
			subgroup = subgroups[-1]
		unbroken_groups.append( frozenset(subgroup) )
	grouped_temperatures = {G: [] for G in set(unbroken_groups)}
	for (G, T) in zip(unbroken_groups, temperatures):
		grouped_temperatures[G].append( float(T) )
	output = [{"group": [g.value for g in G], "temperatures": sorted(Ts)} for (G, Ts) in grouped_temperatures.items()]
	with open(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "unbroken_symmetries.json"), "w") as fp:
		json.dump(output, fp, indent=2)

def plot_symmetry_scores(results_dir, J, observable_name, L):
	data = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "symmetry_scores.npz"))
	temperatures = data["temperatures"]
	symmetry_scores = data["symmetry_scores"]
	temperatures = np.repeat(temperatures, symmetry_scores.shape[1])
	symmetry_scores = symmetry_scores.reshape((-1))
	plt.figure()
	plt.scatter(temperatures, symmetry_scores, color="black", alpha=0.2)
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="blue", linestyle="dashed")
	plt.xlabel(r"$T$", fontsize=12)
	plt.ylabel(r"MMD", fontsize=12)
	plt.title(r"$L="+str(int(L))+r"$", fontsize=16)
	plt.tight_layout()
	plots_dir = os.path.join(results_dir, "plots", J, observable_name, "L{:d}".format(L))
	os.makedirs(plots_dir, exist_ok=True)
	plt.savefig(os.path.join(plots_dir, "mmds.png"))
	plt.close()


def curve_zeros(x, y):
	zeros = []
	x_1, y_1 = x[0], y[0]
	for (x_2, y_2) in zip(x[1:], y[1:]):
		if np.sign(y_1) != np.sign(y_2):
			zeros.append( x_1 - y_1*(x_2-x_1)/(y_2-y_1) )
		x_1, y_1 = x_2, y_2
	return np.array(zeros)

def calculate_binder_ratios(results_dir, J, observable_name, L_1, L_2):
	data_1 = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L_1), "measurements.npz"))
	data_2 = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L_2), "measurements.npz"))
	temperatures = data_1["temperatures"]
	measurements_1 = data_1["measurements"]
	measurements_2 = data_2["measurements"]
	ratio_means, ratio_stds = [], []
	for i in range(measurements_1.shape[0]):
		ratio_mean, ratio_std = Ising().jackknife(measurements_1[i], lambda x: 1 - np.mean(x**4, 1)/(3*np.mean(x**2, 1)**2), samples_2=measurements_2[i])
		ratio_means.append(ratio_mean)
		ratio_stds.append(ratio_std)
	ratio_means = np.array(ratio_means) - 1
	ratio_stds = np.array(ratio_stds)
	zero_means = curve_zeros(temperatures, ratio_means)
	zero_stds_lower = curve_zeros(temperatures, ratio_means-ratio_stds)
	zero_stds_upper = curve_zeros(temperatures, ratio_means+ratio_stds)
	np.savez(os.path.join(results_dir, J, observable_name, "binder_ratios_L{:d}_{:d}.npz".format(L_1, L_2)), temperatures=temperatures, ratio_means=ratio_means, ratio_stds=ratio_stds, zero_means=zero_means, zero_stds_lower=zero_stds_lower, zero_stds_upper=zero_stds_upper)


def generator_table(results_dir, J, Ls):
	output_dir = os.path.join(results_dir, "tables", J)
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "generator_reps.tex"), "w") as fp:
		fp.write("\\begin{tabular}{ccc}\n")
		fp.write("\\toprule\n")
		fp.write("$L$ & Spatial & Internal \\\\\n")
		fp.write("\\midrule\n")
		for L in Ls:
			with open(os.path.join(results_dir, J, "latent_equivariant", "L{:d}".format(L), "generator_reps.json"), "r") as json_file:
				gens = json.load(json_file)
			fp.write("{:d} & {:.3f} & {:.3f} \\\\\n".format(L, gens["spatial"], gens["internal"]))
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


def critical_temperature_table(results_dir, J, observable_names, Ls, row_headings_dict):
	output_dir = os.path.join(results_dir, "tables", J)
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "critical_temperatures.tex"), "w") as fp:
		fp.write("\\begin{tabular}{cc}\n")
		fp.write("\\toprule\n")
		fp.write("Method & Critical Temp. \\\\\n")
		fp.write("\\midrule\n")
		for observable_name in observable_names:
			temperatures = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(Ls[0]), "measurements.npz"))["temperatures"]
			posterior = np.ones_like(temperatures)
			for (L_1, L_2) in zip(Ls[:-1], Ls[1:]):
				bd_means = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L_1), "stats.npz"))["binder_means"] \
					- np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L_2), "stats.npz"))["binder_means"]
				bd_stds = np.sqrt( np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L_1), "stats.npz"))["binder_stds"]**2 \
					+ np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L_2), "stats.npz"))["binder_stds"]**2 )
				posterior = posterior * 1/(np.sqrt(2*np.pi)*bd_stds) * np.exp(-0.5*(bd_means/bd_stds)**2)
			posterior = posterior/posterior.sum()
			Tc_mean = (temperatures*posterior).sum()
			Tc_std = np.sqrt( (temperatures**2*posterior).sum() - Tc_mean**2 )
			fp.write("{} & ${:.3f}\\pm {:.3f}$ \\\\\n".format(row_headings_dict[observable_name], Tc_mean, Tc_std))
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


if __name__ == "__main__":
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	observable_names = ["magnetization", "latent", "latent_equivariant"]
	row_headings_dict = {"magnetization": "Mag", "latent": "AE", "latent_equivariant": "GE-AE"}


	"""
	print("Gathering magnetizations . . . ")
	for J in Js:
		print("\t{} case".format(J))
		for L in Ls:
			print("\t\tL={:d}".format(L))
			gather_Ms("data", J, L)
	"""

	print("===")
	for J in Js:
		"""
		print("{} case".format(J))
		for observable_name in observable_names:
			print("\t{}:".format(observable_name))
			for L in Ls:
				print("\t\tL={:d} . . . ".format(L))
				calculate_stats("results", J, observable_name, L)
				plot("results", J, observable_name, L)
#			print("\tBinder ratios")
#			calculate_binder_ratios("results", J, observable_name, 128, 64)
			if observable_name == "latent_equivariant":
				print("\tgenerator table")
				generator_table("results", J, Ls)
		"""

		print("\tcritical temperature table")
#		observable_names = ["latent_equivariant"]
		critical_temperature_table("results", J, observable_names, Ls, row_headings_dict)


	print("Done!")
