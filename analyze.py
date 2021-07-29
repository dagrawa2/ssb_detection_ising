import os
import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

from phasefinder.datasets import Ising


def gather_Ms(data_dir, L):
	temperatures = []
	measurements = []
	L_dir = os.path.join(data_dir, "L{:d}".format(L))
	for temperature_dir in sorted(os.listdir(L_dir)):
		I = Ising()
		Ms = I.load_M(os.path.join(L_dir, temperature_dir), per_spin=True)
		temperatures.append(I.T)
		measurements.append(Ms)
	temperatures = np.array(temperatures)
	measurements = np.stack(measurements, 0)
	output_dir = "results/magnetization/L{:d}".format(L)
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "measurements.npz"), temperatures=temperatures, measurements=measurements)

def calculate_stats(results_dir, observable_name, L, bins=50):
	data = np.load(os.path.join(results_dir, observable_name, "L{:d}".format(L), "measurements.npz"))
	temperatures = data["temperatures"]
	measurements = data["measurements"]
	distributions = []
	distribution_range = ( round(measurements.min()), round(measurements.max()) )
	order_means, order_stds = [], []
	binder_means, binder_stds = [], []
	for i in range(measurements.shape[0]):
		hist, _ = np.histogram(measurements[i], bins=bins, range=distribution_range, density=False)
		hist = hist/measurements.shape[1]
		distributions.append(hist)
		order_mean, order_std = Ising().jackknife(measurements[i], lambda x: np.mean(np.abs(x)))
		order_means.append(order_mean)
		order_stds.append(order_std)
		binder_mean, binder_std = Ising().jackknife(measurements[i], lambda x: 1 - np.mean(x**4)/(3*np.mean(x**2)**2))
		binder_means.append(binder_mean)
		binder_stds.append(binder_std)
	distributions = np.stack(distributions, 0)
	order_means = np.array(order_means)
	order_stds = np.array(order_stds)
	binder_means = np.array(binder_means)
	binder_stds = np.array(binder_stds)
	distribution_range = np.array(list(distribution_range))
	np.savez(os.path.join(results_dir, observable_name, "L{:d}".format(L), "stats.npz"), distributions=distributions, distribution_range=distribution_range, order_means=order_means, order_stds=order_stds, binder_means=binder_means, binder_stds=binder_stds)

def plot(results_dir, observable_name, L):
	temperatures = np.load(os.path.join(results_dir, observable_name, "L{:d}".format(L), "measurements.npz"))["temperatures"]
	stats = np.load(os.path.join(results_dir, observable_name, "L{:d}".format(L), "stats.npz"))
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
	plots_dir = os.path.join(results_dir, "plots", observable_name, "L{:d}".format(L))
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
	plt.ylim(0, max(distribution_range))
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
	plt.ylim(-0.1, 0.7)
	plt.title(r"$L="+str(int(L))+r"$", fontsize=16)
	plt.tight_layout()
	os.makedirs(plots_dir, exist_ok=True)
	plt.savefig(os.path.join(plots_dir, "binder.png"))
	plt.close()


if __name__ == "__main__":
	Ls = [16, 32, 64, 128]

#	for L in Ls:
#		print("Gathering magnetizations for L={:d} . . . ".format(L))
#		gather_Ms("data", L)

	observable_name = "latent"
	for L in Ls:
		print("Calculating stats for L={:d} . . . ".format(L))
		calculate_stats("results", observable_name, L)
		print("Plotting L={:d} . . . ".format(L))
		plot("results", observable_name, L)

	print("Done!")
