import os
import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

from phasefinder.datasets import Ising


def plot(results_dir):
	bins = 50
	temperatures = []
	distributions = []
	order_means, order_stds = [], []
	binder_means, binder_stds = [], []
	for temperature_dir in sorted(os.listdir(results_dir)):
		I = Ising()
		Ms = I.load_M(os.path.join(results_dir, temperature_dir), per_spin=True)
		hist, bin_edges = np.histogram(Ms, bins=bins, range=(-1, 1), density=False)
		hist = hist/len(Ms)
		distributions.append(hist)
		order_mean, order_std = I.jackknife(Ms, lambda x: np.mean(np.abs(x)))
		order_means.append(order_mean)
		order_stds.append(order_std)
		binder_mean, binder_std = I.jackknife(Ms, lambda x: 1 - np.mean(x**4)/(3*np.mean(x**2)**2))
		binder_means.append(binder_mean)
		binder_stds.append(binder_std)
		temperatures.append(I.T)
	L = I.L
	temperatures = np.array(temperatures)
	distributions = np.stack(distributions, axis=1)
	order_means = np.array(order_means)
	order_stds = np.array(order_stds)
	binder_means = np.array(binder_means)
	binder_stds = np.array(binder_stds)
	# distributions
	plt.figure()
	plt.imshow(distributions, cmap="gray", vmin=0, vmax=1)
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="blue", linestyle="dashed")
	plt.yticks(np.flip(np.arange(bins+1)-0.5), bin_edges)
	plt.xlabel(r"$T$", fontsize=12)
	plt.ylabel(r"$M$", fontsize=12)
	plt.title(r"$L="+str(int(L))+r"$", fontsize=16)
	plt.tight_layout()
	plots_dir = "plots/L{:d}".format(L)
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
	plt.ylim(0, 1)
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

	for L in Ls:
		print("Plotting L={:d} . . . ".format(L))
		plot("data/L{:d}".format(L))

	print("Done!")
