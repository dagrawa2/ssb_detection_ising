import os
import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)

from phasefinder.datasets import Ising


def plot(results_dir):
	temperatures = []
	order_means, order_stds = [], []
	binder_means, binder_stds = [], []
	for temperature_dir in sorted(os.listdir(results_dir)):
		I = Ising()
		Ms = I.load_M(os.path.join(results_dir, temperature_dir))
		order_mean, order_std = I.jackknife(Ms, lambda x: np.mean(np.abs(x)))
		order_means.append(order_mean)
		order_stds.append(order_std)
		binder_mean, binder_std = I.jackknife(Ms, lambda x: 1 - np.mean(x**4)/(3*np.mean(x**2)**2))
		binder_means.append(binder_mean)
		binder_stds.append(binder_std)
		temperatures.append(I.T)
	L = I.L
	temperatures = np.array(temperatures)
	order_means = np.array(order_means)
	order_stds = np.array(order_stds)
	binder_means = np.array(binder_means)
	binder_stds = np.array(binder_stds)
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
	plots_dir = "plots/L{:d}".format(L)
	os.makedirs(plots_dir, exist_ok=True)
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
	os.makedirs(plots_dir, exist_ok=True)
	plt.savefig(os.path.join(plots_dir, "binder.png"))
	plt.close()


if __name__ == "__main__":
	Ls = [16, 32, 64, 128]

	for L in Ls:
		print("Plotting L={:d} . . . ".format(L))
		plot("data/L{:d}".format(L))

	print("Done!")
