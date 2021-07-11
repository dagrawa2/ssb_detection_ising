import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=14)
matplotlib.rc("ytick", labelsize=14)


def order_plot():
	results_dir = "../results/GMM"
	temperatures = []
	distances = []
	for temperature_dir in os.listdir(results_dir):
		temperatures.append( float(temperature_dir.strip("T")) )
		params = np.load(os.path.join(results_dir, temperature_dir, "params.npz"))
		diff = params["mean"]-np.flip(params["mean"])
		distances.append( np.sum(diff**2)/np.sqrt(diff.dot(params["variance"].dot(diff))) )
	plt.figure()
	plt.scatter(temperatures, distances, color="black")
	plt.axvline(x=2/np.log(1+np.sqrt(2)), color="blue", linestyle="dashed")
	plt.xlabel("T", fontsize=12)
	plt.ylabel("Distance", fontsize=12)
	plt.yscale("log")
	plt.tight_layout()
	os.makedirs("figs", exist_ok=True)
	plt.savefig(os.path.join("figs", "order.png"))
	plt.close()


def cluster_plot(temperature_dir, lim=None):
	points = np.load(os.path.join("../results/AE/encodings", temperature_dir+".npy"))
	params = np.load(os.path.join("../results/GMM", temperature_dir, "params.npz"))
	lams, eigvecs = np.linalg.eigh(params["variance"])
	width = np.sqrt(lams[0])
	height = np.sqrt(lams[1])
	angle_1 = 180/np.pi * np.sign(eigvecs[1,0])*np.arccos(eigvecs[0,0])
	angle_2 = 180/np.pi * np.sign(eigvecs[0,0])*np.arccos(eigvecs[1,0])
	ellipse_1 = matplotlib.patches.Ellipse(params["mean"], width, height, angle=angle_1, fill=False, color="red")
	ellipse_2 = matplotlib.patches.Ellipse(np.flip(params["mean"]), width, height, angle=angle_2, fill=False, color="red")
	plt.figure()
	plt.scatter(points[:,0], points[:,1], s=5, marker="o", color="black")
	plt.scatter(params["mean"], np.flip(params["mean"]), marker="*", color="red")
	plt.gca().add_patch(ellipse_1)
	plt.gca().add_patch(ellipse_2)
	if lim is not None:
		plt.xlim(-lim, lim)
		plt.ylim(-lim, lim)
		plt.gca().set_aspect('equal', adjustable="box")
	plt.title("T = "+temperature_dir.strip("T"), fontsize=16)
	plt.tight_layout()
	os.makedirs("figs/cluster", exist_ok=True)
	plt.savefig(os.path.join("figs/cluster", temperature_dir+".png"))
	plt.close()


def axis_limit():
	lims = []
	encodings_dir = "../results/AE/encodings"
	for temperature_file in os.listdir(encodings_dir):
		points = np.load(os.path.join(encodings_dir, temperature_file))
		lims.append( np.abs(points).max() )
	lims = np.array(lims)
	return lims.max()


if __name__ == "__main__":
	print("Plotting order parameter . . . ")
	order_plot()

	print("Plotting clusters . . . ")
	lim = axis_limit()
	for temperature_dir in os.listdir("../results/GMM"):
		cluster_plot(temperature_dir, lim=lim)

	print("Done!")
