import os
import code
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import lstsq
from scipy.optimize import bisect


def jackknife_samples(data):
	M = data.shape[0]
	sums = data.sum(0)
	estimates = sums/M
	samples = (sums[None,:] - data)/(M-1)
	samples = np.concatenate((estimates[None,:], samples), 0)
	return samples

def jackknife_mean_std(samples):
	D = samples.ndim
	if D == 1:
		samples = samples[:,None]
	estimates, samples = samples[0], samples[1:]
	std = np.sqrt(np.sum((samples - estimates[None,:])**2, 0))
	bias = samples.shape[0]*(samples.mean(0) - estimates)
	mean = estimates - bias
	if D == 1:
		mean, std = mean.item(), std.item()
	return mean, std

def U2_samples(results_dir, J, observable_name, L):
	measurements = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "measurements.npz"))["measurements"].T
	samples_square = jackknife_samples(measurements**2)
	samples_abs = jackknife_samples(np.abs(measurements))
	samples = samples_square/samples_abs**2
	return samples

def crossing_samples(samples_1, samples_2, temperatures):
	crossings = []
	for i in range(samples_1.shape[0]):
		f1 = interp1d(temperatures, samples_1[i], kind="cubic")
		f2 = interp1d(temperatures, samples_2[i], kind="cubic")
		f = lambda x: f1(x)-f2(x)
		code.interact(local=locals())
		sign_0 = np.sign(f(temperatures[0]))
		for k in range(1, len(temperatures)+1):
			if np.sign(f(temperatures[-k])) != sign_0:
				break
		if k < len(temperatures):
			crossings.append( bisect(f, temperatures[0], temperatures[-k]) )
		else:
			crossings.append(np.nan)
	return crossings

def yintercept_samples(x, y_samples, weights=None):
	x = np.stack([np.ones_like(x), x], 1)
	if weights is not None:
		x = x*weights[:,None]
		y_samples = y_samples*weights[None,:]
	sol = lstsq(x, y_samples.T)[0]
	return sol[0]

def critical_temperature_table(results_dir, J, observable_names, Ls, row_headings_dict, temperature_range=None):
#	output_dir = os.path.join(results_dir, "tables", J)
#	os.makedirs(output_dir, exist_ok=True)
#	with open(os.path.join(output_dir, "critical_temperatures.tex"), "w") as fp:
	with open("critical_temperatures.tex", "w") as fp:
		fp.write("\\begin{tabular}{cc}\n")
		fp.write("\\toprule\n")
		fp.write("Method & Critical Temp. \\\\\n")
		fp.write("\\midrule\n")
		Ls_recip = np.array([1/L for L in Ls])
		for observable_name in observable_names:
			print("Observable name:", observable_name)
			temperatures = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(Ls[0]), "measurements.npz"))["temperatures"]
			U_2 = np.stack([U2_samples(results_dir, J, observable_name, L) for L in Ls], 0)
			if temperature_range is not None:
				T_within_range_indices = np.where(np.logical_and(temperatures>=temperature_range[0], temperatures<=temperature_range[1]))[0]
				T_min_index, T_max_index = T_within_range_indices.min(), T_within_range_indices.max()
				temperatures = temperatures[T_min_index:T_max_index+1]
				U_2 = U_2[:,:,T_min_index:T_max_index+1]
			crossings = np.stack([crossing_samples(U_2[i], U_2[i+1], temperatures) for i in range(len(Ls)-1)], 1)
			assert ~np.isnan(crossings[0]).any(), "Curves without jackknifing do not cross; crossings[0] = {}.".format(crossings[0])
			crossings = crossings[~np.isnan(crossings).any(1)]
			crossings_mean, crossings_std = jackknife_mean_std(crossings)
			weights = 1/crossings_std
			Tc_samples = yintercept_samples(Ls_recip[1:], crossings, weights=weights)
			Tc_mean, Tc_std = jackknife_mean_std(Tc_samples)
			fp.write("{} & ${:.3f}\\pm {:.3f}$ \\\\\n".format(row_headings_dict[observable_name], Tc_mean, Tc_std))
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


if __name__ == "__main__":
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	observable_names = ["magnetization", "latent", "latent_equivariant"]
	row_headings_dict = {"magnetization": "Mag", "latent": "AE", "latent_equivariant": "GE-AE"}

	J = "ferromagnetic"
	observable_names = ["latent"]  # ["magnetization", "latent_equivariant"]
	critical_temperature_table("results", J, observable_names, Ls, row_headings_dict)  # , temperature_range=(1.0, 3.5))


	print("Done!")
