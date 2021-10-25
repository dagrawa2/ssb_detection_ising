import os
import json
import numpy as np
from scipy.interpolate import CubicSpline

from phasefinder import jackknife
from phasefinder.datasets import Ising


###process magnetization, order parameter curves, and U_4 Binder cumulant curves

def gather_magnetizations(data_dir, results_dir, J, L):
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
	output_dir = "{}/{}/magnetization/L{:d}".format(results_dir, J, L)
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "measurements.npz"), temperatures=temperatures, measurements=measurements)


def calculate_stats(results_dir, J, observable_name, L, N=None, bins=50):
	if N is not None:
		measurements = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "N{:d}".format(N), "measurements.npz"))
	else:
		measurements = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "measurements.npz"))
	temperatures = measurements["temperatures"]
	measurements = measurements["measurements"].T
	order_means, order_stds = jackknife.calculate_mean_std( jackknife.calculate_samples(np.abs(measurements)) )
	u4_means, u4_stds = jackknife.calculate_mean_std( \
		1 - jackknife.calculate_samples(measurements**4)/( 3*jackknife.calculate_samples(measurements**2)**2 ) )
	distribution_range = (measurements.min(), measurements.max())
	distributions = []
	for slice in measurements.T:
		hist, _ = np.histogram(slice, bins=bins, range=distribution_range, density=False)
		hist = hist/len(slice)
		distributions.append(hist)
	distributions = np.stack(distributions, 0)
	distribution_range = np.array(list(distribution_range))
	if N is not None:
		output_dir = os.path.join(results_dir, "processed", J, observable_name, "L{:d}".format(L), "N{:d}".format(N))
	else:
		output_dir = os.path.join(results_dir, "processed", J, observable_name, "L{:d}".format(L))
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "stats.npz"), temperatures=temperatures, distributions=distributions, distribution_range=distribution_range, order_means=order_means, order_stds=order_stds, u4_means=u4_means, u4_stds=u4_stds)


### process critical temperature estimates

def U4_samples(results_dir, J, observable_name, L, N=None):
	if N is not None:
		measurements = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "N{:d}".format(N), "measurements.npz"))["measurements"].T
	else:
		measurements = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "measurements.npz"))["measurements"].T
	samples_2 = jackknife.calculate_samples(measurements**2)
	samples_4 = jackknife.calculate_samples(measurements**4)
	samples = 1 - samples_4/(3*samples_2**2)
	return samples


def critical_temperature_samples(temperatures, u4samples):
	tc_samples = []
	for samples in u4samples:
		cs = CubicSpline(temperatures, samples)
		roots = cs.derivative(2).roots(extrapolate=False)
		values = np.abs(cs(roots, nu=1))
		tc_samples.append( roots[np.argmax(values)] )
	tc_samples = np.array(tc_samples)
	return tc_samples


def calculate_critical_temperatures(results_dir, J, Ls, Ns, encoder_names, remove_bias=True):
	output_dict = {"L{:d}".format(L): {"Ns": Ns, "magnetization": {}} for L in Ls}
	for L in Ls:
		for name in encoder_names:
			output_dict["L{:d}".format(L)][name] = {}
	for L in Ls:
		temperatures = np.load(os.path.join(results_dir, J, "magnetization", "L{:d}".format(L), "measurements.npz"))["temperatures"]
		u4samples = U4_samples(results_dir, J, "magnetization", L)
		tc_samples = critical_temperature_samples(temperatures, u4samples)
		tc_mean, tc_std = jackknife.calculate_mean_std(tc_samples, remove_bias=remove_bias)
		output_dict["L{:d}".format(L)]["magnetization"]["mean"] = float(tc_mean)
		output_dict["L{:d}".format(L)]["magnetization"]["std"] = float(tc_std)
		for name in encoder_names:
			for stats in ["means", "stds"]:
				output_dict["L{:d}".format(L)][name][stats] = []
		for N in Ns:
			for name in encoder_names:
				temperatures = np.load(os.path.join(results_dir, J, name, "L{:d}".format(L), "N{:d}".format(N), "measurements.npz"))["temperatures"]
				u4samples = U4_samples(results_dir, J, name, L, N)
				tc_samples = critical_temperature_samples(temperatures, u4samples)
				tc_mean, tc_std = jackknife.calculate_mean_std(tc_samples, remove_bias=remove_bias)
				output_dict["L{:d}".format(L)][name]["means"].append( float(tc_mean) )
				output_dict["L{:d}".format(L)][name]["stds"].append( float(tc_std) )
	output_dir = os.path.join(results_dir, "processed", J)
	os.makedirs(output_dir, exist_ok=True)
	suffix = "_biased" if not remove_bias else ""
	with open(os.path.join(output_dir, "tc{}.json".format(suffix)), "w") as fp:
		json.dump(output_dict, fp, indent=2)


### process execution times

def calculate_times(results_dir, Js, Ls, Ns, encoder_names):
	output_dict = {"Ls": Ls, "preprocessing_means": [], "preprocessing_stds": []}
	for name in encoder_names:
		for stat in ["mean", "std"]:
			output_dict["{}_{}s".format(name, stat)] = []
	for L in Ls:
		times = []
		for J in Js:
			times_J = []
			for dir in sorted(os.listdir(os.path.join("data", J, "L{:d}".format(L)))):
				with open(os.path.join("data", J, "L{:d}".format(L), dir, "time_symmetric.txt"), "r") as f:
					times_J.append( float(f.read()) )
			times.append(times_J)
		times = np.array(times).sum(1)
		output_dict["preprocessing_means"].append( times.mean() )
		output_dict["preprocessing_stds"].append( times.std() )
	for L in Ls:
		for name in encoder_names:
			times = []
			for J in Js:
				for N in Ns:
					with open(os.path.join(results_dir, J, name, "L{:d}".format(L), "N{:d}".format(N), "results.json"), "r") as f:
						times.append( json.load(f)["time"] )
			times = np.array(times)
			output_dict["{}_means".format(name)].append( times.mean() )
			output_dict["{}_stds".format(name)].append( times.std() )
	output_dict = {key: np.asarray(value) for (key, value) in output_dict.items()}
	output_dir = os.path.join(results_dir, "processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "times.npz"), **output_dict)


### process symmetry generators

def calculate_generators(results_dir, Js, Ls, Ns, encoder_names):
	generator_types = ["spatial", "internal"]
	output_dict = {J: {generator_type: {} for generator_type in generator_types} for J in Js}
	for J in Js:
		for generator_type in generator_types:
			for name in encoder_names:
				output_dict[J][generator_type][name] = {}
				gens = []
				for L in Ls:
					for N in Ns:
						with open(os.path.join(results_dir, J, name, "L{:d}".format(L), "N{:d}".format(N), "generator_reps.json"), "r") as fp:
							gens.append( json.load(fp)[generator_type] )
				gens = np.array(gens)
				output_dict[J][generator_type][name]["mean"] = float( gens.mean() )
				output_dict[J][generator_type][name]["std"] = float( gens.std() )
	output_dir = os.path.join(results_dir, "processed")
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "generators.json"), "w") as fp:
		json.dump(output_dict, fp, indent=2)


if __name__ == "__main__":
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	Ns = [16, 32, 64, 128, 256, 512, 1024, 2048]
	encoder_names = ["latent", "latent_equivariant", "latent_multiscale"]

#	print("Gathering magnetizations . . . ")
#	for J in Js:
#		for L in Ls:
#			gather_magnetizations("data", "results", J, L)

	print("Calculating stats . . . ")
	for J in Js:
		calculate_stats("results", J, "magnetization", 128)
		for name in encoder_names:
			calculate_stats("results", J, name, 128, N=2048)

	print("Calculating critical temperatures . . . ")
	for J in Js:
		for remove_bias in [True, False]:
			calculate_critical_temperatures("results", J, Ls, Ns, encoder_names, remove_bias=remove_bias)

	print("Calculating execution times . . . ")
	calculate_times("results", Js, Ls, Ns, encoder_names)

	print("Calculating symmetry generators . . . ")
	GE_encoder_names = ["latent_equivariant", "latent_multiscale"]
	calculate_generators("results", Js, Ls, Ns, GE_encoder_names)

	print("Done!")
