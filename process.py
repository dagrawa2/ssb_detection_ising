import os
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import lstsq

from phasefinder import jackknife
from phasefinder.datasets import Ising


###process magnetization, order parameter curves, and U_4 Binder cumulant curves

def gather_magnetizations(data_dir, results_dir, J, L):
	temperatures = []
	measurements = []
	L_dir = os.path.join(data_dir, J, "L{:d}".format(L))
	for temperature_dir in sorted(os.listdir(L_dir)):
		if temperature_dir[0] != "T": continue
		I = Ising()
		Ms = I.magnetization(os.path.join(L_dir, temperature_dir), per_spin=True, staggered=(J=="antiferromagnetic"))
		temperatures.append(I.T)
		measurements.append(Ms)
	temperatures = np.array(temperatures)
	measurements = np.stack(measurements, 0)
	output_dir = "{}/{}/magnetization/L{:d}".format(results_dir, J, L)
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "measurements.npz"), temperatures=temperatures, measurements=measurements)


def calculate_stats(results_dir, J, observable_name, L, N=None, L_test=None, bins=50):
	dir = os.path.join(J, observable_name, "L{:d}".format(L))
	for (prefix, value) in [("N", N), ("L", L_test)]:
		if value is not None:
			dir = os.path.join(dir, "{}{:d}".format(prefix, value))
	measurements = np.load(os.path.join(results_dir, dir, "measurements.npz"))
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
	output_dir = os.path.join(results_dir, "processed", dir)
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "stats.npz"), temperatures=temperatures, distributions=distributions, distribution_range=distribution_range, order_means=order_means, order_stds=order_stds, u4_means=u4_means, u4_stds=u4_stds)


### process critical temperature estimates

def U4_samples(results_dir, J, observable_name, L, N=None, L_test=None):
	if L_test == L:
		L_test = None
	dir = os.path.join(results_dir, J, observable_name, "L{:d}".format(L))
	for (prefix, value) in [("N", N), ("L", L_test)]:
		if value is not None:
			dir = os.path.join(dir, "{}{:d}".format(prefix, value))
	measurements = np.load(os.path.join(dir, "measurements.npz"))["measurements"].T
	samples_2 = jackknife.calculate_samples(measurements**2)
	samples_4 = jackknife.calculate_samples(measurements**4)
	samples = 1 - samples_4/(3*samples_2**2)
	return samples


def critical_temperature_samples_max_derivative(temperatures, u4samples):
	tc_samples = []
	for samples in u4samples:
		cs = CubicSpline(temperatures, samples)
		roots = cs.derivative(2).roots(extrapolate=False)
		values = np.abs(cs(roots, nu=1))
		tc_samples.append( roots[np.argmax(values)] )
	tc_samples = np.array(tc_samples)
	return tc_samples


def critical_temperature_samples_step_function(temperatures, u4samples):
	n_samples, n_temperatures = u4samples.shape
	nums = np.arange(n_temperatures+1)[None,:]
	nums[0,0] = 1
	cumsums = np.concatenate([np.zeros((n_samples, 1)), np.cumsum(u4samples, 1)], 1)
	means_forward = cumsums/nums
	means_backward = (cumsums[:,-1,None]-cumsums)/nums[:,::-1]
	tc_samples = []
	for i in range(n_samples):
		means = np.tril(np.repeat(means_forward[i,:,None], n_temperatures, axis=1), k=-1) \
			+ np.triu(np.repeat(means_backward[i,:,None], n_temperatures, axis=1), k=0)
		j = np.argmin( np.mean((u4samples[None,i,:]-means)**2, 1) )
		tc_samples.append( (temperatures[j]+temperatures[j-1])/2 )
	tc_samples = np.array(tc_samples)
	return tc_samples


def critical_temperature_samples(temperatures, u4samples, method="max_derivative"):
	if method == "max_derivative":
		return critical_temperature_samples_max_derivative(temperatures, u4samples)
	if method == "step_function":
		return critical_temperature_samples_step_function(temperatures, u4samples)


def lstsq_samples(x, y_samples, weights=None):
	ones = np.ones((x.shape[0], 1))
	if weights is None:
		weights = ones
	if weights.ndim == 1:
		weights = weights[:,None]
	if x.ndim == 1:
		x = x[:,None]
	x = np.concatenate((ones, x), 1)
	fit_samples = lstsq(x*weights, y_samples*weights)[0]
	yhat_samples = x.dot(fit_samples)
	y_samples_centered = y_samples - y_samples.mean(0, keepdims=True)
	yhat_samples_centered = yhat_samples - yhat_samples.mean(0, keepdims=True)
	r2_samples = (y_samples_centered*yhat_samples_centered).sum(0)**2/( np.sum(y_samples_centered**2, 0)*np.sum(yhat_samples_centered**2, 0) )
	return fit_samples, r2_samples


def calculate_critical_temperatures(results_dir, J, Ls, Ns, encoder_names, remove_bias=True, method="max_derivative", jitter=1e-9):
	output_dict = {"L{:d}".format(L): {"Ns": Ns, "magnetization": {}} for L in Ls}
	for L in Ls:
		for name in encoder_names:
			output_dict["L{:d}".format(L)][name] = {}
	for L in Ls:
		temperatures = np.load(os.path.join(results_dir, J, "magnetization", "L{:d}".format(L), "measurements.npz"))["temperatures"]
		u4samples = U4_samples(results_dir, J, "magnetization", L)
		tc_samples = critical_temperature_samples(temperatures, u4samples, method=method)
		tc_mean, tc_std = jackknife.calculate_mean_std(tc_samples, remove_bias=remove_bias)
		output_dict["L{:d}".format(L)]["magnetization"]["samples"] = tc_samples.tolist()
		output_dict["L{:d}".format(L)]["magnetization"]["mean"] = float(tc_mean)
		output_dict["L{:d}".format(L)]["magnetization"]["std"] = float(tc_std)
		for name in encoder_names:
			for stats in ["samples", "means", "stds"]:
				output_dict["L{:d}".format(L)][name][stats] = []
		for N in Ns:
			for name in encoder_names:
				if not os.path.exists(os.path.join(results_dir, J, name, "L{:d}".format(L), "N{:d}".format(N))):
					output_dict["L{:d}".format(L)][name]["samples"].append("nan")
					output_dict["L{:d}".format(L)][name]["means"].append("nan")
					output_dict["L{:d}".format(L)][name]["stds"].append("nan")
					continue
				temperatures = np.load(os.path.join(results_dir, J, name, "L{:d}".format(L), "N{:d}".format(N), "measurements.npz"))["temperatures"]
				u4samples = U4_samples(results_dir, J, name, L, N)
				tc_samples = critical_temperature_samples(temperatures, u4samples, method=method)
				tc_mean, tc_std = jackknife.calculate_mean_std(tc_samples, remove_bias=remove_bias)
				output_dict["L{:d}".format(L)][name]["samples"].append( tc_samples.tolist() )
				output_dict["L{:d}".format(L)][name]["means"].append( float(tc_mean) )
				output_dict["L{:d}".format(L)][name]["stds"].append( float(tc_std) )
	def np2py(*args):
		if type(args[0]) == np.ndarray:
			return tuple([arg.tolist() for arg in args])
		return tuple([float(arg) for arg in args])
	for n_Ls in range(2, len(Ls)+1):
		Ls_partial = Ls[:n_Ls]
		x = 1/np.array(Ls_partial)
		y_samples = np.array([output_dict["L{:d}".format(L)]["magnetization"]["samples"] for L in Ls_partial])
		weights = 1/(np.array([output_dict["L{:d}".format(L)]["magnetization"]["std"] for L in Ls_partial]) + jitter)
		fit_samples, r2_samples = lstsq_samples(x, y_samples, weights=weights)
		samples = {"infty": fit_samples[0].T, "fit": fit_samples.T, "r2": r2_samples}
		for (key, value) in samples.items():
			key_n_Ls = "{}_{:d}".format(key, n_Ls)
			mean, std = np2py( *jackknife.calculate_mean_std(value) )
			output_dict[key_n_Ls] = {"Ns": Ns, "magnetization": {"mean": mean, "std": std}}
			for name in encoder_names:
				output_dict[key_n_Ls][name] = {"means": [], "stds": []}
		for (i, N) in enumerate(Ns):
			for name in encoder_names:
				if "nan" in [output_dict["L{:d}".format(L)][name]["samples"][i] for L in Ls_partial]:
					for key in ["infty", "fit", "r2"]:
						output_dict["{}_{:d}".format(key, n_Ls)][name]["means"].append("nan")
						output_dict["{}_{:d}".format(key, n_Ls)][name]["stds"].append("nan")
					continue
				y_samples = np.array([output_dict["L{:d}".format(L)][name]["samples"][i] for L in Ls_partial])
				weights = 1/(np.array([output_dict["L{:d}".format(L)][name]["stds"][i] for L in Ls_partial]) + jitter)
				fit_samples, r2_samples = lstsq_samples(x, y_samples, weights=weights)
				samples = {"infty": fit_samples[0].T, "fit": fit_samples.T, "r2": r2_samples}
				for (key, value) in samples.items():
					key_n_Ls = "{}_{:d}".format(key, n_Ls)
					mean, std = np2py( *jackknife.calculate_mean_std(value) )
					output_dict[key_n_Ls][name]["means"].append(mean)
					output_dict[key_n_Ls][name]["stds"].append(std)
	for L in Ls:
		for name in ["magnetization"] + encoder_names:
			del output_dict["L{:d}".format(L)][name]["samples"]
	output_dir = os.path.join(results_dir, "processed", J)
	os.makedirs(output_dir, exist_ok=True)
	suffix = "_biased" if not remove_bias else ""
	with open(os.path.join(output_dir, "tc{}.json".format(suffix)), "w") as fp:
		json.dump(output_dict, fp, indent=2)


def calculate_critical_temperatures_cross_lattice_sizes(results_dir, J, Ls, Ns, encoder_name, remove_bias=True, method="max_derivative"):
	output_dict = {"encoder_name": encoder_name, "Ls": Ls, "Ns": Ns, "means": [], "stds": []}
	for N in Ns:
		mat = {"means": [], "stds": []}
		for L in Ls:
			row = {"means": [], "stds": []}
			temperatures = np.load(os.path.join(results_dir, J, encoder_name, "L{:d}".format(L), "N{:d}".format(N), "measurements.npz"))["temperatures"]
			for L_test in Ls:
				u4samples = U4_samples(results_dir, J, encoder_name, L, N, L_test)
				tc_samples = critical_temperature_samples(temperatures, u4samples, method=method)
				tc_mean, tc_std = jackknife.calculate_mean_std(tc_samples, remove_bias=remove_bias)
				row["means"].append( float(tc_mean) )
				row["stds"].append( float(tc_std) )
			mat["means"].append(row["means"])
			mat["stds"].append(row["stds"])
		output_dict["means"].append(mat["means"])
		output_dict["stds"].append(mat["stds"])
	output_dir = os.path.join(results_dir, "processed", J)
	os.makedirs(output_dir, exist_ok=True)
	suffix = "_biased" if not remove_bias else ""
	with open(os.path.join(output_dir, "tc_cross{}.json".format(suffix)), "w") as fp:
		json.dump(output_dict, fp, indent=2)


### process execution times and error vs time trends

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
				if dir[0] != "T": continue
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


def calculate_error_vs_time(results_dir, J, Ls, N, observable_names, preprocess_bools, remove_bias=True):
	output_dict = {name: {"times_excluding_training": [], "times": [], "errors": []} for name in observable_names}
	tc = 2/np.log(1+np.sqrt(2))
	for L in Ls:
		data_times_temp = []
		preprocess_times_temp = []
		data_dir = os.path.join("data", J, "L{:d}".format(L))
		for dir in sorted(os.listdir(data_dir)):
			if dir[0] != "T": continue
			with open(os.path.join(data_dir, dir, "time.txt"), "r") as fp:
				data_times_temp.append( float(fp.read()) )
			with open(os.path.join(data_dir, dir, "time_symmetric.txt"), "r") as fp:
				preprocess_times_temp.append( float(fp.read()) )
		data_time = sum(data_times_temp)
		preprocess_time = sum(preprocess_times_temp)
		for (name, preprocess) in zip(observable_names, preprocess_bools):
			output_dict[name]["times_excluding_training"].append(data_time)
			if preprocess:
				output_dict[name]["times_excluding_training"][-1] += preprocess_time
			if name == "magnetization":
				output_dict[name]["times"].append( output_dict[name]["times_excluding_training"][-1] )
			else:
				with open(os.path.join(results_dir, J, name, "L{:d}".format(L), "N{:d}".format(N), "results.json"), "r") as fp:
					output_dict[name]["times"].append( json.load(fp)["time"] + output_dict[name]["times_excluding_training"][-1] )
			suffix = "" if remove_bias else "_biased"
			with open(os.path.join(results_dir, "processed", J, "tc{}.json".format(suffix)), "r") as fp:
				if name == "magnetization":
					tc_estimate = json.load(fp)["L{:d}".format(L)][name]["mean"]
				else:
					tc_data = json.load(fp)["L{:d}".format(L)]
					tc_estimate = tc_data[name]["means"][tc_data["Ns"].index(N)]
			output_dict[name]["errors"].append( 100*(tc_estimate/tc-1) )
	output_dir = os.path.join(results_dir, "processed", J, "N{:d}".format(N))
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "error_vs_time.json"), "w") as fp:
		json.dump(output_dict, fp, indent=2)


def calculate_error_vs_time_infty(results_dir, J, Ls, N, observable_names, multiscale_bools, remove_bias=True):
	output_dict = {name: {"times": [], "errors": []} for name in observable_names}
	tc = 2/np.log(1+np.sqrt(2))
	suffix = "" if remove_bias else "_biased"
	with open(os.path.join(results_dir, "processed", J, "tc{}.json".format(suffix)), "r") as fp:
		tc_dict = json.load(fp)
	with open(os.path.join(results_dir, "processed", J, "N{:d}".format(N), "error_vs_time.json"), "r") as fp:
		errortime_dict = json.load(fp)
	for n_Ls in range(2, len(Ls)+1):
		Ls_partial = Ls[:n_Ls]
		for (name, multiscale) in zip(observable_names, multiscale_bools):
			if multiscale:
				with open(os.path.join(results_dir, J, name, "L{:d}".format(Ls[0]), "N{:d}".format(N), "results.json"), "r") as fp:
					output_dict[name]["times"].append( sum(errortime_dict[name]["times_excluding_training"][:n_Ls]) + json.load(fp)["time"] )
			else:
				output_dict[name]["times"].append( sum(errortime_dict[name]["times"][:n_Ls]) )
			if name == "magnetization":
				output_dict[name]["errors"].append( tc_dict["infty_{:d}".format(n_Ls)][name]["mean"] )
			else:
				i = tc_dict["infty_{:d}".format(n_Ls)]["Ns"].index(N)
				output_dict[name]["errors"].append( tc_dict["infty_{:d}".format(n_Ls)][name]["means"][i] )
			output_dict[name]["errors"][-1] = 100*(output_dict[name]["errors"][-1]/tc-1)
	output_dir = os.path.join(results_dir, "processed", J, "N{:d}".format(N))
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "error_vs_time_infty.json"), "w") as fp:
		json.dump(output_dict, fp, indent=2)


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
	results_dir = "results"
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	Ns = [2, 4, 8, 16, 32, 64, 128, 256]
	encoder_names = ["latent", "latent_equivariant", "latent_multiscale_4"]

#	print("Gathering magnetizations . . . ")
#	for J in Js:
#		for L in Ls:
#			gather_magnetizations("data", results_dir, J, L)

	print("Calculating stats . . . ")
	for J in Js:
		calculate_stats(results_dir, J, "magnetization", 128)
		for name in encoder_names:
			calculate_stats(results_dir, J, name, 128, N=128)

	print("Calculating critical temperatures . . . ")
	for J in Js:
		print("J:", J)
		for remove_bias in [False]:  # , True]:
			print("remove bias:", remove_bias)
			calculate_critical_temperatures(results_dir, J, Ls, Ns, encoder_names, remove_bias=remove_bias, method="step_function")
#			calculate_critical_temperatures_cross_lattice_sizes(results_dir, J, Ls, Ns, "latent_equivariant", remove_bias=remove_bias, method="max_derivative")

	print("Calculating execution times . . . ")
	calculate_times(results_dir, Js, Ls, Ns, ["latent", "latent_equivariant"])

	print("Calculating error vs time . . . ")
	observable_names = ["magnetization"] + encoder_names
	for J in Js:
		calculate_error_vs_time(results_dir, J, Ls, 128, observable_names, [False, False, True, True, True], remove_bias=False)
		calculate_error_vs_time_infty(results_dir, J, Ls, 128, observable_names, [False, False, False, True, True], remove_bias=False)

	print("Calculating symmetry generators . . . ")
	GE_encoder_names = ["latent_equivariant", "latent_multiscale_4"]
	calculate_generators(results_dir, Js, Ls, [N for N in Ns if N >= 8], GE_encoder_names)

	print("Done!")
