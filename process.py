import os
import json
import itertools
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import lstsq

from phasefinder import jackknife
from phasefinder.datasets import Ising


def build_path(results_dir, J, observable_name, L, N=None, fold=None, seed=None, L_test=None, subdir=None):
	if subdir is not None:
		results_dir = os.path.join(results_dir, subdir)
	path = os.path.join(results_dir, J, observable_name)
	if L is not None:
		path = os.path.join(path, "L{:d}".format(L))
	if observable_name == "magnetization":
		return path
	path = os.path.join(path, "N{:d}".format(N), "fold{:d}".format(fold), "seed{:d}".format(seed))
	if L_test is None or L_test == L:
		return path
	path = os.path.join(path, "L{:d}".format(L_test))
	return path


###process magnetization, order parameter curves, and U_4 Binder cumulant curves

def gather_magnetizations(data_dir, results_dir, J, L):
	temperatures = []
	measurements = []
	L_dir = os.path.join(data_dir, J, "L{:d}".format(L))
	for temperature_dir in sorted(os.listdir(L_dir)):
		if temperature_dir[0] != "T":
			continue
		I = Ising()
		Ms = I.magnetization(os.path.join(L_dir, temperature_dir), per_spin=True, staggered=(J=="antiferromagnetic"))
		temperatures.append(I.T)
		measurements.append(Ms)
	temperatures = np.array(temperatures)
	measurements = np.stack(measurements, 0)
	output_dir = "{}/{}/magnetization/L{:d}".format(results_dir, J, L)
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "measurements.npz"), temperatures=temperatures, measurements=measurements)


def calculate_stats(results_dir, J, observable_name, L, N_test, N=None, fold=None, seed=None, L_test=None, bins=50):
	dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=L_test)
	measurements = np.load(os.path.join(dir, "measurements.npz"))
	temperatures = measurements["temperatures"]
	measurements = measurements["measurements"].T[:N_test]
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
	output_dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=L_test, subdir="processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "stats.npz"), temperatures=temperatures, distributions=distributions, distribution_range=distribution_range, order_means=order_means, order_stds=order_stds, u4_means=u4_means, u4_stds=u4_stds)


### process critical temperature estimates

def U4_samples(measurements):
	samples_2 = jackknife.calculate_samples(measurements**2)
	samples_4 = jackknife.calculate_samples(measurements**4)
	samples = 1 - samples_4/(3*samples_2**2)
	return samples


def critical_temperature_samples(temperatures, u4samples):
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


def calculate_critical_temperatures(results_dir, J, observable_name, L, N_tests, N=None, fold=None, seed=None, L_test=None):
	dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=L_test)
	measurements = np.load(os.path.join(dir, "measurements.npz"))
	temperatures = measurements["temperatures"]
	measurements = measurements["measurements"].T
	output_dict = {"N_tests": N_tests, "means": [], "stds": [], "biases": [], "samples": []}
	for N_test in N_tests:
		samples = critical_temperature_samples(temperatures, U4_samples(measurements[:N_test]))
		mean, std, bias = jackknife.calculate_mean_std(samples, remove_bias=True, return_bias=True)
		samples = np.pad(samples, ((0, max(N_tests)-N_test),), constant_values=np.nan)
		for (key, value) in [("means", mean), ("stds", std), ("biases", bias), ("samples", samples)]:
			output_dict[key].append(value)
	output_dict["samples"] = np.stack(output_dict["samples"], 0)
	for (key, value) in output_dict.items():
		output_dict[key] = np.asarray(value)
	output_dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=L_test, subdir="processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "tc.npz"), **output_dict)


def calculate_critical_temperature_extrapolates(results_dir, J, observable_name, Ls, N_tests, N=None, fold=None, seed=None, jitter=1e-9):
	load_tc = lambda L: np.load(os.path.join(build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "tc.npz"))
	x = 1/np.array(Ls)
	y_samples = np.stack([load_tc(L)["samples"] for L in Ls], 1)
	weights = 1/(np.stack([load_tc(L)["stds"] for L in Ls], 1) + jitter)
	n_Lss = np.arange(2, len(Ls)+1)
	output_dict = {"N_tests": np.asarray(N_tests), "n_Lss": n_Lss}
	for key in ["yintercept", "slope", "r2"]:
		for stats in ["means", "stds", "biases"]:
			output_dict["{}_{}".format(key, stats)] = []
	for (i, N_test) in enumerate(N_tests):
		for n_Ls in n_Lss:
			fit_samples, r2_samples = lstsq_samples(x[:n_Ls], y_samples[i,:n_Ls,:N_test], weights=weights[i,:n_Ls])
			samples = np.concatenate([fit_samples, r2_samples[None,:]], 0).T
			mean, std, bias = jackknife.calculate_mean_std(samples, remove_bias=True, return_bias=True)
			for (stats, value) in [("means", mean), ("stds", std), ("biases", bias)]:
				for (j, key) in enumerate(["yintercept", "slope", "r2"]):
					output_dict["{}_{}".format(key, stats)].append(value[j])
	for (key, value) in output_dict.items():
		if "means" in key or "stds" in key or "biases" in key:
			output_dict[key] = np.array(value).reshape((len(N_tests), len(n_Lss)))
	output_dir = build_path(results_dir, J, observable_name, None, N, fold, seed, subdir="processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "tc_extrapolate.npz"), **output_dict)


### process execution times

def calculate_times(results_dir, J, observable_name, L, N_tests, N=None, fold=None, seed=None, L_test=None):
	L_dir = os.path.join("data", J, "L{:d}".format(L))	
	T_dirs = sorted(os.listdir(L_dir))
	T_dirs = [os.path.join(L_dir, dir) for dir in T_dirs if dir[0] == "T"]
	with open(os.path.join(T_dirs[0], "args.json"), "r") as fp:
		args = json.load(fp)
	N_max = args["nmcs"]//args["nmeas"]
	after_ieq_frac = args["nmcs"]/(args["ieq"] + args["nmcs"])
	generation_time = 0
	for dir in T_dirs:
		with open(os.path.join(dir, "time.txt"), "r") as fp:
			generation_time += float(fp.read())
	generation_time *= after_ieq_frac
	if observable_name == "magnetization":
		N = 0
		training_time = 0
		processing_time = 0
	else:
		dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=None)
		with open(os.path.join(dir, "results.json"), "r") as fp:
			results = json.load(fp)
		training_time = results["time"]
		with open(os.path.join(L_dir, "aggregate", "times.json"), "r") as fp:
			key = "states_symmetric" if results["args"]["symmetric"] else "states"
			processing_time = json.load(fp)[key]
	output_dict = {"N_tests": N_tests, "times": []}
	for N_test in N_tests:
		t = training_time \
			+ (generation_time + processing_time) * (N+N_test)/N_max
		output_dict["times"].append(t)
	output_dict = {key: np.array(value) for (key, value) in output_dict.items()}
	output_dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=L_test, subdir="processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "times.npz"), **output_dict)


def calculate_time_extrapolates(results_dir, J, observable_name, Ls, N_tests, N=None, fold=None, seed=None):
	load_times = lambda L: np.load(os.path.join(build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "times.npz"))
	times = np.stack([load_times(L)["times"] for L in Ls], 1)
	n_Lss = np.arange(2, len(Ls)+1)
	output_dict = {"N_tests": np.asarray(N_tests), "n_Lss": n_Lss}
	output_dict["times"] = np.stack([times[:,:n_Ls].sum(1) for n_Ls in n_Lss], 1)
	output_dir = build_path(results_dir, J, observable_name, None, N, fold, seed, subdir="processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "times_extrapolate.npz"), **output_dict)


### gather critical temperature vs time data

def gather_tc_vs_time(results_dir, J, observable_name, Ls, Ns=None, folds=None, seeds=None):
	suffixes = ["means", "stds", "biases"]
	output_dict = {"times": []}
	for s in suffixes:
		output_dict["tc_"+s] = []
	if observable_name == "magnetization":
		for L in Ls:
			with np.load(os.path.join(build_path(results_dir, J, observable_name, L, subdir="processed"), "times.npz")) as D:
				output_dict["times"].append(D["times"])
			with np.load(os.path.join(build_path(results_dir, J, observable_name, L, subdir="processed"), "tc.npz")) as D:
				for s in suffixes:
					output_dict["tc_"+s].append(D[s])
	else:
		for (L, N, fold, seed) in itertools.product(Ls, Ns, folds, seeds):
			with np.load(os.path.join(build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "times.npz")) as D:
				output_dict["times"].append(D["times"])
			with np.load(os.path.join(build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "tc.npz")) as D:
				for s in suffixes:
					output_dict["tc_"+s].append(D[s])
	output_dict = {key: np.concatenate(value, 0) for (key, value) in output_dict.items()}
	output_dir = os.path.join(results_dir, "processed", J, observable_name)
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "tc_vs_time.npz"), **output_dict)


def gather_tc_vs_time_extrapolate(results_dir, J, observable_name, Ns=None, folds=None, seeds=None):
	suffixes = ["means", "stds", "biases"]
	output_dict = {"times": []}
	for s in suffixes:
		output_dict["tc_"+s] = []
	if observable_name == "magnetization":
		with np.load(os.path.join(build_path(results_dir, J, observable_name, None, subdir="processed"), "times_extrapolate.npz")) as D:
			output_dict["times"].append(D["times"])
		with np.load(os.path.join(build_path(results_dir, J, observable_name, None, subdir="processed"), "tc_extrapolate.npz")) as D:
			for s in suffixes:
				output_dict["tc_"+s].append(D["yintercept_"+s])
	else:
		for (N, fold, seed) in itertools.product(Ns, folds, seeds):
			with np.load(os.path.join(build_path(results_dir, J, observable_name, None, N=N, fold=fold, seed=seed, subdir="processed"), "times_extrapolate.npz")) as D:
				output_dict["times"].append(D["times"])
			with np.load(os.path.join(build_path(results_dir, J, observable_name, None, N=N, fold=fold, seed=seed, subdir="processed"), "tc_extrapolate.npz")) as D:
				for s in suffixes:
					output_dict["tc_"+s].append(D["yintercept_"+s])
	output_dict = {key: np.concatenate(value, 0).T for (key, value) in output_dict.items()}
	output_dir = os.path.join(results_dir, "processed", J, observable_name)
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "tc_vs_time_extrapolate.npz"), **output_dict)


### process symmetry generators

def calculate_generators(results_dir, Js, encoder_names, Ls, Ns, folds, seeds):
	generator_types = ["spatial", "internal"]
	output_dict = {J: {name: {gen_type: {} for gen_type in generator_types} for name in encoder_names} for J in Js}
	for (J, name, gen_type) in itertools.product(Js, encoder_names, generator_types):
		gens = []
		for (L, N, fold, seed) in itertools.product(Ls, Ns, folds, seeds):
			with open(os.path.join(build_path(results_dir, J, name, L, N=N, fold=fold, seed=seed), "generator_reps.json"), "r") as fp:
				gens.append( json.load(fp)[gen_type] )
		gens = np.array(gens)
		output_dict[J][name][gen_type]["mean"] = float(gens.mean())
		output_dict[J][name][gen_type]["std"] = float(gens.std())
	output_dir = os.path.join(results_dir, "processed")
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "generators.json"), "w") as fp:
		json.dump(output_dict, fp, indent=2)


if __name__ == "__main__":
	results_dir = "results3"
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	Ns = [8, 16, 32, 64, 128, 256]
	N_tests = [256, 512, 1024, 2048]
	observable_names = ["magnetization", "latent", "latent_equivariant"]
	encoder_names = ["latent", "latent_equivariant", "latent_multiscale_2", "latent_multiscale_4"]

	Ns_dict = {name: Ns for name in encoder_names}
	Ns_dict["latent_multiscale_2"] = [N for N in Ns if N >= 8]
	Ns_dict["latent_multiscale_4"] = [N for N in Ns if N >= 8]

	folds = [0, 1, 2, 3]
	seeds = [0, 1, 2]

#	print("Gathering magnetizations . . . ")
#	for J in Js:
#		for L in Ls:
#			gather_magnetizations("data", results_dir, J, L)

	print("Calculating stats . . . ")
	for J in Js:
		for name in observable_names:
			calculate_stats(results_dir, J, name, 128, 2048, N=256, fold=0, seed=0)

	print("Calculating critical temperatures . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		for L in Ls:
			calculate_critical_temperatures(results_dir, J, "magnetization", L, N_tests)
		for encoder_name in encoder_names:
			print(encoder_name)
			for (L, N, fold, seed) in itertools.product(Ls, Ns_dict[encoder_name], folds, seeds):
				if os.path.exists(os.path.join(build_path(results_dir, J, encoder_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "tc.npz")):
					continue
				calculate_critical_temperatures(results_dir, J, encoder_name, L, N_tests, N=N, fold=fold, seed=seed)

	print("Calculating critical temperature extrapolates . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		calculate_critical_temperature_extrapolates(results_dir, J, "magnetization", Ls, N_tests)
		for encoder_name in encoder_names:
			print(encoder_name)
			for (N, fold, seed) in itertools.product(Ns_dict[encoder_name], folds, seeds):
				calculate_critical_temperature_extrapolates(results_dir, J, encoder_name, Ls, N_tests, N=N, fold=fold, seed=seed)

	print("Calculating execution times . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		for L in Ls:
			calculate_times(results_dir, J, "magnetization", L, N_tests)
		for encoder_name in encoder_names:
			print(encoder_name)
			for (L, N, fold, seed) in itertools.product(Ls, Ns_dict[encoder_name], folds, seeds):
				calculate_times(results_dir, J, encoder_name, L, N_tests, N=N, fold=fold, seed=seed)

	print("Calculating execution time extrapolates . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		for L in Ls:
			calculate_time_extrapolates(results_dir, J, "magnetization", Ls, N_tests)
		for encoder_name in encoder_names:
			print(encoder_name)
			for (N, fold, seed) in itertools.product(Ns_dict[encoder_name], folds, seeds):
				calculate_time_extrapolates(results_dir, J, encoder_name, Ls, N_tests, N=N, fold=fold, seed=seed)

	print("Gathering T_c vs time data . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		gather_tc_vs_time(results_dir, J, "magnetization", Ls)
		for encoder_name in encoder_names:
			print(encoder_name)
			gather_tc_vs_time(results_dir, J, encoder_name, Ls, Ns=Ns_dict[encoder_name], folds=folds, seeds=seeds)

	print("Gathering T_c vs time extrapolate data . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		gather_tc_vs_time_extrapolate(results_dir, J, "magnetization")
		for encoder_name in encoder_names:
			print(encoder_name)
			gather_tc_vs_time_extrapolate(results_dir, J, encoder_name, Ns=Ns_dict[encoder_name], folds=folds, seeds=seeds)

	print("Combining multiscale T_c vs time extrapolate data . . . ")
	row_dict = {"latent_multiscale_2": 0, "latent_multiscale_4": 2}
	for J in Js:
		Ds = []
		for (name, row) in row_dict.items():
			with np.load(os.path.join(results_dir, "processed", J, name, "tc_vs_time_extrapolate.npz")) as fp:
				D = dict(fp)
			Ds.append( {key: value[row] for (key, value) in D.items()} )
		D = {key: np.stack([D[key] for D in Ds], 0) for key in Ds[0].keys()}
		output_dir = os.path.join(results_dir, "processed", J, "latent_multiscale")
		os.makedirs(output_dir, exist_ok=True)
		np.savez(os.path.join(output_dir, "tc_vs_time_extrapolate.npz"), **D)

	print("Calculating symmetry generators . . . ")
	calculate_generators(results_dir, Js, ["latent_equivariant"], Ls, Ns, folds, seeds)

	print("Done!")
