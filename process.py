import os
import json
import itertools
import numpy as np
from scipy.linalg import lstsq
from types import SimpleNamespace

from phasefinder import jackknife
from phasefinder.datasets import Ising
from phasefinder.utils import build_path


###process magnetization, order parameter curves, and U_4 Binder cumulant curves

def gather_magnetizations(data_dir, results_dir, J, L, N_test=None):
	temperatures = []
	measurements = []
	L_dir = os.path.join(data_dir, J, "L{:d}".format(L))
	for temperature_dir in sorted(os.listdir(L_dir)):
		if temperature_dir[0] != "T":
			continue
		I = Ising()
		Ms = I.magnetization(os.path.join(L_dir, temperature_dir), per_spin=True, staggered=(J=="antiferromagnetic"))[-2::-2]
		temperatures.append(I.T)
		measurements.append(Ms)
	temperatures = np.array(temperatures)
	measurements = np.stack(measurements, 0)
	if N_test is not None:
		measurements = measurements[:,:N_test]
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


def calculate_critical_temperatures(results_dir, J, observable_name, L, N=None, fold=None, seed=None, L_test=None):
	dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=L_test)
	measurements = np.load(os.path.join(dir, "measurements.npz"))
	temperatures = measurements["temperatures"]
	measurements = measurements["measurements"].T
	samples = critical_temperature_samples(temperatures, U4_samples(measurements))
	mean, std, bias = jackknife.calculate_mean_std(samples, remove_bias=True, return_bias=True)
	output_dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=L_test, subdir="processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "tc.npz"), mean=mean, std=std, bias=bias, samples=samples)


def calculate_critical_temperature_extrapolates(results_dir, J, observable_name, Ls, N=None, fold=None, seed=None, jitter=1e-9):
	load_tc = lambda L: np.load(os.path.join(build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "tc.npz"))
	x = 1/np.array(Ls)
	y_samples = np.stack([load_tc(L)["samples"] for L in Ls], 0)
	weights = 1/(np.array([float(load_tc(L)["std"]) for L in Ls]) + jitter)
	fit_samples, r2_samples = lstsq_samples(x, y_samples, weights=weights)
	samples = np.concatenate([fit_samples, r2_samples[None,:]], 0).T
	mean, std, bias = jackknife.calculate_mean_std(samples, remove_bias=True, return_bias=True)
	stats = ["mean", "std", "bias"]
	values = [mean, std, bias]
	kwds = ["yintercept", "slope", "r2"]
	output_dict = {stat: value[0] for (stat, value) in zip(stats, values)}
	for ((i, kwd), (stat, value)) in itertools.product(enumerate(kwds), zip(stats, values)):
		output_dict["{}_{}".format(kwd, stat)] = value[i]
	output_dir = build_path(results_dir, J, observable_name, None, N, fold, seed, subdir="processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "tc.npz"), **output_dict)


### process execution times

def calculate_times(results_dir, J, observable_name, L, N_test, N=None, fold=None, seed=None, L_test=None, n_Ls=1):
	L_dir = os.path.join("data", J, "L{:d}".format(L))	
	T_dirs = sorted(os.listdir(L_dir))
	T_dirs = [os.path.join(L_dir, dir) for dir in T_dirs if dir[0] == "T"]
	with open(os.path.join(T_dirs[0], "args.json"), "r") as fp:
		args = json.load(fp)
	N_max = args["nmcs"]//args["nmeas"]
	generation_time = 0
	for dir in T_dirs:
		with open(os.path.join(dir, "time.txt"), "r") as fp:
			generation_time += float(fp.read())
	generation_time_ieq = generation_time*args["ieq"]/(args["ieq"] + args["nmcs"])
	generation_time_mc = generation_time*args["nmcs"]/(args["ieq"] + args["nmcs"])
	if observable_name == "magnetization":
		N = 0
		training_time = 0
		preprocessing_time = 0
	else:
		dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=None)
		with open(os.path.join(dir, "results.json"), "r") as fp:
			results = json.load(fp)
		training_time = results["time"]
		with open(os.path.join(L_dir, "aggregate", "times.json"), "r") as fp:
			key = "states_symmetric" if results["args"]["symmetric"] else "states"
			preprocessing_time = json.load(fp)[key]
	generation_time_mc *= (N_test + N/n_Ls)/N_max
	preprocessing_time *= (N_test + N/n_Ls)/N_max
	output_dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, L_test=L_test, subdir="processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "time.npz"), generation=generation_time_ieq+generation_time_mc, preprocessing=preprocessing_time, training=training_time)


def calculate_time_extrapolates(results_dir, J, observable_name, Ls, N=None, fold=None, seed=None):
	dicts = [dict(np.load(os.path.join(build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "time.npz"))) for L in Ls]
	output_dict = {key: sum(D[key] for D in dicts) for key in dicts[0]}
	output_dir = build_path(results_dir, J, observable_name, None, N=N, fold=fold, seed=seed, subdir="processed")
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "time.npz"), **output_dict)


### gather critical temperatures and times into CSV

def aggregate_tc(results_dir, J, observable_name, L, N=None, folds=None, seeds=None, jackknife_std=True, r2=False):
	stats = ["mean", "std", "bias"]
	key = SimpleNamespace(**{stat: stat for stat in stats})
	if r2:
		for stat in stats:
			setattr(key, stat, "r2_"+stat)
	if observable_name == "magnetization":
		with np.load(os.path.join(build_path(results_dir, J, observable_name, L, subdir="processed"), "tc.npz")) as D:
			mean, std = D[key.mean]+D[key.bias], D[key.std]*int(jackknife_std)
		return mean, std
	else:
		means, stds = [], []
		for (fold, seed) in itertools.product(folds, seeds):
			with np.load(os.path.join(build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "tc.npz")) as D:
				means.append(D[key.mean]+D[key.bias])
				stds.append(D[key.std]*int(jackknife_std))
		means = np.array(means)
		stds = np.array(stds)
		mean = means.mean()
		std = np.sqrt( (stds**2).mean() + means.std()**2 )
		return mean, std


def aggregate_time(results_dir, J, observable_name, L, N=None, folds=None, seeds=None, key="training", reduce="mean"):
	assert reduce in ["mean", "sum"], "Keyword argument reduce must be either sum or mean; got {} instead.".format(reduce)
	if observable_name == "magnetization":
		with np.load(os.path.join(build_path(results_dir, J, observable_name, L, subdir="processed"), "time.npz")) as D:
			t = D[key]
		return t
	else:
		ts = []
		for (fold, seed) in itertools.product(folds, seeds):
			with np.load(os.path.join(build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "time.npz")) as D:
				ts.append(D[key])
		ts = np.array(ts)
		t = ts.mean() if reduce == "mean" else ts.sum()
		return t


def gather_tc_time(results_dir, Js, observable_names, Ls, Ns, folds, seeds, jackknife_std=True):
	tc = 2/np.log(1+np.sqrt(2))
	output_dir = 	os.path.join(results_dir, "processed")
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "tc_time.csv"), "w") as fp:
		fp.write("J,L,observable,N,tc_mean,tc_std,generation_time,preprocessing_time,training_time\n")
		for (J, L, observable_name, N) in itertools.product(Js, Ls, observable_names, Ns):
			tc_mean, tc_std = aggregate_tc(results_dir, J, observable_name, L, N=N, folds=folds, seeds=seeds, jackknife_std=jackknife_std)
			tc_mean = 100*(tc_mean/tc - 1)
			tc_std = 100*tc_std/tc
			generation_time = aggregate_time(results_dir, J, observable_name, L, N=N, folds=folds, seeds=seeds, key="generation", reduce="mean")/60
			preprocessing_time = aggregate_time(results_dir, J, observable_name, L, N=N, folds=folds, seeds=seeds, key="preprocessing", reduce="mean")/60
			training_time = aggregate_time(results_dir, J, observable_name, L, N=N, folds=folds, seeds=seeds, key="training", reduce="sum")/60
			L_str = "{:d}".format(L) if L is not None else "inf"
			fp.write("{},{},{},{:d},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(J, L, observable_name, N, tc_mean, tc_std, generation_time, preprocessing_time, training_time))


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
	results_dir = "results4"
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	Ns = [8, 16, 32, 64, 128, 256]
	N_test = 2048

	observable_names = ["magnetization", "latent", "latent_equivariant"]
	encoder_names = ["latent", "latent_equivariant", "latent_multiscale_4"]
	n_Lss = [1, 1, 4]

	folds = list(range(8))
	seeds = list(range(3))

	print("Gathering magnetizations . . . ")
	for (J, L) in itertools.product(Js, Ls):
		gather_magnetizations("data", results_dir, J, L)

	print("Calculating stats . . . ")
	for (J, name) in itertools.product(Js, observable_names):
		calculate_stats(results_dir, J, name, Ls[-1], N_test, N=Ns[-1], fold=0, seed=0)

	print("Calculating critical temperatures . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		for L in Ls:
			calculate_critical_temperatures(results_dir, J, "magnetization", L)
		for encoder_name in encoder_names:
			print(encoder_name)
			for (L, N, fold, seed) in itertools.product(Ls, Ns, folds, seeds):
				if os.path.exists(os.path.join(build_path(results_dir, J, encoder_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "tc.npz")):
					continue
				calculate_critical_temperatures(results_dir, J, encoder_name, L, N=N, fold=fold, seed=seed)

	print("Calculating critical temperature extrapolates . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		calculate_critical_temperature_extrapolates(results_dir, J, "magnetization", Ls)
		for encoder_name in encoder_names:
			print(encoder_name)
			for (N, fold, seed) in itertools.product(Ns, folds, seeds):
				calculate_critical_temperature_extrapolates(results_dir, J, encoder_name, Ls, N=N, fold=fold, seed=seed)

	print("Calculating execution times . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		for L in Ls:
			calculate_times(results_dir, J, "magnetization", L, N_test)
		for (encoder_name, n_Ls) in zip(encoder_names, n_Lss):
			print(encoder_name)
			for (L, N, fold, seed) in itertools.product(Ls, Ns, folds, seeds):
				calculate_times(results_dir, J, encoder_name, L, N_test, N=N, fold=fold, seed=seed, n_Ls=n_Ls)

	print("Calculating execution time extrapolates . . . ")
	for J in Js:
		print("J:", J)
		print("magnetization")
		for L in Ls:
			calculate_time_extrapolates(results_dir, J, "magnetization", Ls)
		for encoder_name in encoder_names:
			print(encoder_name)
			for (N, fold, seed) in itertools.product(Ns, folds, seeds):
				calculate_time_extrapolates(results_dir, J, encoder_name, Ls, N=N, fold=fold, seed=seed)

	print("Gathering critical temperatures and times into CSV . . . ")
	gather_tc_time(results_dir, Js, ["magnetization"]+encoder_names, Ls+[None], Ns, folds, seeds, jackknife_std=False)

	print("Calculating symmetry generators . . . ")
	calculate_generators(results_dir, Js, ["latent_equivariant"], Ls, Ns, folds, seeds)

	print("Done!")
