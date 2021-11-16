import os
import itertools
import numpy as np

results_dir = "results8"
Js = ["ferromagnetic", "antiferromagnetic"]
Ls = [16, 32, 64, 128, "infty"]
Ns = [8, 16, 32, 64, 128, 256]
N_tests = [256, 512, 1024, 2048]
folds = [0, 1, 2, 3]
seeds = [0, 1, 2]
biased = True

tc = 2/np.log(1+np.sqrt(2))

def tc_format(mean, std):
	if mean.dtype not in [np.float32, np.float64]:
		return "nan"
	m = 100*(mean/tc-1)
	s = 100*std/tc
	return "{:.1f} +- {:.1f}".format(m, s)

def r2_format(mean, std):
	if mean.dtype not in [np.float32, np.float64]:
		return "nan"
	return "{:.1f} +- {:.1f}".format(mean, std)

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

def gather_tc(results_dir, J, observable_name, L, N=None, folds=None, seeds=None, biased=False, key="yintercept"):
	if observable_name == "latent_multiscale_4" and N < 8:
		nan = np.full((len(N_tests)), "nan")
		return nan, nan
	if L is None or L == "infty":
		return gather_tc_extrapolate(results_dir, J, observable_name, N=N, folds=folds, seeds=seeds, biased=biased, key=key)
	if observable_name == "magnetization":
		dir = build_path(results_dir, J, observable_name, L, subdir="processed")
		with np.load(os.path.join(dir, "tc.npz")) as D:
			means, stds = D["means"], D["stds"]
			if biased:
				means = means + D["biases"]
		return means, stds
	else:
		means, stds = [], []
		for (fold, seed) in itertools.product(folds, seeds):
			dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed")
			with np.load(os.path.join(dir, "tc.npz")) as D:
				means.append(D["means"])
				stds.append(D["stds"])
				if biased:
					means[-1] = D["means"] + D["biases"]
		means = np.stack(means, 0)
		stds = np.stack(stds, 0)
		m = means.mean(0)
		s = np.sqrt( (stds**2).mean(0) + means.std(0)**2 )
		return m, s


def gather_tc_extrapolate(results_dir, J, observable_name, N=None, folds=None, seeds=None, biased=False, key="yintercept"):
	if observable_name == "magnetization":
		dir = build_path(results_dir, J, observable_name, None, subdir="processed")
		with np.load(os.path.join(dir, "tc_extrapolate.npz")) as D:
			means, stds = D[key+"_means"][:,-1], D[key+"_stds"][:,-1]
			if biased:
				means = means + D[key+"_biases"][:,-1]
		return means, stds
	else:
		means, stds = [], []
		for (fold, seed) in itertools.product(folds, seeds):
			dir = build_path(results_dir, J, observable_name, None, N=N, fold=fold, seed=seed, subdir="processed")
			with np.load(os.path.join(dir, "tc_extrapolate.npz")) as D:
				means.append(D[key+"_means"][:,-1])
				stds.append(D[key+"_stds"][:,-1])
				if biased:
					means[-1] = D[key+"_means"][:,-1] + D[key+"_biases"][:,-1]
		means = np.stack(means, 0)
		stds = np.stack(stds, 0)
		m = means.mean(0)
		s = np.sqrt( (stds**2).mean(0) + means.std(0)**2 )
		return m, s


for J in Js:
	print("{}\n===\n".format(J))
	for L in Ls:
		print("L = {}:".format(L))
		means, stds = gather_tc(results_dir, J, "magnetization", L, biased=biased)
		print("M: ", end="")
		for (m, s) in zip(means, stds):
			print("{}, ".format(tc_format(m, s)), end="")
		print()
		print("N, N_test, AE, GE, MS")
		for N in Ns:
			AE_means, AE_stds = gather_tc(results_dir, J, "latent", L, N=N, folds=folds, seeds=seeds, biased=biased)
			GE_means, GE_stds = gather_tc(results_dir, J, "latent_equivariant", L, N=N, folds=folds, seeds=seeds, biased=biased)
			MS_means, MS_stds = gather_tc(results_dir, J, "latent_multiscale_4", L, N=N, folds=folds, seeds=seeds, biased=biased)
			for (N_test, AE_m, AE_s, GE_m, GE_s, MS_m, MS_s) in zip(N_tests, AE_means, AE_stds, GE_means, GE_stds, MS_means, MS_stds):
				print("{:d}, {:d}, {}, {}, {}".format(N, N_test, tc_format(AE_m, AE_s), tc_format(GE_m, GE_s), tc_format(MS_m, MS_s)))
		print()

	print("L = infty (r^2 value)")
	means, stds = gather_tc(results_dir, J, "magnetization", "infty", biased=biased, key="r2")
	print("M: ", end="")
	for (m, s) in zip(means, stds):
		print("{}, ".format(r2_format(m, s)), end="")
	print()
	print("N, N_test, AE, GE, MS")
	for N in Ns:
		AE_means, AE_stds = gather_tc(results_dir, J, "latent", "infty", N=N, folds=folds, seeds=seeds, biased=biased, key="r2")
		GE_means, GE_stds = gather_tc(results_dir, J, "latent_equivariant", "infty", N=N, folds=folds, seeds=seeds, biased=biased, key="r2")
		MS_means, MS_stds = gather_tc(results_dir, J, "latent_multiscale_4", "infty", N=N, folds=folds, seeds=seeds, biased=biased, key="r2")
		for (N_test, AE_m, AE_s, GE_m, GE_s, MS_m, MS_s) in zip(N_tests, AE_means, AE_stds, GE_means, GE_stds, MS_means, MS_stds):
			print("{:d}, {:d}, {}, {}, {}".format(N, N_test, r2_format(AE_m, AE_s), r2_format(GE_m, GE_s), r2_format(MS_m, MS_s)))
	print()
