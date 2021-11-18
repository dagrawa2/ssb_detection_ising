import os
import itertools
import numpy as np
from types import SimpleNamespace

results_dir = "results4"
Js = ["ferromagnetic"]
Ls = [16, 32, 64, 128, None]
Ns = [8, 16, 32, 64, 128, 256]
folds = [0, 1, 2, 3]
seeds = [0, 1, 2]

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

def gather_tc(results_dir, J, observable_name, L, N=None, folds=None, seeds=None, r2=False):
	stats = ["mean", "std", "bias"]
	key = SimpleNamespace(**{stat: stat for stat in stats})
	if r2:
		for stat in stats:
			setattr(key, stat, "r2_"+stat)
	if observable_name == "magnetization":
		with np.load(os.path.join(build_path(results_dir, J, observable_name, L, subdir="processed"), "tc.npz")) as D:
			mean, std = D[key.mean]+D[key.bias], D[key.std]
		return mean, std
	else:
		means, stds = [], []
		for (fold, seed) in itertools.product(folds, seeds):
			with np.load(os.path.join(build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed"), "tc.npz")) as D:
				means.append(D[key.mean]+D[key.bias])
				stds.append(D[key.std])
		means = np.array(means)
		stds = np.array(stds)
		mean = means.mean()
		std = np.sqrt( (stds**2).mean() + means.std()**2 )
		return mean, std


for J in Js:
	print("{}\n===\n".format(J))
	for L in Ls:
		print("L = {}:".format(L))
		mean, std = gather_tc(results_dir, J, "magnetization", L)
		print("M: {}".format(tc_format(mean, std)))
		print("N, AE, GE, MS")
		for N in Ns:
			AE_mean, AE_std = gather_tc(results_dir, J, "latent", L, N=N, folds=folds, seeds=seeds)
			GE_mean, GE_std = gather_tc(results_dir, J, "latent_equivariant", L, N=N, folds=folds, seeds=seeds)
			MS_mean, MS_std = gather_tc(results_dir, J, "latent_multiscale_4", L, N=N, folds=folds, seeds=seeds)
			print("{:d}, {}, {}, {}".format(N, tc_format(AE_mean, AE_std), tc_format(GE_mean, GE_std), tc_format(MS_mean, MS_std)))
		print()

	print("L = None (r^2 value)")
	mean, std = gather_tc(results_dir, J, "magnetization", None, r2=True)
	print("M: {}".format(tc_format(mean, std)))
	print("N, AE, GE, MS")
	for N in Ns:
		AE_mean, AE_std = gather_tc(results_dir, J, "latent", None, N=N, folds=folds, seeds=seeds, r2=True)
		GE_mean, GE_std = gather_tc(results_dir, J, "latent_equivariant", None, N=N, folds=folds, seeds=seeds, r2=True)
		MS_mean, MS_std = gather_tc(results_dir, J, "latent_multiscale_4", None, N=N, folds=folds, seeds=seeds, r2=True)
		print("{:d}, {}, {}, {}".format(N, r2_format(AE_mean, AE_std), r2_format(GE_mean, GE_std), r2_format(MS_mean, MS_std)))
	print()
