import os
import itertools
import numpy as np
from types import SimpleNamespace
from phasefinder.utils import build_path

results_dir = "results4"
Js = ["ferromagnetic", "antiferromagnetic"]
Ls = [16, 32, 64, 128, None]
Ns = [8, 16, 32, 64, 128, 256]
folds = [0, 1, 2, 3]
seeds = [0, 1, 2]
jackknife_std = False

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

def gather_tc(results_dir, J, observable_name, L, N=None, folds=None, seeds=None, jackknife_std=True, r2=False):
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


for J in Js:
	print("{}\n===\n".format(J))
	for L in Ls:
		print("L = {}:".format(L))
		mean, std = gather_tc(results_dir, J, "magnetization", L, jackknife_std=jackknife_std)
		print("M: {}".format(tc_format(mean, std)))
		print("N, AE, GE, MS")
		for N in Ns:
			AE_mean, AE_std = gather_tc(results_dir, J, "latent", L, N=N, folds=folds, seeds=seeds, jackknife_std=jackknife_std)
			GE_mean, GE_std = gather_tc(results_dir, J, "latent_equivariant", L, N=N, folds=folds, seeds=seeds, jackknife_std=jackknife_std)
			MS_mean, MS_std = gather_tc(results_dir, J, "latent_multiscale_4", L, N=N, folds=folds, seeds=seeds, jackknife_std=jackknife_std)
			print("{:d}, {}, {}, {}".format(N, tc_format(AE_mean, AE_std), tc_format(GE_mean, GE_std), tc_format(MS_mean, MS_std)))
		print()

	print("L = None (r^2 value)")
	mean, std = gather_tc(results_dir, J, "magnetization", None, r2=True, jackknife_std=jackknife_std)
	print("M: {}".format(tc_format(mean, std)))
	print("N, AE, GE, MS")
	for N in Ns:
		AE_mean, AE_std = gather_tc(results_dir, J, "latent", None, N=N, folds=folds, seeds=seeds, r2=True, jackknife_std=jackknife_std)
		GE_mean, GE_std = gather_tc(results_dir, J, "latent_equivariant", None, N=N, folds=folds, seeds=seeds, r2=True, jackknife_std=jackknife_std)
		MS_mean, MS_std = gather_tc(results_dir, J, "latent_multiscale_4", None, N=N, folds=folds, seeds=seeds, r2=True, jackknife_std=jackknife_std)
		print("{:d}, {}, {}, {}".format(N, r2_format(AE_mean, AE_std), r2_format(GE_mean, GE_std), r2_format(MS_mean, MS_std)))
	print()
