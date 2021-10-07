import os
import json
import numpy as np
from scipy.interpolate import CubicSpline
from phasefinder import jackknife

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


def critical_temperature_table(results_dir, J, Ls, Ns, temperature_range=None):
	output_dir = os.path.join(results_dir, "tables", J)
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "tc.tex"), "w") as fp:
#	with open("tc.tex", "w") as fp:
		fp.write("\\begin{{tabular}}{{{}}}\n".format("c"*(1+2*len(Ls))))
		fp.write("\\toprule\n")
		fp.write("$L$")
		for L in Ls:
			fp.write(" & \\multicolumn{{2}}{{c}}{{{:d}}}".format(L))
		fp.write(" \\\\\n")
		fp.write("M")
		tc = 2/np.log(1+np.sqrt(2))
		for L in Ls:
			temperatures = np.load(os.path.join(results_dir, J, "magnetization", "L{:d}".format(L), "measurements.npz"))["temperatures"]
			u4samples = U4_samples(results_dir, J, "magnetization", L)
			if temperature_range is not None:
				T_within_range_indices = np.where(np.logical_and(temperatures>=temperature_range[0], temperatures<=temperature_range[1]))[0]
				T_min_index, T_max_index = T_within_range_indices.min(), T_within_range_indices.max()
				temperatures = temperatures[T_min_index:T_max_index+1]
				u4samples = u4samples[:,T_min_index:T_max_index+1]
			tc_samples = critical_temperature_samples(temperatures, u4samples)
			tc_mean, tc_std = jackknife.calculate_mean_std(tc_samples)
			fp.write(" & \\multicolumn{{2}}{{c}}{{{:.0f}}}".format(100*(tc_mean-tc)/tc))
		fp.write(" \\\\\n")
		fp.write("\\midrule\n")
		fp.write("$N$")
		for L in Ls:
			fp.write(" & AE & GE")
		fp.write(" \\\\\n")
		fp.write("\\midrule\n")
		for N in Ns:
			fp.write("{:d}".format(N))
			for L in Ls:
				for observable_name in ["latent", "latent_equivariant"]:
					temperatures = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "N{:d}".format(N), "measurements.npz"))["temperatures"]
					u4samples = U4_samples(results_dir, J, observable_name, L, N)
					if temperature_range is not None:
						T_within_range_indices = np.where(np.logical_and(temperatures>=temperature_range[0], temperatures<=temperature_range[1]))[0]
						T_min_index, T_max_index = T_within_range_indices.min(), T_within_range_indices.max()
						temperatures = temperatures[T_min_index:T_max_index+1]
						u4samples = u4samples[:,T_min_index:T_max_index+1]
					tc_samples = critical_temperature_samples(temperatures, u4samples)
					tc_mean, tc_std = jackknife.calculate_mean_std(tc_samples)
					fp.write(" & {:.0f}".format(100*(tc_mean-tc)/tc))
			fp.write(" \\\\\n")
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


def generator_table(results_dir, J, Ls, Ns, generator_type):
	output_dir = os.path.join(results_dir, "tables", J)
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "psi_{}.tex".format(generator_type)), "w") as fp:
#	with open("psi.tex", "w") as fp:
		fp.write("\\begin{{tabular}}{{{}}}\n".format("c"*(1+len(Ls))))
		fp.write("\\toprule\n")
		fp.write("\\quad")
		for L in Ls:
			fp.write(" & {:d}".format(L))
		fp.write(" \\\\\n")
		for N in Ns:
			fp.write("{:d}".format(N))
			for L in Ls:
				with open(os.path.join(results_dir, J, "latent_equivariant", "L{:d}".format(L), "N{:d}".format(N), "generator_reps.json"), "r") as json_file:
					gens = json.load(json_file)
				fp.write(" & {:.3f}".format(gens[generator_type]))
			fp.write(" \\\\\n")
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


if __name__ == "__main__":
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	Ns = [16, 32, 64, 128, 256, 512, 1024, 2048]

	for J in Js:
		critical_temperature_table("results_new", J, Ls, Ns)
		generator_table("results_new", J, Ls, Ns, "spatial")
		generator_table("results_new", J, Ls, Ns, "internal")

	print("Done!")
