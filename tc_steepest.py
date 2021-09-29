import os
import numpy as np
from scipy.interpolate import CubicSpline
from phasefinder import jackknife

def U4_samples(results_dir, J, observable_name, L):
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


def critical_temperature_table(results_dir, J, observable_names, L, row_headings_dict, temperature_range=None):
#	output_dir = os.path.join(results_dir, "tables", J)
#	os.makedirs(output_dir, exist_ok=True)
#	with open(os.path.join(output_dir, "critical_temperatures.tex"), "w") as fp:
	with open("critical_temperatures.tex", "w") as fp:
		fp.write("\\begin{tabular}{cc}\n")
		fp.write("\\toprule\n")
		fp.write("Method & $T_c$ \\\\\n")
		fp.write("\\midrule\n")
		for observable_name in observable_names:
			temperatures = np.load(os.path.join(results_dir, J, observable_name, "L{:d}".format(L), "measurements.npz"))["temperatures"]
			u4samples = U4_samples(results_dir, J, observable_name, L)
			if temperature_range is not None:
				T_within_range_indices = np.where(np.logical_and(temperatures>=temperature_range[0], temperatures<=temperature_range[1]))[0]
				T_min_index, T_max_index = T_within_range_indices.min(), T_within_range_indices.max()
				temperatures = temperatures[T_min_index:T_max_index+1]
				u4samples = u4samples[:,T_min_index:T_max_index+1]
			tc_samples = critical_temperature_samples(temperatures, u4samples)
			tc_mean, tc_std = jackknife.calculate_mean_std(tc_samples)
			fp.write("{} & ${:.3f}\\pm {:.3f}$ \\\\\n".format(row_headings_dict[observable_name], tc_mean, tc_std))
		fp.write("\\bottomrule\n")
		fp.write("\\end{tabular}")


if __name__ == "__main__":
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	observable_names = ["magnetization", "latent", "latent_equivariant"]
	row_headings_dict = {"magnetization": "Mag", "latent": "AE", "latent_equivariant": "GE-AE"}

	J = "ferromagnetic"
	L = 64
	critical_temperature_table("results", J, observable_names, L, row_headings_dict)  # , temperature_range=(1.04, 3.5))


	print("Done!")
