import json
import numpy as np

results_dir = "results"
Js = ["ferromagnetic", "antiferromagnetic"]
Ls = [16, 32, 64, 128]
tc = 2/np.log(1+np.sqrt(2))

def tc_format(mean, std):
	if type(mean) == str:
		return "nan"
	m = 100*(mean/tc-1)
	s = 100*std/tc
	return "{:.1f} +- {:.1f}".format(m, s)

def r2_format(mean, std):
	if type(mean) == str:
		return "nan"
	return "{:.1f} +- {:.1f}".format(mean, std)

for J in Js:
	print("{}\n===\n".format(J))
	with open("{}/processed/{}/tc_biased.json".format(results_dir, J), "r") as fp:
		data = json.load(fp)
	for L in Ls:
		print("L = {:d}:".format(L))
		D = data["L{}".format(L)]
		print("M: {}".format(tc_format(D["magnetization"]["mean"], D["magnetization"]["std"])))
		print("N, AE, GE, MS")
		for i in range(len(D["Ns"])):
			print("{:d}, {}, {}, {}".format(D["Ns"][i], tc_format(D["latent"]["means"][i], D["latent"]["stds"][i]), tc_format(D["latent_equivariant"]["means"][i], D["latent_equivariant"]["stds"][i]), tc_format(D["latent_multiscale_4"]["means"][i], D["latent_multiscale_4"]["stds"][i])))
		print()

	print("L = infty")
	D = data["infty_4"]
	print("M: {}".format(tc_format(D["magnetization"]["mean"], D["magnetization"]["std"])))
	print("N, AE, GE, MS")
	for i in range(len(D["Ns"])):
		print("{:d}, {}, {}, {}".format(D["Ns"][i], tc_format(D["latent"]["means"][i], D["latent"]["stds"][i]), tc_format(D["latent_equivariant"]["means"][i], D["latent_equivariant"]["stds"][i]), tc_format(D["latent_multiscale_4"]["means"][i], D["latent_multiscale_4"]["stds"][i])))
	print()

	print("L = infty (r^2 score)")
	D = data["r2_4"]
	print("M: {}".format(r2_format(D["magnetization"]["mean"], D["magnetization"]["std"])))
	print("N, AE, GE, MS")
	for i in range(len(D["Ns"])):
		print("{:d}, {}, {}, {}".format(D["Ns"][i], r2_format(D["latent"]["means"][i], D["latent"]["stds"][i]), r2_format(D["latent_equivariant"]["means"][i], D["latent_equivariant"]["stds"][i]), r2_format(D["latent_multiscale_4"]["means"][i], D["latent_multiscale_4"]["stds"][i])))
	print()
