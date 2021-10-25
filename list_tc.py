import json
import numpy as np

Js = ["ferromagnetic", "antiferromagnetic"]
Ls = [16, 32, 64, 128]
tc = 2/np.log(1+np.sqrt(2))

def tc_format(mean, std):
	m = 100*(mean/tc-1)
	s = 100*std/tc
	return "{:.1f} +- {:.1f}".format(m, s)

for J in Js:
	print("{}\n===\n".format(J))
	with open("results/processed/{}/tc_biased.json".format(J), "r") as fp:
		data = json.load(fp)
	for L in Ls:
		print("L = {:d}:".format(L))
		D = data["L{}".format(L)]
		print("M: {}".format(tc_format(D["magnetization"]["mean"], D["magnetization"]["std"])))
		print("N, AE, GE, MS")
		for i in range(len(D["Ns"])):
			print("{:d}, {}, {}, {}".format(D["Ns"][i], tc_format(D["latent"]["means"][i], D["latent"]["stds"][i]), tc_format(D["latent_equivariant"]["means"][i], D["latent_equivariant"]["stds"][i]), tc_format(D["latent_multiscale"]["means"][i], D["latent_multiscale"]["stds"][i])))
		print()
