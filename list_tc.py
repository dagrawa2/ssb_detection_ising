import json
import numpy as np

Js = ["ferromagnetic", "antiferromagnetic"]
Ls = [16, 32, 64, 128]
tc = 2/np.log(1+np.sqrt(2))

for J in Js:
	print("{}\n===\n".format(J))
	with open("results/processed/{}/tc_biased.json".format(J), "r") as fp:
		data = json.load(fp)
	for L in Ls:
		print("L = {:d}:".format(L))
		D = data["L{}".format(L)]
		values = np.array( sorted(D["AE"]["means"] + D["GE"]["means"]) )
		values = np.round(100*(values/tc-1), 1)
		for v in values:
			print(v)
		print()
