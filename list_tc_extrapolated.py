import json
import numpy as np

J = "ferromagnetic"
biased = True

biased = "_biased" if biased else ""
with open("results/processed/{}/tc{}.json".format(J, biased), "r") as fp:
	data = json.load(fp)

tc_exact = 2/np.log(1+np.sqrt(2))
Ns = [0] + [2**n for n in range(1, 12)]

print("magnetization\n===")
print("T_c: {:.0f} +- {:.0f}".format(100*(data["fit"]["M"]["mean"][0]-tc_exact)/tc_exact, 100*data["fit"]["M"]["std"][0]/tc_exact))
print("r^2: {:.3f} +- {:.3f}".format(data["r2"]["M"]["mean"], data["r2"]["M"]["std"]))
print()

for (model, observable_name) in [("AE", "latent"), ("GE", "latent_equivariant")]:
	print(observable_name+"\n===")
	print("N, T_c, r^2")
	for (i, N) in enumerate(Ns):
		print("{:d}, {:.0f} +- {:.0f}, {:.3f} +- {:.3f}".format(N, 100*(data["fit"][model]["means"][i][0]-tc_exact)/tc_exact, 100*data["fit"][model]["stds"][i][0]/tc_exact, data["r2"][model]["means"][i], data["r2"][model]["stds"][i]))
	print()
