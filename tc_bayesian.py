import numpy as np

def curve_zeros(x, y):
	zeros = []
	x_1, y_1 = x[0], y[0]
	for (x_2, y_2) in zip(x[1:], y[1:]):
		if np.sign(y_1) != np.sign(y_2):
			zeros.append( x_1 - y_1*(x_2-x_1)/(y_2-y_1) )
		x_1, y_1 = x_2, y_2
	return np.array(zeros)


observable_name = "latent"
t = np.load("results/ferromagnetic/"+observable_name+"/L16/measurements.npz")["temperatures"]

Ls = [16, 32, 64, 128]
p = np.ones_like(t)
for (L_1, L_2) in zip(Ls[:-1], Ls[1:]):
	m = np.load("results/ferromagnetic/"+observable_name+"/L"+str(L_1)+"/stats.npz")["binder_means"] \
	- np.load("results/ferromagnetic/"+observable_name+"/L"+str(L_2)+"/stats.npz")["binder_means"]
	s = np.sqrt( np.load("results/ferromagnetic/"+observable_name+"/L"+str(L_1)+"/stats.npz")["binder_stds"]**2 \
	+ np.load("results/ferromagnetic/"+observable_name+"/L"+str(L_2)+"/stats.npz")["binder_stds"]**2 )
	p = p * 1/(np.sqrt(2*np.pi)*s)*np.exp(-0.5*(m/s)**2)

p = p/np.sum(p)
print(p.max())

"""
p[p<1e-20] = 0
for (x, y) in zip(t, p):
	print("{}: {}".format(x, y))
"""
