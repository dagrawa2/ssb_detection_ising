import numpy as np

np.random.seed(0)

def generate(path, size=None, p=0.5):
	X_1 = np.random.choice([0, 1], size=size, p=[1-p, p]).astype(str)
	X_2 = np.random.choice([0, 1], size=size, p=[p, 1-p]).astype(str)
	X = np.concatenate((X_1, X_2), axis=0)
	np.random.shuffle(X)
	X = X[:X.shape[0]//2]
	with open(path, "w") as fp:
		for x in X:
			fp.write("".join(x)+"\n")


generate("T1.txt", size=(256, 16), p=0.5)
generate("T1.txt", size=(256, 16), p=0.5)
generate("T2.txt", size=(256, 16), p=0.9)
