import numpy as np

def Ising(datafile):
	with open(datafile, "r") as fp:
		data = fp.readlines()
	data = np.array([[char for char in l.strip("\n")] for l in data if len(l) > 0]).astype(np.int32)
	data = (2*data-1).astype(np.float32)
	return data
