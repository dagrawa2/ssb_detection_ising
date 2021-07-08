import numpy as np
from sklearn.model_selection import train_test_split

def Ising(datafile, val_size=0.2):
	with open(datafile, "r") as fp:
		data = fp.readlines()
	data = np.array([[char for char in l.strip("\n")] for l in data if len(l) > 0]).astype(np.int32)
	dim = int(np.sqrt(data.shape[-1]))
	data = (2*data-1).astype(np.float32).reshape((-1, 1, dim, dim))
	data_train, data_val = train_test_split(data, test_size=val_size)
	return data_train, data_val
