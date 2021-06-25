import numpy as np

def read(filename, unique=False):
	with open(filename, "r") as fp:
		lines = fp.readlines()
	lines = np.array([l.strip("\n") for l in lines if len(l) > 0])
	if unique:
		lines = np.unique(lines)
	return lines

def str2array(strings):
	A = np.array([[int(c) for c in s] for s in strings])
	l = int(np.sqrt(A.shape[1]))
	A = A.reshape((A.shape[0], l, l)).astype(np.int32)
	return 2*A-1

def array2str(A):
	A = (A+1)/2
	A = A.reshape(A.shape[0], A.shape[1]**2).astype(np.int32).astype(str)
	strings = ["".join(a) for a in A]
	return strings

def transform(A, gen):
	if gen == 1:
		return np.roll(A, 1, axis=0)
	if gen == 2:
		return np.roll(A, 1, axis=1)
	if gen == 3:
		return -A

def is_symmetry(A, gen):
	l = A.shape[1]
	A = A.reshape((A.shape[0], l**2))
	result = False
	for S in A:
		gS = transform(S.reshape((l, l)), gen=gen).reshape((l**2))
		hamming_dists = (l**2 - A.dot(gS))/2
		if min(hamming_dists) < 0.2:
#		if s in strings:
			result = True
			break
	return result

def magnetization(A):
	return np.sum(A)/(A.shape[0]*A.shape[1]**2)


for T in range(1, 5):
	print("T =", T, ":")
	A = str2array( read("results/T"+str(T)+"/states.txt", unique=True) )
	print("M =", np.format_float_scientific(magnetization(A), unique=True))
	for gen in range(1, 4):
		result = is_symmetry(A, gen)
		if result:
			print("generator", gen)

print("Done!")
