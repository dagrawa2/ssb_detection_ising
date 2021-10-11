import numpy as np

def calculate_samples(data):
	M = data.shape[0]
	sums = data.sum(0)
	estimates = sums/M
	samples = (sums[None,:] - data)/(M-1)
	samples = np.concatenate((estimates[None,:], samples), 0)
	return samples

def calculate_mean_std(samples, remove_bias=True):
	D = samples.ndim
	if D == 1:
		samples = samples[:,None]
	estimates, samples = samples[0], samples[1:]
	std = np.sqrt((samples.shape[0]-1)*np.mean((samples - estimates[None,:])**2, 0))
	bias = (samples.shape[0]-1)*(samples.mean(0) - estimates) if remove_bias else 0
	mean = estimates - bias
	if D == 1:
		mean, std = mean.item(), std.item()
	return mean, std
