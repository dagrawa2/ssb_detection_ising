import os
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")  # due to no GPU

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import phasefinder as pf

# command-line arguments
parser=argparse.ArgumentParser()
# datasets
parser.add_argument('--data', '-d', required=True, type=str, help='Data file.')
# optimization hyperparameters
parser.add_argument('--epochs','-e', default=100, type=int, help='Number of epochs for training.')
# covariance settings
parser.add_argument('--full_cov', '-fc', action="store_true", help='Use full covariance matrix instead of scalar variance.')
parser.add_argument('--jitter','-j', default=1e-6, type=float, help='Jitter added to variance.')
# misc
parser.add_argument('--device', '-dv', default="cpu", type=str, help='Device.')
parser.add_argument('--results_dir', '-rd', default="default", type=str, help='Results directory.')
args=parser.parse_args()


# fix the random seed
np.random.seed(1)
torch.manual_seed(2)

# record initial time
time_start = time.time()

# load data
X = torch.as_tensor(np.load(args.data), dtype=torch.double)

# build model
group = pf.models.Group(2, [[(0, 1)]])
model = pf.models.GMM(2, group, full_cov=args.full_cov, jitter=args.jitter)
model.initialize(X)

# create model trainer and callbacks
trainer = pf.trainers.GMM(model, epochs=args.epochs, device=args.device)
callbacks = [pf.callbacks.Training()]

# train model
trainer.fit(callbacks)

# get trained parameters
mean, variance, variance_eigs = trainer.get_params()
print("mean:", np.round(mean, 3))
print("variance_eigs:", np.round(variance_eigs, 3))

# function to convert np array to list of python numbers
ndarray2list = lambda arr, dtype: [getattr(__builtins__, dtype)(x) for x in arr]

# create results directory
if args.results_dir == "default":
#	results_dir = os.path.join("./results", args.data, args.model)
	results_dir = os.path.join("./results", "default")
else:
	results_dir = args.results_dir
os.makedirs(results_dir, exist_ok=True)

# collect results
results_dict = {
	"data_shapes": {name: list(A.shape) for (name, A) in [("X", X)]}, 
	"train": {key: ndarray2list(value, "float") for cb in callbacks for (key, value) in cb.history.items()}, 
	"params": {
		"mean": ndarray2list(mean, "float"), 
#		"variance": float(variance)
		"variance_eigs": ndarray2list(variance_eigs, "float")
	}
}

# add command-line arguments and script execution time to results
results_dict["args"] = dict(vars(args))
results_dict["time"] = time.time()-time_start

# save results
print("Saving results ...")
with open(os.path.join(results_dir, "results.json"), "w") as fp:
	json.dump(results_dict, fp, indent=2)
#trainer.save_model(os.path.join(results_dir, "params.pth"))
np.savez(os.path.join(results_dir, "params.npz"), mean=mean, variance=variance)

print("Done!")