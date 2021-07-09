import os
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")  # due to no GPU

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import phasefinder2 as pf

# command-line arguments
parser=argparse.ArgumentParser()
# datasets
parser.add_argument('--data_dir', '-d', required=True, type=str, help='Directory containing data files.')
# SGD hyperparameters
parser.add_argument('--batch_size','-b', default=128, type=int, help='Minibatch size.')
parser.add_argument('--epochs','-e', default=100, type=int, help='Number of epochs for training.')
parser.add_argument('--lr','-lr', default=1e-3, type=float, help='Learning rate.')
# validation
parser.add_argument('--val_size','-vs', default=0.2, type=float, help='Fraction of data to use as validation set.')
parser.add_argument('--val_batch_size','-vbs', default=512, type=int, help='Minibatch size during validation/testing.')
parser.add_argument('--val_interval','-vi', default=0, type=int, help='Epoch interval at which to record validation metrics. If 0, test metrics are not recorded.')
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
data_files = []
train_loaders = []
val_loaders = []
for data_file in os.listdir(args.data_dir):
	data_file_path = os.path.join(args.data_dir, data_file)
	if os.path.isdir(data_file_path) \
	or os.path.splitext(data_file_path)[1] in [".c", ".py"]:
		continue
	data_files.append(data_file)
	X_train, X_val = pf.datasets.Ising(data_file_path, val_size=args.val_size, symmetrize=True)
	train_loaders.append( DataLoader(TensorDataset(torch.as_tensor(X_train)), batch_size=args.batch_size, shuffle=True, drop_last=True) )  # , num_workers=8)
	val_loaders.append( DataLoader(TensorDataset(torch.as_tensor(X_val)), batch_size=args.val_batch_size, shuffle=False, drop_last=False) )  # , num_workers=8)

# build model
encoder = pf.models.Encoder((X_train.shape[-1], X_train.shape[-1]), 3, 1)
decoders = []
for _ in range(len(data_files)):
	decoders.append( pf.models.Decoder(2, 64, (X_train.shape[-1], X_train.shape[-1])) )

# create trainer and callbacks
trainer = pf.trainers.Autoencoder(encoder, decoders, epochs=args.epochs, lr=args.lr, device=args.device)
callbacks = [pf.callbacks.Training()]
if args.val_interval > 0:
	callbacks.append( pf.callbacks.Validation(trainer, val_loaders, epoch_interval=args.val_interval) )

# train model
trainer.fit(train_loaders, callbacks)

# function to convert np array to list of python numbers
ndarray2list = lambda arr, dtype: [getattr(__builtins__, dtype)(x) for x in arr]

# create results directory
if args.results_dir == "default":
#	results_dir = os.path.join("./results", args.data, args.model)
	results_dir = os.path.join("./results", "default")
else:
	results_dir = args.results_dir
os.makedirs(results_dir, exist_ok=True)
encodings_dir = os.path.join(results_dir, "encodings")
os.makedirs(encodings_dir, exist_ok=True)

# generate encodings
for (data_file, train_loader, val_loader) in zip(data_files, train_loaders, val_loaders):
	data_file_no_ext, _ = os.path.splitext(data_file)
	encodings = np.concatenate((trainer.encode(train_loader), trainer.encode(val_loader)), 0)
	np.save(os.path.join(encodings_dir, data_file_no_ext+".npy"), encodings)

# collect results
results_dict = {
#	"data_shapes": {name: list(A.shape) for (name, A) in [("X_1", X_1), ("X_2", X_2)]}, 
	"train": {key: ndarray2list(value, "float") for cb in callbacks for (key, value) in cb.history.items()}, 
}

# add command-line arguments and script execution time to results
results_dict["args"] = dict(vars(args))
results_dict["time"] = time.time()-time_start

# save results
print("Saving results ...")
with open(os.path.join(results_dir, "results.json"), "w") as fp:
	json.dump(results_dict, fp, indent=2)
#trainer.save_model(os.path.join(results_dir, "params.pth"))


print("Done!")