import gc
import os
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")  # due to no GPU

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import phasefinder as pf

# command-line arguments
parser=argparse.ArgumentParser()
# datasets
parser.add_argument('--data_dir', '-d', required=True, type=str, help='Master directory of the data.')
parser.add_argument('--L','-l', required=True, type=int, help='Linear size of lattice for training.')
parser.add_argument('--Ls','-ls', type=lambda s: [int(i) for i in s.split(",") if i != ""], default="", help='Multiple lattice sizes for training (comma separated).')
parser.add_argument('--L_test','-lt', type=int, default=0, help='Lattice size for testing; default is set equal to --L.')
# network architecture
parser.add_argument('--encoder_hidden', '-eh', default=4, type=int, help='Hidden neurons in the encoder.')
parser.add_argument('--decoder_hidden', '-dh', default=64, type=int, help='Hidden neurons in the decoder.')
parser.add_argument('--symmetric', '-s', action="store_true", help='Enforce symmetries resulting from latent dimension.')
# SGD hyperparameters
parser.add_argument('--batch_size', '-b', default=128, type=int, help='Minibatch size.')
parser.add_argument('--epochs', '-e', default=10, type=int, help='Number of epochs for training.')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='Learning rate.')
# equivariance regularization
parser.add_argument('--equivariance_reg', '-er', default=0, type=float, help='Coefficient of equivariance-enforcing terms in loss function.')
parser.add_argument('--equivariance_pre', '-ep', default=0, type=int, help='Number of training epochs ignoring --equivariance_reg.')
# validation
parser.add_argument('--n_train_val', '-n', default=2000, type=int, help='Number of training + validation samples.')
parser.add_argument('--n_test', '-nt', default=2000, type=int, help='Number of test samples.')
parser.add_argument('--val_size','-vs', default=0.5, type=float, help='Fraction of data to use as validation set.')
parser.add_argument('--val_batch_size','-vbs', default=512, type=int, help='Minibatch size during validation/testing.')
parser.add_argument('--val_interval','-vi', default=0, type=int, help='Epoch interval at which to record validation metrics. If 0, test metrics are not recorded.')
# misc
parser.add_argument('--device', '-dv', default="cpu", type=str, help='Device.')
parser.add_argument('--load_model', '-lm', default="", type=str, help='Load pretrained encoder and decoder from given directory.')
parser.add_argument('--results_dir', '-rd', required=True, type=str, help='Master results directory.')
parser.add_argument('--observable_name', '-on', required=True, type=str, help='Name of subdirectory in results_dir.')
parser.add_argument('--exist_ok', '-ok', action="store_true", help='Allow overwriting the results_dir/observable_name/L{--L} directory.')
args=parser.parse_args()


# fix the random seed
np.random.seed(1)
torch.manual_seed(2)

# record initial time
time_start = time.time()

# create results directory
results_dir = os.path.join(args.results_dir, args.observable_name, "L{:d}".format(args.L), "N{:d}".format(args.n_train_val))
os.makedirs(results_dir, exist_ok=args.exist_ok)

# if multiple lattice sizes
n_Ls = len(args.Ls)
if n_Ls > 0:
	assert args.symmetric, "--Ls may be set only if --symmetric is also set."
	assert args.n_train_val%n_Ls == 0, "--n_train_val must be a multiple of len(--Ls)."
	assert args.n_train_val//n_Ls >= 2, "--n_train_val must be at least twice len(--Ls)."

# set lattice size for testing
if args.L_test == 0:
	assert n_Ls == 0, "If using multiple lattice sizes with --Ls, then --L_test must be set explicitly."
	args.L_test = args.L
if args.L_test != args.L:
	assert args.symmetric, "--L and --L_test can be different only if --symmetric is set."
	results_dir = os.path.join(results_dir, "L{:d}".format(args.L_test))
	os.makedirs(results_dir, exist_ok=True)

# make no training data equivalent to no training
if args.n_train_val == 0:
	args.n_train_val = 2
	args.epochs = 0

# load data
X = []
T = []
for temperature_dir in sorted(os.listdir(os.path.join(args.data_dir, "L{:d}".format(args.L)))):
	if temperature_dir[0] != "T": continue
	I = pf.datasets.Ising()
	if n_Ls > 0:
		for L in args.Ls:
			X.append( I.load_states(os.path.join(args.data_dir, "L{:d}".format(L), temperature_dir), decode=True, n_samples=args.n_train_val//n_Ls, dtype=np.float32, flatten=not args.symmetric, symmetric=args.symmetric) )
	else:
		X.append( I.load_states(os.path.join(args.data_dir, "L{:d}".format(args.L), temperature_dir), decode=True, n_samples=args.n_train_val, dtype=np.float32, flatten=not args.symmetric, symmetric=args.symmetric) )
	T.append( np.full((args.n_train_val, 1), I.T, dtype=np.float32) )
X = np.concatenate(X, 0)
T = np.concatenate(T, 0)
if n_Ls > 0:
	stratifier = np.concatenate((T, \
		np.tile(np.repeat(np.array(args.Ls), args.n_train_val//n_Ls), len(T)//args.n_train_val)[:,None] \
	), 1)
else:
	stratifier = np.copy(T)
X_train, X_val, T_train, T_val = train_test_split(X, T, stratify=stratifier, test_size=args.val_size)
del X; del T; del stratifier; gc.collect()
train_loader = DataLoader(TensorDataset(torch.as_tensor(X_train), torch.as_tensor(T_train)), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
val_loader = DataLoader(TensorDataset(torch.as_tensor(X_val), torch.as_tensor(T_val)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)

# build model
input_dim = 2 if args.symmetric else I.L**I.d
latent_dim = 1
encoder = pf.models.MLP(input_dim, args.encoder_hidden, latent_dim)
decoder = pf.models.MLP(latent_dim+1, args.decoder_hidden, input_dim)

# create trainer and callbacks
trainer = pf.trainers.Autoencoder(encoder, decoder, epochs=args.epochs, lr=args.lr, equivariance_reg=args.equivariance_reg, equivariance_pre=args.equivariance_pre, device=args.device)
callbacks = [pf.callbacks.Training()]
if args.val_interval > 0:
	callbacks.append( pf.callbacks.Validation(trainer, val_loader, epoch_interval=args.val_interval) )

# load pretrained parameters
if len(args.load_model) > 0:
	trainer.load_encoder(os.path.join(args.load_model, "encoder.pth"))
	trainer.load_decoder(os.path.join(args.load_model, "decoder.pth"))

# train model
trainer.fit(train_loader, callbacks)

# estimate symmetry generator reps
if args.equivariance_reg > 0:
	train_loader = DataLoader(train_loader.dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)
	flip_rep, neg_rep = trainer.generator_reps(train_loader)
	generator_reps = {"spatial": float(flip_rep), "internal": float(neg_rep)}
	with open(os.path.join(results_dir, "generator_reps.json"), "w") as fp:
		json.dump(generator_reps, fp, indent=2)

# generate encodings
del train_loader; del val_loader; gc.collect()
temperatures = []
measurements = []
for temperature_dir in sorted(os.listdir(os.path.join(args.data_dir, "L{:d}".format(args.L_test)))):
	if temperature_dir[0] != "T": continue
	I = pf.datasets.Ising()
	X = I.load_states(os.path.join(args.data_dir, "L{:d}".format(args.L_test), temperature_dir), decode=True, n_samples=args.n_test, dtype=np.float32, flatten=not args.symmetric, symmetric=args.symmetric)
	T = np.full((args.n_test, 1), I.T, dtype=np.float32)
	test_loader = DataLoader(TensorDataset(torch.as_tensor(X), torch.as_tensor(T)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)
	encodings = trainer.encode(test_loader)
	assert encodings.shape[1] == 1, "This script currently only supports 1D encodings."
	temperatures.append(I.T)
	measurements.append( encodings.squeeze(-1) )
temperatures = np.array(temperatures)
measurements = np.stack(measurements, 0)
np.savez(os.path.join(results_dir, "measurements.npz"), temperatures=temperatures, measurements=measurements)

# function to convert np array to list of python numbers
ndarray2list = lambda arr, dtype: [getattr(__builtins__, dtype)(x) for x in arr]

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
trainer.save_encoder(os.path.join(results_dir, "encoder.pth"))
trainer.save_decoder(os.path.join(results_dir, "decoder.pth"))


print("Done!")