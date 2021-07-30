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
parser.add_argument('--L','-l', required=True, type=int, help='Linear size of lattice.')
# network architecture
parser.add_argument('--encoder_hidden', '-eh', default=4, type=int, help='Hidden neurons in the encoder.')
parser.add_argument('--decoder_hidden', '-dh', default=64, type=int, help='Hidden neurons in the decoder.')
parser.add_argument('--symmetric', '-s', action="store_true", help='Enforce symmetries resulting from latent dimension.')
# SGD hyperparameters
parser.add_argument('--batch_size', '-b', default=128, type=int, help='Minibatch size.')
parser.add_argument('--epochs', '-e', default=100, type=int, help='Number of epochs for training.')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='Learning rate.')
# validation
parser.add_argument('--n_train_val', '-n', default=2000, type=int, help='Number of training + validation samples.')
parser.add_argument('--n_test', '-nt', default=2000, type=int, help='Number of test samples.')
parser.add_argument('--val_size','-vs', default=0.5, type=float, help='Fraction of data to use as validation set.')
parser.add_argument('--val_batch_size','-vbs', default=512, type=int, help='Minibatch size during validation/testing.')
parser.add_argument('--val_interval','-vi', default=0, type=int, help='Epoch interval at which to record validation metrics. If 0, test metrics are not recorded.')
# misc
parser.add_argument('--device', '-dv', default="cpu", type=str, help='Device.')
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
results_dir = os.path.join(args.results_dir, args.observable_name, "L{:d}".format(args.L))
os.makedirs(results_dir, exist_ok=args.exist_ok)

# load data
X = []
T = []
for temperature_dir in sorted(os.listdir(os.path.join(args.data_dir, "L{:d}".format(args.L)))):
	I = pf.datasets.Ising()
	X.append( I.load_states(os.path.join(args.data_dir, "L{:d}".format(args.L), temperature_dir), decode=True, n_samples=args.n_train_val, dtype=np.float32, flatten=not args.symmetric, channel_dim=args.symmetric) )
	T.append( np.full((args.n_train_val, 1), I.T, dtype=np.float32) )
X = np.concatenate(X, 0)
T = np.concatenate(T, 0)
X_train, X_val, T_train, T_val = train_test_split(X, T, stratify=T, test_size=args.val_size)
del X; del T; gc.collect()
train_loader = DataLoader(TensorDataset(torch.as_tensor(X_train), torch.as_tensor(T_train)), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
val_loader = DataLoader(TensorDataset(torch.as_tensor(X_val), torch.as_tensor(T_val)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)

# build model
input_dim = [I.L]*I.d if args.symmetric else I.L**I.d
latent_dim = 1
encoder = pf.models.Encoder(input_dim, args.encoder_hidden, latent_dim, symmetric=args.symmetric)
decoder = pf.models.Decoder(latent_dim, args.decoder_hidden, input_dim, symmetric=args.symmetric)

# create trainer and callbacks
trainer = pf.trainers.Autoencoder(encoder, decoder, epochs=args.epochs, lr=args.lr, device=args.device)
callbacks = [pf.callbacks.Training()]
if args.val_interval > 0:
	callbacks.append( pf.callbacks.Validation(trainer, val_loader, epoch_interval=args.val_interval) )

# train model
trainer.fit(train_loader, callbacks)

# generate encodings
del train_loader; del val_loader; gc.collect()
temperatures = []
measurements = []
for temperature_dir in sorted(os.listdir(os.path.join(args.data_dir, "L{:d}".format(args.L)))):
	I = pf.datasets.Ising()
	X = I.load_states(os.path.join(args.data_dir, "L{:d}".format(args.L), temperature_dir), decode=True, n_samples=args.n_test, dtype=np.float32, flatten=not args.symmetric, channel_dim=args.symmetric)
	T = np.full((args.n_test, 1), I.T, dtype=np.float32)
	test_loader = DataLoader(TensorDataset(torch.as_tensor(X), torch.as_tensor(T)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)
	encodings = trainer.encode(test_loader)
	assert encodings.shape[1] == 1, "This script currently only supports 1D encodings."
	temperatures.append(I.T)
	measurements.append( encodings.squeeze(-1) )
temperatures = np.array(temperatures)
measurements = np.stack(measurements, 0)
np.savez(os.path.join(results_dir, "measurements.npz"), temperatures=temperatures, measurements=measurements)

# generate encodings of symmetry-transformed inputs
if args.symmetric:
	del test_loader; gc.collect()
	G = list( pf.groups.generate_group() )
	group_elements = np.array([g.value for g in G])
	symmetry_scores = []
	for j, temperature_dir in enumerate(sorted(os.listdir(os.path.join(args.data_dir, "L{:d}".format(args.L))))):
		I = pf.datasets.Ising()
		X = I.load_states(os.path.join(args.data_dir, "L{:d}".format(args.L), temperature_dir), decode=True, n_samples=args.n_test, dtype=np.float32, flatten=not args.symmetric, channel_dim=args.symmetric)
		T = np.full((args.n_test, 1), I.T, dtype=np.float32)
		mmds = []
		for g in G:
			test_loader = DataLoader(TensorDataset(torch.as_tensor(g.action(X).copy()), torch.as_tensor(T)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)
			encodings_transformed = trainer.encode(test_loader).squeeze(-1)
			mmds.append( np.mean((measurements[j]-encodings_transformed)**2) )
		symmetry_scores.append(mmds)
	symmetry_scores = np.stack(symmetry_scores, 0)
	np.savez(os.path.join(results_dir, "symmetry_scores.npz"), temperatures=temperatures, group_elements=group_elements, symmetry_scores=symmetry_scores)

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
#trainer.save_model(os.path.join(results_dir, "params.pth"))


print("Done!")