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
parser.add_argument('--data_1', '-d1', required=True, type=str, help='Data file at higher temperature.')
parser.add_argument('--data_2', '-d2', required=True, type=str, help='Data file at lower temperature.')
# SGD hyperparameters
parser.add_argument('--batch_size','-b', default=128, type=int, help='Minibatch size.')
parser.add_argument('--epochs','-e', default=100, type=int, help='Number of epochs for training.')
parser.add_argument('--lr','-lr', default=1e-3, type=float, help='Learning rate.')
# regularization
parser.add_argument('--reg_weight','-rw', default=0.1, type=float, help='Coefficient for broken symmetry MRA regularizer.')
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
X_1 = pf.datasets.Ising(args.data_1)
X_2 = pf.datasets.Ising(args.data_2)

# wrap as data loaders
train_loader_1 = DataLoader(TensorDataset(torch.as_tensor(X_1)), batch_size=args.batch_size, shuffle=True, drop_last=True)  # , num_workers=8)
train_loader_2 = DataLoader(TensorDataset(torch.as_tensor(X_2)), batch_size=args.batch_size, shuffle=True, drop_last=True)  # , num_workers=8)

# build model
group = pf.models.Group(2, [(0, 1)])
observable = pf.models.IsingObservable( int(np.sqrt(X_1.shape[1])) )
model = pf.models.PFNet(observable, group, regularizer_weight=args.reg_weight)
callbacks = [pf.callbacks.Training()]

# create model trainer
trainer = pf.Trainer(model, epochs=args.epochs, lr=args.lr, device=args.device)

# train model
#trainer.init_MRA(train_loader_1, train_loader_2, iters=5)
#trainer.fit(train_loader_1, train_loader_2, callbacks)
# testing pretraining here
trainer.epochs = 20
trainer.fit(train_loader_1, train_loader_2, [])
trainer.epochs = args.epochs - trainer.epochs
trainer.unfreeze_MRA()
trainer.fit(train_loader_1, train_loader_2, callbacks)

# evaluate on full training set
print("Evaluating on full training set ...")
loss_1, loss_2, loss = trainer.evaluate(train_loader_1, train_loader_2)
print("Loss_1: {:.5f}".format(loss_1))
print("Loss_2: {:.5f}".format(loss_2))
print("Loss: {:.5f}".format(loss))

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
	"data_shapes": {name: list(A.shape) for (name, A) in [("X_1", X_1), ("X_2", X_2)]}, 
	"train": {key: ndarray2list(value, "float") for cb in callbacks for (key, value) in cb.history.items()}, 
	"test": {"loss_1": float(loss_1), "loss_2": float(loss_2), "loss": float(loss)}, 
	"mean": ndarray2list(model.aligner_2.mean.data.numpy(), "float")
}

# add command-line arguments and script execution time to results
results_dict["args"] = dict(vars(args))
results_dict["time"] = time.time()-time_start

# save results
print("Saving results ...")
with open(os.path.join(results_dir, "results.json"), "w") as fp:
	json.dump(results_dict, fp, indent=2)
trainer.save_model(os.path.join(results_dir, "params.pth"))


print("Done!")