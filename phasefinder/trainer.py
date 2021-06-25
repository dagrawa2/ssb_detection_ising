import numpy as np
import torch
from torch import nn, optim


class Trainer(object):

	def __init__(self, model, epochs=1, lr=1e-3, device="cpu"):
		self.model = model
		self.epochs = epochs
		self.lr = lr
		self.device = device

		self.model = self.model.to(self.device)
		self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

	def fit(self, train_loader_1, train_loader_2, callbacks):
		# callbacks before training
		callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
		for cb in callbacks:
			cb.start_of_training()
		print("Training model ...")
		print("---")
		for epoch in range(self.epochs):
			# callbacks at the start of the epoch
			for cb in callbacks:
				cb.start_of_epoch(epoch)
			batch_logs = {"loss_1": [], "loss_2": [], "loss": []}
			for ((X_1_batch, ), (X_2_batch, )) in zip(train_loader_1, train_loader_2):
				X_1_batch = X_1_batch.to(self.device)
				X_2_batch = X_2_batch.to(self.device)
				loss_1 = self.model(X_1_batch, broken=False)
				loss_2 = self.model(X_2_batch, broken=True)
				loss = loss_1 + loss_2 + self.model.regularizer()
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()
				batch_logs["loss_1"].append(loss_1.item())
				batch_logs["loss_2"].append(loss_2.item())
				batch_logs["loss"].append(loss.item())
			# callbacks at the end of the epoch
			for cb in callbacks:
				cb.end_of_epoch(epoch, batch_logs)

		# callbacks at the end of training
		for cb in callbacks:
			cb.end_of_training()
		print("---")


	def evaluate(self, val_loader_1, val_loader_2):
		losses_1 = []
		losses_2 = []
		losses = []
		with torch.no_grad():
			for ((X_1_batch, ), (X_2_batch, )) in zip(val_loader_1, val_loader_2):
				X_1_batch = X_1_batch.to(self.device)
				X_2_batch = X_2_batch.to(self.device)
				losses_1.append( self.model(X_1_batch, broken=False).item() )
				losses_2.append( self.model(X_2_batch, broken=True).item() )
				losses.append( losses_1[-1]+losses_2[-1] + self.model.regularizer().item() )

		loss_1 = np.mean(np.array(losses_1))
		loss_2 = np.mean(np.array(losses_2))
		loss = np.mean(np.array(losses))

		return loss_1, loss_2, loss

	def init_MRA(self, train_loader_1, train_loader_2, iters=1):
		with torch.no_grad():
			Z = torch.cat([self.model.observable(X_batch.to(self.device)) for (X_batch, ) in train_loader_1], 0)
			self.model.aligner_1.EM(Z, iters=iters)
			Z = torch.cat([self.model.observable(X_batch.to(self.device)) for (X_batch, ) in train_loader_2], 0)
			self.model.aligner_2.EM(Z, iters=iters)

	def unfreeze_MRA(self):
		self.model.aligner_2.mean.requires_grad = True
		self.model.aligner_2.variance_unconstrained.requires_grad = True
		self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)


	def save_model(self, filename):
		torch.save(self.model.state_dict(), filename)

	def load_model(self, filename):
		self.model.load_state_dict(torch.load(filename))
