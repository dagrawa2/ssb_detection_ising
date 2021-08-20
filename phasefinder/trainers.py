import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


class Autoencoder(object):

	def __init__(self, encoder, decoder, epochs=1, lr=1e-3, equivariance_reg=0, equivariance_pre=0, device="cpu"):
		self.encoder = encoder.to(device)
		self.decoder = decoder.to(device)
		self.epochs = epochs
		self.lr = lr
		self.equivariance_reg = equivariance_reg
		self.equivariance_pre = equivariance_pre
		self.device = device

		parameters = list(self.encoder.parameters()) + list(decoder.parameters())
		self.optimizer = optim.Adam(parameters, lr=self.lr)

	def fit(self, train_loader, callbacks):
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
			batch_logs = {"loss": []}
			for (X_batch, T_batch) in train_loader:
				X_batch = X_batch.to(self.device)
				T_batch = T_batch.to(self.device)
				if self.equivariance_reg > 0 and epoch >= self.equivariance_pre:
					Z = self.encoder(X_batch)
					Z_flip = self.encoder(torch.flip(X_batch, [1]))
					Z_neg = self.encoder(-X_batch)
					logits = self.decoder(torch.cat([Z, T_batch], 1))
					loss = F.binary_cross_entropy_with_logits(logits, (X_batch+1)/2) + self.equivariance_reg*( \
						1 - (Z*Z_flip).sum()**2/(Z.pow(2).sum()*Z_flip.pow(2).sum()) \
						+ 1 - (Z*Z_neg).sum()**2/(Z.pow(2).sum()*Z_neg.pow(2).sum()) )
				else:
					logits = self.decoder(torch.cat([self.encoder(X_batch), T_batch], 1))
					loss = F.binary_cross_entropy_with_logits(logits, (X_batch+1)/2)
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()
				batch_logs["loss"].append(loss.item())
			# callbacks at the end of the epoch
			for cb in callbacks:
				cb.end_of_epoch(epoch, batch_logs)

		# callbacks at the end of training
		for cb in callbacks:
			cb.end_of_training()
		print("---")


	def evaluate(self, val_loader):
		losses = []
		with torch.no_grad():
			for (X_batch, T_batch) in val_loader:
				X_batch = X_batch.to(self.device)
				T_batch = T_batch.to(self.device)
				logits = self.decoder(torch.cat([self.encoder(X_batch), T_batch], 1))
				loss = F.binary_cross_entropy_with_logits(logits, (X_batch+1)/2)
				losses.append(loss.item())
		loss = np.mean(np.array(losses))

		return loss

	def encode(self, val_loader, transform=None):
		encodings = []
		with torch.no_grad():
			for (X_batch, _) in val_loader:
				X_batch = X_batch.to(self.device)
				if transform is not None:
					X_batch = transform(X_batch)
				encodings_batch = self.encoder(X_batch)
				encodings.append( encodings_batch.cpu().numpy() )
		encodings = np.concatenate(encodings, 0)

		return encodings

	def generator_reps(self, val_loader):
		Z = self.encode(val_loader)
		Z_flip = self.encode(val_loader, transform=lambda X: torch.flip(X, [1]))
		Z_neg = self.encode(val_loader, transform=lambda X: -X)

		flip_rep = np.sum(Z*Z_flip)/np.sum(Z**2)
		neg_rep = np.sum(Z*Z_neg)/np.sum(Z**2)
		return flip_rep, neg_rep

	def save_encoder(self, filename):
		torch.save(self.encoder.state_dict(), filename)

	def save_decoder(self, filename):
		torch.save(self.decoder.state_dict(), filename)

	def load_encoder(self, filename):
		self.encoder.load_state_dict(torch.load(filename))

	def load_decoder(self, filename):
		self.decoder.load_state_dict(torch.load(filename))
