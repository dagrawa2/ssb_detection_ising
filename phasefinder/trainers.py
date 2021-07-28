import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


class Autoencoder(object):

	def __init__(self, encoder, decoder, epochs=1, lr=1e-3, device="cpu"):
		self.encoder = encoder.to(device)
		self.decoder = decoder.to(device)
		self.epochs = epochs
		self.lr = lr
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
				logits = self.decoder(self.encoder(X_batch), T_batch)
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
				logits = self.decoder(self.encoder(X_batch), T_batch)
				loss = F.binary_cross_entropy_with_logits(logits, (X_batch+1)/2)
				losses.append(loss.item())
		loss = np.mean(np.array(losses))

		return loss

	def encode(self, val_loader):
		encodings = []
		with torch.no_grad():
			for (X_batch, _) in val_loader:
				X_batch = X_batch.to(self.device)
				encodings_batch = self.encoder(X_batch)
				encodings.append( encodings_batch.cpu().numpy() )
		encodings = np.concatenate(encodings, 0)

		return encodings

	def save_encoder(self, filename):
		torch.save(self.encoder.state_dict(), filename)

	def save_decoder(self, filename):
		torch.save(self.decoder.state_dict(), filename)

	def load_encoder(self, filename):
		self.encoder.load_state_dict(torch.load(filename))

	def load_decoder(self, filename):
		self.decoder.load_state_dict(torch.load(filename))
