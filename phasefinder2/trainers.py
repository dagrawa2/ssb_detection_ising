import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


class Autoencoder(object):

	def __init__(self, encoder, decoders, epochs=1, lr=1e-3, device="cpu"):
		self.encoder = encoder.to(device)
		self.decoders = [decoder.to(device) for decoder in decoders]
		self.epochs = epochs
		self.lr = lr
		self.device = device

		parameters = list(self.encoder.parameters())
		for decoder in self.decoders:
			parameters += list(decoder.parameters())
		self.optimizer = optim.Adam(parameters, lr=self.lr)

	def fit(self, train_loaders, callbacks):
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
			for batches in zip(*train_loaders):
				loss = 0
				for ((X_batch, ), decoder) in zip(batches, self.decoders):
					X_batch = X_batch.to(self.device)
					logits = decoder(self.encoder(X_batch))
					loss += F.binary_cross_entropy_with_logits(logits, (X_batch+1)/2)
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


	def evaluate(self, val_loaders):
		losses = []
		with torch.no_grad():
			for batches in zip(*val_loaders):
				loss = 0
				for ((X_batch, ), decoder) in zip(batches, self.decoders):
					X_batch = X_batch.to(self.device)
					logits = decoder(self.encoder(X_batch))
					loss += F.binary_cross_entropy_with_logits(logits, (X_batch+1)/2)
				losses.append(loss.item())
		loss = np.mean(np.array(losses))

		return loss

	def encode(self, val_loader):
		encodings = []
		with torch.no_grad():
			for (X_batch, ) in val_loader:
				X_batch = X_batch.to(self.device)
				encodings_batch = self.encoder(X_batch)
				encodings.append( encodings_batch.cpu().numpy() )
		encodings = np.concatenate(encodings, 0)

		return encodings

	def save_model(self, filename):
		torch.save(self.model.state_dict(), filename)

	def load_model(self, filename):
		self.model.load_state_dict(torch.load(filename))
