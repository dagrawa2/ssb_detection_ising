import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


class Autoencoder(object):

	def __init__(self, encoder, decoder, epochs=1, lr=1e-3, rescale_lr=False, equivariance_reg=0, equivariance_pre=0, device="cpu"):
		self.encoder = encoder.to(device)
		self.decoder = decoder.to(device)
		self.epochs = epochs
		self.lr = lr
		self.rescale_lr = rescale_lr
		self.equivariance_reg = equivariance_reg
		self.equivariance_pre = equivariance_pre
		self.device = device

		self.latent_dim = self.encoder.output_dim

		if self.rescale_lr:
			param_group1 = [encoder.linear1.weight]
			param_group2 = [decoder.linear2.weight, decoder.linear2.bias]
			param_group3 = [encoder.linear1.bias, encoder.linear2.weight, encoder.linear2.bias, decoder.linear1.weight, decoder.linear1.bias]
			self.optimizer = optim.Adam([{"params": param_group1, "lr": 2*self.lr/encoder.linear1.weight.shape[1]}, {"params": param_group2, "lr": self.lr/2}, {"params": param_group3, "lr": self.lr}])
		else:
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
				Z = self.encoder(X_batch)
				logits = self.decoder(torch.cat([Z, T_batch], 1))
				loss = F.binary_cross_entropy_with_logits(logits, (X_batch+1)/2)
				if self.equivariance_reg > 0 and epoch >= self.equivariance_pre:
					if self.latent_dim == 1:
						transforms = [ \
							lambda x: torch.flip(x, [1]), \
							lambda x: -x \
						]
						cos_similarities = []
						for transform in transforms:
							Z_transformed = self.encoder(transform(X_batch))
							Z_square_norm = Z.pow(2).sum()
							Z_transformed_square_norm = Z_transformed.pow(2).sum()
							loss += self.equivariance_reg*( \
								(1 - Z_transformed_square_norm/Z_square_norm)**2 + \
								1 - (Z*Z_transformed).sum()**2/(Z_square_norm*Z_transformed_square_norm) )
							cos_similarities.append( (Z*Z_transformed).sum().unsqueeze(0)/torch.sqrt(Z_square_norm*Z_transformed_square_norm) )
						loss += 1 + torch.cat(cos_similarities, 0).min()
					else:
						transforms = [ \
							lambda x: x[:,[1,3,0,2]], \
							lambda x: x[:,[1,0,3,2]], \
							lambda x: -x \
						]
						Z_transformed = torch.cat([self.encoder(transform(X_batch)) for transform in transforms], 1)
						psi = torch.linalg.lstsq(Z, Z_transformed).solution.T
						psi_rho = psi[:2]
						psi_tau = psi[2:4]
						psi_sigma = psi[4:]
						I = torch.eye(self.latent_dim, dtype=Z.dtype)
						loss += self.equivariance_reg*( \
							torch.stack([((I - A.T@A)**2).sum() for A in [psi_rho, psi_tau, psi_sigma]], 0).sum() \
							+ torch.stack([((I - A@A)**2).sum() for A in [psi_tau, psi_sigma, psi_rho@psi_rho, psi_rho@psi_tau, psi_tau@psi_sigma]], 0).sum() \
							+ sum((I - psi_rho.T@psi_sigma@psi_rho@psi_sigma)**2).sum() \
							+ torch.stack([F.relu(torch.trace(A)) for A in [psi_rho, psi_tau, psi_sigma]], 0).prod() )
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
		assert self.latent_dim == 1, "generator_reps can only be used for latent dimension 1. For latent dimension 2, use generator_reps_2D."
		Z = self.encode(val_loader)
		Z_flip = self.encode(val_loader, transform=lambda X: torch.flip(X, [1]))
		Z_neg = self.encode(val_loader, transform=lambda X: -X)

		flip_rep = np.sum(Z*Z_flip)/np.sum(Z**2)
		neg_rep = np.sum(Z*Z_neg)/np.sum(Z**2)
		return flip_rep, neg_rep

	def generator_reps_2D(self, val_loader):
		assert self.latent_dim == 2, "generator_reps_2D can only be used for latent dimension 2. For latent dimension 1, use generator_reps."
		Z = self.encode(val_loader)
		Z_transformed = { \
			"rho": self.encode(val_loader, transform=lambda x: x[:,[1,3,0,2]]), \
			"tau": self.encode(val_loader, transform=lambda x: x[:,[1,0,3,2]]), \
			"sigma": self.encode(val_loader, transform=lambda x: -x) \
		}
		reps = {key: np.linalg.lstsq(Z, val)[0].T for (key, val) in Z_transformed.items()}
		return reps

	def save_encoder(self, filename):
		torch.save(self.encoder.state_dict(), filename)

	def save_decoder(self, filename):
		torch.save(self.decoder.state_dict(), filename)

	def load_encoder(self, filename):
		self.encoder.load_state_dict(torch.load(filename))

	def load_decoder(self, filename):
		self.decoder.load_state_dict(torch.load(filename))
