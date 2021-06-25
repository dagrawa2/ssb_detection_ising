import time
import numpy as np


class Callback(object):
	"""Base class for callbacks to monitor training."""

	def __init__(self, *args, **kwargs):
		"""Class constructor."""
		pass

	def start_of_training(self):
		"""Called before training loop."""
		pass

	def start_of_epoch(self, epoch):
		"""Called at the start of each epoch.

		Args:
			epoch: Type int, number of epochs that have already completed.
		"""
		pass

	def end_of_epoch(self, epoch, batch_logs):
		"""Called at the end of each epoch.
			epochs: Type int, number of epochs that have already completed,
				not including the current one.
			batch_logs: Dict of metrics on the training minibatches of this epoch.
		"""
		pass

	def end_of_training(self):
		"""Called after the training loop."""
		pass


class Training(Callback):
	"""Monitor metrics on training set based on minibatches."""

	def __init__(self):
		super(Training, self).__init__()
		self.history = {"epoch": [], "time": [], "loss_1": [], "loss_2": [], "loss": []}

	def start_of_training(self):
		self.history["epoch"].append(0)
		self.history["time"].append(0)
		self.history["loss_1"].append(0)
		self.history["loss_2"].append(0)
		self.history["loss"].append(0)

	def start_of_epoch(self, epoch):
		self._initial_time = time.time()

	def end_of_epoch(self, epoch, batch_logs):
		delta_time = time.time() - self._initial_time
		loss_1 = np.mean(np.asarray(batch_logs["loss_1"]))
		loss_2 = np.mean(np.asarray(batch_logs["loss_2"]))
		loss = np.mean(np.asarray(batch_logs["loss"]))
		self.history["epoch"].append(epoch+1)
		self.history["time"].append( self.history["time"][-1]+delta_time )
		self.history["loss_1"].append(loss_1)
		self.history["loss_2"].append(loss_2)
		self.history["loss"].append(loss)
		print("Epoch {} loss {:.3f}".format(epoch+1, loss))

	def end_of_training(self):
		self.history = {key: np.array(value) for (key, value) in self.history.items()}
