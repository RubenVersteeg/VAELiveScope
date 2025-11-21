
"""
A Variational Autoencoder (VAE) for spectrogram data.

VAE References
--------------
.. [1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
	arXiv preprint arXiv:1312.6114 (2013).

	`<https://arxiv.org/abs/1312.6114>`_


.. [2] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic
	backpropagation and approximate inference in deep generative models." arXiv
	preprint arXiv:1401.4082 (2014).

	`<https://arxiv.org/abs/1401.4082>`_
"""
__date__ = "November 2018 - November 2019"
import numpy as np
import cupy as cp
import os
import torch
from torch.distributions import LowRankMultivariateNormal, Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast 

# from ava.models.vae_dataset import SyllableDataset
# from ava.plotting.grid_plot import grid_plot

#change the shape to our preferred spectogram
# Spectrogram shape (129, 160)
X_SHAPE = (129,160)
"""Processed spectrogram shape: ``[freq_bins, time_bins]``"""
X_DIM = np.prod(X_SHAPE)
"""Processed spectrogram dimension: ``freq_bins * time_bins``"""



class VAE_KL_leaky(nn.Module):
	"""Variational Autoencoder class for single-channel images.

	Attributes
	----------
	save_dir : str, optional
		Directory where the model is saved. Defaults to ``''``.
	lr : float, optional
		Model learning rate. Defaults to ``1e-3``.
	z_dim : int, optional
		Latent dimension. Defaults to ``32``.
	model_precision : float, optional
		Precision of the observation model. Defaults to ``10.0``.
	device_name : {'cpu', 'cuda', 'auto'}, optional
		Name of device to train the model on. When ``'auto'`` is passed,
		``'cuda'`` is chosen if ``torch.cuda.is_available()``, otherwise
		``'cpu'`` is chosen. Defaults to ``'auto'``.

	Notes
	-----
	The model is trained to maximize the standard ELBO objective:

	.. math:: \mathcal{L} = \mathbb{E}_{q(z|x)} log p(x,z) + \mathbb{H}[q(z|x)]

	where :math:`p(x,z) = p(z)p(x|z)` and :math:`\mathbb{H}` is differential
	entropy. The prior :math:`p(z)` is a unit spherical normal distribution. The
	conditional distribution :math:`p(x|z)` is set as a spherical normal
	distribution to prevent overfitting. The variational distribution,
	:math:`q(z|x)` is an approximately rank-1 multivariate normal distribution.
	Here, :math:`q(z|x)` and :math:`p(x|z)` are parameterized by neural
	networks. Gradients are passed through stochastic layers via the
	reparameterization trick, implemented by the PyTorch `rsample` method.

	The dimensions of the network are hard-coded for use with 128 x 128
	spectrograms. Although a desired latent dimension can be passed to
	`__init__`, the dimensions of the network limit the practical range of
	values roughly 8 to 64 dimensions. Fiddling with the image dimensions will
	require updating the parameters of the layers defined in `_build_network`.
	"""

	def __init__(self, save_dir='', lr=1e-3, z_dim=32, model_precision=10.0, slope = 0.1,
		device_name="auto"):
		"""Construct a VAE.

		Parameters
		----------
		save_dir : str, optional
			Directory where the model is saved. Defaults to the current working
			directory.
		lr : float, optional
			Learning rate of the ADAM optimizer. Defaults to 1e-3.
		z_dim : int, optional
			Dimension of the latent space. Defaults to 32.
		model_precision : float, optional
			Precision of the noise model, p(x|z) = N(mu(z), \Lambda) where
			\Lambda = model_precision * I. Defaults to 10.0.
		device_name: str, optional
			Name of device to train the model on. Valid options are ["cpu",
			"cuda", "auto"]. "auto" will choose "cuda" if it is available.
			Defaults to "auto".

		Note
		----
		- The model is built before it's parameters can be loaded from a file.
			This means `self.z_dim` must match `z_dim` of the model being
			loaded.
		"""
		super(VAE_KL_leaky, self).__init__()
		self.save_dir = save_dir
		self.lr = lr
		self.z_dim = z_dim
		self.model_precision = model_precision
		self.slope = slope
		self.scaler = GradScaler() ###
		assert device_name != "cuda" or torch.cuda.is_available()
		if device_name == "auto":
			device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
		if self.save_dir != '' and not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self._build_network()
		self.optimizer = Adam(self.parameters(), lr=self.lr)
		self.epoch = 0
		self.loss = {'train':{}, 'test':{}}
		self.to(self.device)


	def _build_network(self):
		"""Define all the network layers."""
		# Encoder
		self.conv1 = nn.Conv2d(1, 8, 3,1,padding=1)
		self.conv2 = nn.Conv2d(8, 8, 3,2,padding=1)
		self.conv3 = nn.Conv2d(8, 16,3,1,padding=1)
		self.conv4 = nn.Conv2d(16,16,3,2,padding=1)
		self.conv5 = nn.Conv2d(16,24,3,1,padding=1)
		self.conv6 = nn.Conv2d(24,24,3,2,padding=1)
		self.conv7 = nn.Conv2d(24,32,3,1,padding=1)
		self.bn1 = nn.BatchNorm2d(1)
		self.bn2 = nn.BatchNorm2d(8)
		self.bn3 = nn.BatchNorm2d(8)
		self.bn4 = nn.BatchNorm2d(16)
		self.bn5 = nn.BatchNorm2d(16)
		self.bn6 = nn.BatchNorm2d(24)
		self.bn7 = nn.BatchNorm2d(24)
		self.fc1 = nn.Linear(10880,1024)
		self.fc2 = nn.Linear(1024,256)
		self.fc31 = nn.Linear(256,64)
		self.fc32 = nn.Linear(256,64)
		self.fc33 = nn.Linear(256,64)
		self.fc41 = nn.Linear(64,self.z_dim)
		self.fc42 = nn.Linear(64,self.z_dim)
		self.fc43 = nn.Linear(64,self.z_dim)
		# Decoder
		self.fc5 = nn.Linear(self.z_dim,64)
		self.fc6 = nn.Linear(64,256)
		self.fc7 = nn.Linear(256,1024)
		self.fc8 = nn.Linear(1024,10880)
		self.convt1 = nn.ConvTranspose2d(32,24,3,1,padding=1)
		self.convt2 = nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=(0,1))
		self.convt3 = nn.ConvTranspose2d(24,16,3,1,padding=1)
		self.convt4 = nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=(0,1))
		self.convt5 = nn.ConvTranspose2d(16,8,3,1,padding=1)
		self.convt6 = nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=(0,1))
		self.convt7 = nn.ConvTranspose2d(8,1,3,1,padding=1)
		self.bn8 = nn.BatchNorm2d(32)
		self.bn9 = nn.BatchNorm2d(24)
		self.bn10 = nn.BatchNorm2d(24)
		self.bn11 = nn.BatchNorm2d(16)
		self.bn12 = nn.BatchNorm2d(16)
		self.bn13 = nn.BatchNorm2d(8)
		self.bn14 = nn.BatchNorm2d(8)


	def _get_layers(self):
		"""Return a dictionary mapping names to network layers."""
		return {'fc1':self.fc1, 'fc2':self.fc2, 'fc31':self.fc31,
				'fc32':self.fc32, 'fc33':self.fc33, 'fc41':self.fc41,
				'fc42':self.fc42, 'fc43':self.fc43, 'fc5':self.fc5,
				'fc6':self.fc6, 'fc7':self.fc7, 'fc8':self.fc8, 'bn1':self.bn1,
				'bn2':self.bn2, 'bn3':self.bn3, 'bn4':self.bn4, 'bn5':self.bn5,
				'bn6':self.bn6, 'bn7':self.bn7, 'bn8':self.bn8, 'bn9':self.bn9,
				'bn10':self.bn10, 'bn11':self.bn11, 'bn12':self.bn12,
				'bn13':self.bn13, 'bn14':self.bn14, 'conv1':self.conv1,
				'conv2':self.conv2, 'conv3':self.conv3, 'conv4':self.conv4,
				'conv5':self.conv5, 'conv6':self.conv6, 'conv7':self.conv7,
				'convt1':self.convt1, 'convt2':self.convt2,
				'convt3':self.convt3, 'convt4':self.convt4,
				'convt5':self.convt5, 'convt6':self.convt6,
				'convt7':self.convt7}


	def encode(self, x):
		"""
		Compute :math:`q(z|x)`.

		.. math:: q(z|x) = \mathcal{N}(\mu, \Sigma)
		.. math:: \Sigma = u u^{T} + \mathtt{diag}(d)

		where :math:`\mu`, :math:`u`, and :math:`d` are deterministic functions
		of `x` and :math:`\Sigma` denotes a covariance matrix.

		Parameters
		----------
		x : torch.Tensor
			The input images, with shape: ``[batch_size, height=128,
			width=128]``

		Returns
		-------
		mu : torch.Tensor
			Posterior mean, with shape ``[batch_size, self.z_dim]``
		u : torch.Tensor
			Posterior covariance factor, as defined above. Shape:
			``[batch_size, self.z_dim]``
		d : torch.Tensor
			Posterior diagonal factor, as defined above. Shape:
			``[batch_size, self.z_dim]``
		"""
		slope = self.slope
		x = x.unsqueeze(1)
		x = F.leaky_relu(self.conv1(self.bn1(x)), negative_slope=slope)
		x = F.leaky_relu(self.conv2(self.bn2(x)), negative_slope=slope)
		x = F.leaky_relu(self.conv3(self.bn3(x)), negative_slope=slope)
		x = F.leaky_relu(self.conv4(self.bn4(x)), negative_slope=slope)
		x = F.leaky_relu(self.conv5(self.bn5(x)), negative_slope=slope)
		x = F.leaky_relu(self.conv6(self.bn6(x)), negative_slope=slope)
		x = F.leaky_relu(self.conv7(self.bn7(x)), negative_slope=slope)
		x = x.view(-1, 10880)
		x = F.leaky_relu(self.fc1(x), negative_slope=slope)
		x = F.leaky_relu(self.fc2(x), negative_slope=slope)
		mu = F.leaky_relu(self.fc31(x), negative_slope=slope)
		mu = self.fc41(mu)
		u = F.leaky_relu(self.fc32(x), negative_slope=slope)
		u = self.fc42(u).unsqueeze(-1) # Last dimension is rank \Sigma = 1.
		d = F.leaky_relu(self.fc33(x), negative_slope=slope)
		d = torch.exp(self.fc43(d)) # d must be positive.
		return mu, u, d


	def decode(self, z):
		"""
		Compute :math:`p(x|z)`.

		.. math:: p(x|z) = \mathcal{N}(\mu, \Lambda)

		.. math:: \Lambda = \mathtt{model\_precision} \cdot I

		where :math:`\mu` is a deterministic function of `z`, :math:`\Lambda` is
		a precision matrix, and :math:`I` is the identity matrix.

		Parameters
		----------
		z : torch.Tensor
			Batch of latent samples with shape ``[batch_size, self.z_dim]``

		Returns
		-------
		x : torch.Tensor
			Batch of means mu, described above. Shape: ``[batch_size,
			X_DIM=128*128]``
		"""
		slope = self.slope
		z = F.leaky_relu(self.fc5(z), negative_slope=slope)
		z = F.leaky_relu(self.fc6(z), negative_slope=slope)
		z = F.leaky_relu(self.fc7(z), negative_slope=slope)
		z = F.leaky_relu(self.fc8(z), negative_slope=slope)
		z = z.view(-1,32,17,20)
		z = F.leaky_relu(self.convt1(self.bn8(z)), negative_slope=slope)
		z = F.leaky_relu(self.convt2(self.bn9(z)), negative_slope=slope)
		z = F.leaky_relu(self.convt3(self.bn10(z)), negative_slope=slope)
		z = F.leaky_relu(self.convt4(self.bn11(z)), negative_slope=slope)
		z = F.leaky_relu(self.convt5(self.bn12(z)), negative_slope=slope)
		z = F.leaky_relu(self.convt6(self.bn13(z)), negative_slope=slope)
		z = F.leaky_relu(self.convt7(self.bn14(z)), negative_slope=slope) 
		return z.view(-1, X_DIM)


	def forward(self, x, return_latent_rec=False):
		"""
		Send `x` round trip and compute a loss.

		Parameters
		----------
		x : torch.Tensor
			A batch of samples from the data distribution (spectrograms).
			Shape: ``[batch_size, height=129, width=160]``
		return_latent_rec : bool, optional
			Whether to return latent means and reconstructions. Defaults to
			``False``.

		Returns
		-------
	    loss : torch.Tensor
		reconstruction_loss : torch.Tensor
		KLD : torch.Tensor

		if `return_latent_rec` is ``True``, also returns:
		z : np.ndarray
		"""
		
		mu, _, d = self.encode(x)

		mean = mu
		logvar = torch.log(d + 1e-8)

		logvar = torch.clamp(logvar, min=-20, max=10) # to prevent numerical stability

		# Sample from the posterior.
		eps = torch.randn_like(mean)
		z = mean + eps * torch.exp(0.5*logvar)

		# Compute the reconstruction.
		x_recon = self.decode(z)

		# Compute the loss.
		# reconstruction_loss = F.mse_loss(x_recon, x.view(-1, X_DIM), reduction='sum')
		l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_recon, 2), dim=1)
		# reconstruction_loss = self.model_precision * torch.sum(l2s) #vary the model precision to change the weight between losses
		reconstruction_loss = torch.sum(l2s) 
		KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

		loss = reconstruction_loss + KLD

		if return_latent_rec:
			return loss, z.detach().cpu().numpy(), \
				x_recon.view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy()
		return loss, reconstruction_loss, KLD
	
	# def forward(self, x, return_latent_rec=False):
	# 	# 1. Encode to get posterior parameters
	# 	mu, u, d = self.encode(x)

	# 	# 2. Define the posterior distribution
	# 	q_z_x = LowRankMultivariateNormal(loc=mu, cov_factor=u.unsqueeze(-1), cov_diag=d)

	# 	# 3. Sample from the posterior using the reparameterization trick
	# 	z = q_z_x.rsample()

	# 	# 4. Decode the sample to get the reconstruction mean
	# 	x_recon = self.decode(z)

	# 	# 5. Define the loss components

	# 	# Reconstruction Loss (Negative Log-Likelihood of a Gaussian Observation Model)
	# 	# Using MSE. The `model_precision` weights this term.
	# 	reconstruction_loss = self.model_precision * torch.sum(
	# 		torch.pow(x.view(x.shape[0], -1) - x_recon, 2)
	# 	)

	# 	# KL Divergence
	# 	# Define the standard normal prior distribution p(z)
	# 	p_z = Normal(torch.zeros_like(mu), torch.ones_like(mu))
		
	# 	# Calculate KL divergence using PyTorch's built-in function
	# 	KLD = torch.sum(torch.distributions.kl_divergence(q_z_x, p_z))

	# 	# 6. Total Loss (Negative ELBO)
	# 	loss = reconstruction_loss + KLD

	# 	if return_latent_rec:
	# 		return loss, z.detach().cpu().numpy(), \
	# 			x_recon.view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy()
	# 	return loss, reconstruction_loss, KLD


	def train_epoch(self, train_loader):
		"""
		Train the model for a single epoch.

		Parameters
		----------
		train_loader : torch.utils.data.Dataloader
			ava.models.vae_dataset.SyllableDataset Dataloader for training set

		Returns
		-------
		elbo : float
			A biased estimate of the ELBO, estimated using samples from
			`train_loader`.
		"""
		self.train()
		train_loss = 0.0
		train_reconstruction_loss = 0.0
		train_KLD = 0.0
		for batch_idx, (data,) in enumerate(train_loader): ###use for training with numpy data in notebook
		# for batch_idx, (data, labels) in enumerate(train_loader): ###use for training with pytorch data in python file
			self.optimizer.zero_grad()
			data = data.to(self.device)
			# print('data shape in train_epoch:', data.shape)
			# print(f"Batch {batch_idx} data device (after .to()): {data.device}")
			with autocast(): ###
				loss, l2s, KL = self.forward(data)
			train_loss += loss.item()
			train_reconstruction_loss += l2s.item()
			train_KLD += KL.item()
			self.scaler.scale(loss).backward() ###
			torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # gradient clipping to prevent numerical instability
			# unclipped_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
			# print(f"Unclipped gradient norm: {unclipped_norm.item():.4f}")
			self.scaler.step(self.optimizer) ###
			self.scaler.update() ###
			# loss.backward()
			# self.optimizer.step()
		# Average the loss over the dataset.
		num_samples_in_loader = len(train_loader.dataset)
		train_loss /= num_samples_in_loader
		train_reconstruction_loss /= num_samples_in_loader
		train_KLD /= num_samples_in_loader

		print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, \
				train_loss))
		print('Reconstruction loss: {:.4f}'.format(train_reconstruction_loss))
		print('KLD: {:.4f}'.format(train_KLD))
		self.epoch += 1
		return train_loss, train_reconstruction_loss, train_KLD


	def test_epoch(self, test_loader):
		"""
		Test the model on a held-out test set, return an ELBO estimate.

		Parameters
		----------
		test_loader : torch.utils.data.Dataloader
			ava.models.vae_dataset.SyllableDataset Dataloader for test set

		Returns
		-------
		elbo : float
			An unbiased estimate of the ELBO, estimated using samples from
			`test_loader`.
		"""
		self.eval()
		test_loss = 0.0
		test_reconstruction_loss = 0.0
		test_KLD = 0.0
		with torch.no_grad():
			for i, (data,) in enumerate(test_loader): ###use for testing with numpy data in notebook
			# for i, (data, labels) in enumerate(test_loader): ###use for testing with
				data = data.to(self.device)
				# print(f"Batch {i} data device (after .to()): {data.device}")
				with autocast(): ###
					loss, l2s, KL= self.forward(data)
				test_loss += loss.item()
				test_reconstruction_loss += l2s.item()
				test_KLD += KL.item()
		# Average the loss over the dataset.
		num_samples_in_loader = len(test_loader.dataset)
		test_loss /= num_samples_in_loader
		test_reconstruction_loss /= num_samples_in_loader
		test_KLD /= num_samples_in_loader
	
		print('Test loss: {:.4f}'.format(test_loss))
		print('Test reconstruction loss: {:.4f}'.format(test_reconstruction_loss))
		print('Test KLD: {:.4f}'.format(test_KLD))
		return test_loss, test_reconstruction_loss, test_KLD


	def train_loop(self, loaders, epochs=100, test_freq=2, save_freq=10,
		vis_freq=1, patience=10):
		"""
		Train the model for multiple epochs, testing and saving along the way.

		Parameters
		----------
		loaders : dictionary
			Dictionary mapping the keys ``'test'`` and ``'train'`` to respective
			torch.utils.data.Dataloader objects.
		epochs : int, optional
			Number of (possibly additional) epochs to train the model for.
			Defaults to ``100``.
		test_freq : int, optional
			Testing is performed every `test_freq` epochs. Defaults to ``2``.
		save_freq : int, optional
			The model is saved every `save_freq` epochs. Defaults to ``10``.
		vis_freq : int, optional
			Syllable reconstructions are plotted every `vis_freq` epochs.
			Defaults to ``1``.
		"""
		print("="*40)
		print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
		print("Training set:", len(loaders['train'].dataset))
		print("Test set:", len(loaders['test'].dataset))
		print("="*40)
		# For some number of epochs...
		train_loss = np.array([])
		test_loss = np.array([])
		reconstruction_loss_train = np.array([])
		reconstruction_loss_test = np.array([])
		KLD_loss_train = np.array([])
		KLD_loss_test = np.array([])
		early_stopping_length = np.array([])
		early_stopping_counter = 0
		min_test_loss = np.inf
		for epoch in range(self.epoch, self.epoch+epochs):
			# Run through the training data and record a loss.
			loss, reconstruction_loss,KLD = self.train_epoch(loaders['train'])
			self.loss['train'][epoch] = loss
			train_loss = np.append(train_loss, loss)
			reconstruction_loss_train = np.append(reconstruction_loss_train, reconstruction_loss)
			KLD_loss_train = np.append(KLD_loss_train, KLD)
			#for reconstruction
			# Run through the test data and record a loss.
			if (test_freq is not None) and (epoch % test_freq == 0):
				losst, reconstruction_losst,KLDt= self.test_epoch(loaders['test'])
				self.loss['test'][epoch] = losst
				test_loss = np.append(test_loss, losst)
				reconstruction_loss_test =  np.append(reconstruction_loss_test, reconstruction_losst)
				KLD_loss_test = np.append(KLD_loss_test, KLDt)
				#early stopping
				if losst < min_test_loss:
					min_test_loss = losst
					early_stopping_counter = 0
					filename_best = "checkpoint_best.tar"
					self.save_state(filename_best)
				else:
					early_stopping_counter += 1
				if early_stopping_counter > patience:
					print('Early stopping at epoch', epoch)
					#load the best model
					self.load_state(filename_best)
					break
				early_stopping_length = np.append(early_stopping_length, early_stopping_counter)
			# Save the model.
			if (save_freq is not None) and (epoch % save_freq == 0) and \
					(epoch > 0):
				filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
				self.save_state(filename)
			# # Plot reconstructions.
			# if (vis_freq is not None) and (epoch % vis_freq == 0):
			# 	self.visualize(loaders['test'])
			#plotting the training and test loss and early stopping length
		plt.figure()
		plt.subplot(3,1,1)
		plt.plot(train_loss)
		plt.yscale('log')
		plt.title('Training Loss')
		plt.subplot(3,1,2)
		plt.plot(test_loss)
		plt.yscale('log')
		plt.title('Test Loss')
		plt.subplot(3,1,3)
		plt.plot(early_stopping_length)
		plt.title('Early Stopping Length')
		plt.tight_layout()
		plt.show()

		return train_loss, test_loss, early_stopping_length, reconstruction_loss_train, reconstruction_loss_test, KLD_loss_train, KLD_loss_test


	def save_state(self, filename):
		"""Save all the model parameters to the given file."""
		layers = self._get_layers()
		state = {}
		for layer_name in layers:
			state[layer_name] = layers[layer_name].state_dict()
		state['optimizer_state'] = self.optimizer.state_dict()
		state['loss'] = self.loss
		state['z_dim'] = self.z_dim
		state['epoch'] = self.epoch
		state['lr'] = self.lr
		state['save_dir'] = self.save_dir
		filename = os.path.join(self.save_dir, filename)
		torch.save(state, filename)


	def load_state(self, filename):
		"""
		Load all the model parameters from the given ``.tar`` file.

		The ``.tar`` file should be written by `self.save_state`.

		Parameters
		----------
		filename : str
			File containing a model state.

		Note
		----
		- `self.lr`, `self.save_dir`, and `self.z_dim` are not loaded.
		"""
		checkpoint = torch.load(filename, map_location=self.device)
		assert checkpoint['z_dim'] == self.z_dim
		layers = self._get_layers()
		for layer_name in layers:
			layer = layers[layer_name]
			layer.load_state_dict(checkpoint[layer_name])
		self.optimizer.load_state_dict(checkpoint['optimizer_state'])
		self.loss = checkpoint['loss']
		self.epoch = checkpoint['epoch']


	# def visualize(self, loader, num_specs=5, gap=(2,6), \
	# 	save_filename='reconstruction.pdf'):
	# 	"""
	# 	Plot spectrograms and their reconstructions.

	# 	Spectrograms are chosen at random from the Dataloader Dataset.

	# 	Parameters
	# 	----------
	# 	loader : torch.utils.data.Dataloader
	# 		Spectrogram Dataloader
	# 	num_specs : int, optional
	# 		Number of spectrogram pairs to plot. Defaults to ``5``.
	# 	gap : int or tuple of two ints, optional
	# 		The vertical and horizontal gap between images, in pixels. Defaults
	# 		to ``(2,6)``.
	# 	save_filename : str, optional
	# 		Where to save the plot, relative to `self.save_dir`. Defaults to
	# 		``'temp.pdf'``.

	# 	Returns
	# 	-------
	# 	specs : numpy.ndarray
	# 		Spectgorams from `loader`.
	# 	rec_specs : numpy.ndarray
	# 		Corresponding spectrogram reconstructions.
	# 	"""
	# 	# Collect random indices.
	# 	assert num_specs <= len(loader.dataset) and num_specs >= 1
	# 	indices = np.random.choice(np.arange(len(loader.dataset)),
	# 		size=num_specs,replace=False)
	# 	# Retrieve spectrograms from the loader.
	# 	specs = torch.stack(loader.dataset[indices]).to(self.device)
	# 	# Get resonstructions.
	# 	with torch.no_grad():
	# 		_, _, rec_specs = self.forward(specs, return_latent_rec=True)
	# 	specs = specs.detach().cpu().numpy()
	# 	all_specs = np.stack([specs, rec_specs])
	# 	# Plot.
	# 	save_filename = os.path.join(self.save_dir, save_filename)
	# 	grid_plot(all_specs, gap=gap, filename=save_filename)
	# 	return specs, rec_specs


	def get_latent(self, loader):
		"""
		Get latent means for all syllable in the given loader.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			ava.models.vae_dataset.SyllableDataset Dataloader.

		Returns
		-------
		latent : numpy.ndarray
			Latent means. Shape: ``[len(loader.dataset), self.z_dim]``

		Note
		----
		- Make sure your loader is not set to shuffle if you're going to match
		  these with labels or other fields later.
		"""
		latent = np.zeros((len(loader.dataset), self.z_dim))
		i = 0
		for data in loader:
			data = data.to(self.device)
			with torch.no_grad():
				mu, _, _ = self.encode(data)
			mu = mu.detach().cpu().numpy()
			latent[i:i+len(mu)] = mu
			i += len(mu)
		return latent
	

	def get_state_encoder(self):
		"""
		Return the state of the encoder and not the decoder.

		Returns
		-------
		state : dict
			Dictionary containing the state of the encoder layers.
		"""
		layers = self._get_layers()
		state = {}
		for layer_name in layers:
			if 'fc' in layer_name or 'conv' in layer_name or 'bn' in layer_name:
				state[layer_name] = layers[layer_name].state_dict()
		#remove decoder layers
		state.pop('fc5', None)
		state.pop('fc6', None)
		state.pop('fc7', None)
		state.pop('fc8', None)
		state.pop('convt1', None)
		state.pop('convt2', None)
		state.pop('convt3', None)
		state.pop('convt4', None)
		state.pop('convt5', None)
		state.pop('convt6', None)
		state.pop('convt7', None)
		state.pop('bn8', None)
		state.pop('bn9', None)
		state.pop('bn10', None)
		state.pop('bn11', None)
		state.pop('bn12', None)
		state.pop('bn13', None)
		state.pop('bn14', None)

		#flatten the dictionary
		state_flatten = {}
		for key in state:
			for subkey in state[key]:
				state_flatten[key+'.'+subkey] = state[key][subkey]
		return state_flatten




if __name__ == '__main__':
	pass
