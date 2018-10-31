from __future__ import print_function, division

import torch
import torch.nn as nn

def randomSeed(random_seed):
	"""Given a random seed, this will help reproduce results across runs"""
	if random_seed is not None:
		torch.manual_seed(random_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(random_seed)


def getActivation(activation):
	if activation == 'softplus':
		return nn.Softplus()
	elif activation == 'relu':
		return nn.ReLU()


class ConvLayer(nn.Module):
	"""
	Convolutional operation on graphs
	"""

	def __init__(self, atom_fea_len, nbr_fea_len, random_seed=None, activation='relu'):
		"""
		Initialize ConvLayer.

		Parameters
		----------

		atom_fea_len: int
			Number of atom hidden features.
		nbr_fea_len: int
			Number of bond features.
		random_seed: int
			Seed to reproduce consistent runs
		activation: string ('relu' or 'softplus')
			Decides the activation function
		"""
		randomSeed(random_seed)
		super(ConvLayer, self).__init__()
		self.atom_fea_len = atom_fea_len
		self.nbr_fea_len = nbr_fea_len
		self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
								 2 * self.atom_fea_len)
		self.sigmoid = nn.Sigmoid()
		self.activation1 = getActivation(activation)
		self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
		self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
		self.activation2 = getActivation(activation)

	def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
		"""
		Forward pass

		N: Total number of atoms in the batch
		M: Max number of neighbors

		Parameters
		----------

		atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
		  Atom hidden features before convolution
		nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
		  Bond features of each atom's M neighbors
		nbr_fea_idx: torch.LongTensor shape (N, M)
		  Indices of M neighbors of each atom

		Returns
		-------

		atom_out_fea: nn.Variable shape (N, atom_fea_len)
		  Atom hidden features after convolution

		"""
		# TODO will there be problems with the index zero padding?
		N, M = nbr_fea_idx.shape
		# convolution
		atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]		# [N, M, atom_fea_len]
		total_nbr_fea = torch.cat(
			[atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
			 atom_nbr_fea, nbr_fea], dim=2)		# [N, M, nbr_fea_len + 2*atom_fea_len]
		total_gated_fea = self.fc_full(total_nbr_fea)		# [N, M, 2*atom_fea_len]
		total_gated_fea = self.bn1(total_gated_fea.view(
			-1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)
		nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)		# [N, M, atom_fea_len] each
		nbr_filter = self.sigmoid(nbr_filter)
		nbr_core = self.activation1(nbr_core)
		nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)		# [N, atom_fea_len]
		nbr_sumed = self.bn2(nbr_sumed)
		out = self.activation2(atom_in_fea + nbr_sumed)
		return out


class MTCGCNN(nn.Module):
	"""
	Create a multi-task crystal graph convolutional neural network for predicting multiple
	material properties.
	"""

	def __init__(self, orig_atom_fea_len, nbr_fea_len,
					atom_fea_len=64, n_conv=3, h_fea_len=128, n_p=1, activation='softplus',
					random_seed=None, hard_parameter_sharing=True, n_hp=1, dropout=0):
		"""
		Initialize MTCGCNN.

		Parameters
		----------

		orig_atom_fea_len: int
			Number of atom features in the input.
		nbr_fea_len: int
			Number of bond features.
		atom_fea_len: int
			Number of hidden atom features in the convolutional layers
		n_conv: int
			Number of convolutional layers
		h_fea_len: int
			Number of hidden features after pooling
		n_p: int
			Number of final output nodes, equivalent to the number of properties
			predicting (for regression case)
		random_seed: int
			Seed to reproduce consistent runs
		hard_parameter_sharing: int
			This shares the embedding network across various multi-property prediction
			and has self defined linear layers for each property in the end. NOTE that
			this is the most sensible way or using multitasking (otherwise, one single
			layer at the end might overfit to the training set badly)
		n_hp: int
			Number of hidden layers in the hard parameter sharing level.
		activation: string ('relu' or 'softplus')
			Decides the activation function
		dropout: fraction of nodes to dropout every forward iteration while training
		"""
		randomSeed(random_seed)
		super(MTCGCNN, self).__init__()
		self.hard_parameter_sharing = hard_parameter_sharing
		self.num_outputs = n_p
		self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
		self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
											  nbr_fea_len=nbr_fea_len, random_seed=random_seed,
											  activation=activation)
									for _ in range(n_conv)])
		self.dropout1 = nn.Dropout(p=dropout)

		if self.hard_parameter_sharing:
			self.conv_to_fc = nn.ModuleList([nn.Linear(atom_fea_len, h_fea_len)\
									for _ in range(self.num_outputs)])
			self.conv_to_fc_activation = nn.ModuleList([getActivation(activation) for _ in range(self.num_outputs)])
			if n_hp > 1:
				self.fc_hp = nn.ModuleList([
								nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_hp - 1)])
									for _ in range(self.num_outputs)])
				self.fc_activation = nn.ModuleList([
								nn.ModuleList([getActivation(activation) for _ in range(n_hp - 1)])
									for _ in range(self.num_outputs)])
			self.fc_out = nn.ModuleList([nn.Linear(h_fea_len, 1) for _ in range(self.num_outputs)])
		else:
			self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
			self.conv_to_fc_activation = getActivation(activation)
			self.fc_out = nn.Linear(h_fea_len, self.num_outputs)

	def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
		"""
		Forward pass

		N: Total number of atoms in the batch
		M: Max number of neighbors
		N0: Total number of crystals in the batch

		Parameters
		----------

		atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
		  Atom features from atom type
		nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
		  Bond features of each atom's M neighbors
		nbr_fea_idx: torch.LongTensor shape (N, M)
		  Indices of M neighbors of each atom
		crystal_atom_idx: list of torch.LongTensor of length N0
		  Mapping from the crystal idx to atom idx

		Returns
		-------

		prediction: nn.Variable shape (N, )
		  Atom hidden features after convolution

		"""
		atom_fea = self.embedding(atom_fea)
		for conv_func in self.convs:
			atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
		crys_fea = self.pooling(atom_fea, crystal_atom_idx)
		crys_fea = self.dropout1(crys_fea)
		# crys_fea is the descriptor of the crystal which is shared by all the further tasks

		if self.hard_parameter_sharing:
			crys_features = [self.conv_to_fc[i](self.conv_to_fc_activation[i](crys_fea))\
								for i in range(self.num_outputs)]
			crys_features = [self.conv_to_fc_activation[i](crys_features[i]) for i in range(self.num_outputs)]

			# get the processed feature vectors using which we get the outputs
			processed_features = []
			for i in range(self.num_outputs):
				out_val = crys_features[i]
				if hasattr(self, 'fc_hp'):
					for fc, activation in zip(self.fc_hp[i], self.fc_activation[i]):
						out_val = activation(fc(out_val))
				processed_features.append(out_val)

			# final output layer
			out = [self.fc_out[i](processed_features[i]) for i in range(self.num_outputs)]
			out = torch.cat(out, 1)
		else:
			crys_fea = self.conv_to_fc(self.conv_to_fc_activation(crys_fea))
			crys_fea = self.conv_to_fc_activation(crys_fea)
			out = self.fc_out(crys_fea)
		return out, crys_fea

	def pooling(self, atom_fea, crystal_atom_idx):
		"""
		Pooling the atom features to crystal features

		N: Total number of atoms in the batch
		N0: Total number of crystals in the batch

		Parameters
		----------

		atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
		  Atom feature vectors of the batch
		crystal_atom_idx: list of torch.LongTensor of length N0
		  Mapping from the crystal idx to atom idx
		"""
		assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
			   atom_fea.data.shape[0]
		summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
					  for idx_map in crystal_atom_idx]
		return torch.cat(summed_fea, dim=0)
