import sys, os, shutil, time, csv, warnings, random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from tqdm import tqdm
# from torchviz import make_dot

from model import MTCGCNN
from data import collate_pool, get_train_val_test_loader
from data import CIFData
from plotter import plotMultiGraph, plotGraph

best_error = 0
UNDEFINED_INF = 1000000
USE_WEIGHTED_LOSS = False

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Argument(object):

	def __init__(self, root_dir, save_dir, disable_cuda=True, workers=0, scheduler='MultiStepLR', metric='mae',
				epochs=30, start_epoch=0, batch_size=256, lr=0.01, lr_milestones=[100], momentum=0.9,
				weight_decay=0.0, print_freq=10, resume='', train_size=None, val_size=1000,
				test_size=1000, optimizer='SGD', atom_fea_len=64, h_fea_len=128, n_conv=3, n_hp=1,
				seed=123, weights=None, dropout=0):
		self.root_dir = root_dir
		self.save_dir = save_dir
		self.disable_cuda = disable_cuda
		self.workers = workers
		self.epochs = epochs
		self.start_epoch = start_epoch
		self.batch_size = batch_size
		self.lr = lr
		self.lr_milestones = lr_milestones
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.print_freq = print_freq
		self.resume = resume
		self.train_size = train_size
		self.val_size = val_size
		self.test_size = test_size
		self.optimizer = optimizer
		self.atom_fea_len = atom_fea_len
		self.h_fea_len = h_fea_len
		self.n_conv = n_conv
		self.n_hp = n_hp
		self.scheduler = scheduler
		self.metric = metric
		self.seed = seed
		self.weights = weights
		self.dropout = dropout
		# self.cuda = not self.disable_cuda and torch.cuda.is_available()
		self.cuda = torch.cuda.is_available()
		print("Cuda enabled: ", self.cuda)


def main(root_dir, save_dir, disable_cuda=True, workers=0, epochs=30,
				start_epoch=0, batch_size=256, lr=0.01, lr_milestones=[100], momentum=0.9,
				weight_decay=0.0, print_freq=10, resume='', train_size=None, val_size=1000,
				test_size=1000, optimizer='SGD', atom_fea_len=64, h_fea_len=128, n_conv=3, n_hp=1,
				print_checkpoints=False, save_checkpoints=False, scheduler='MultiStepLR', metric='mae',
				seed=123, weights=None, dropout=0):
	global args, best_error, USE_WEIGHTED_LOSS, STORE_GRAD
	args = Argument(root_dir, save_dir, disable_cuda=disable_cuda,
			workers=workers, epochs=epochs, start_epoch=start_epoch, batch_size=batch_size, lr=lr,
			lr_milestones=lr_milestones, momentum=momentum, weight_decay=weight_decay, print_freq=print_freq,
			resume=resume, train_size=train_size, val_size=val_size, test_size=test_size, optimizer=optimizer,
			atom_fea_len=atom_fea_len, h_fea_len=h_fea_len, n_conv=n_conv, n_hp=n_hp, weights=weights,
			scheduler=scheduler, metric=metric, seed=seed, dropout=dropout)

	print(vars(args))

	best_error = 1e10
	best_error_vec = None

	# load data
	print("Loading datasets...")
	full_dataset = CIFData(args.root_dir, random_seed=args.seed)

	# build model
	structures, targets, _ = full_dataset[0]
	orig_atom_fea_len = structures[0].shape[-1]
	nbr_fea_len = structures[1].shape[-1]
	n_p = len(targets)
	print("Predicting ", n_p, " properties!!")
	model = MTCGCNN(orig_atom_fea_len, nbr_fea_len,
								atom_fea_len=args.atom_fea_len,
								n_conv=args.n_conv,
								h_fea_len=args.h_fea_len,
								n_p=n_p, n_hp=args.n_hp, dropout=args.dropout)

	if args.cuda:
		model.cuda()

	# set some defaults
	properties_loss_weight = torch.ones(n_p)

	if args.weights is not None:
		USE_WEIGHTED_LOSS = True
		properties_loss_weight = FloatTensor(args.weights)
		print('Using weights: ', properties_loss_weight)

	collate_fn = collate_pool
	# Only training loader needs to be differentiated, val/test only use full dataset
	train_loader, val_loader, test_loader = get_train_val_test_loader(
		dataset=full_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
		train_size=args.train_size, num_workers=args.workers,
		val_size=args.val_size, test_size=args.test_size,
		pin_memory=args.cuda, return_test=True, return_val=True)
	
	# obtain target value normalizer
	if len(full_dataset) < 2000:
		warnings.warn('Dataset has less than 2000 data points. '
			'Lower accuracy is expected. ')
		sample_data_list = [full_dataset[i] for i in tqdm(range(len(full_dataset)))]
	else:
		sample_data_list = [full_dataset[i] for i in
							tqdm(random.sample(range(len(full_dataset)), 2000))]
	_, sample_target, _ = collate_pool(sample_data_list)
	normalizer = Normalizer(sample_target)

	# define loss func and optimizer
	if args.cuda:
		criterion = ModifiedMSELoss().cuda()
	else:
		criterion = ModifiedMSELoss()
	if args.optimizer == 'SGD':
		optimizer = optim.SGD(model.parameters(), args.lr,
							  momentum=args.momentum,
							  weight_decay=args.weight_decay)
	elif args.optimizer == 'Adam':
		optimizer = optim.Adam(model.parameters(), args.lr,
							weight_decay=args.weight_decay)
	else:
		raise NameError('Only SGD or Adam is allowed as optimizer')

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_error = checkpoint['best_error']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			normalizer.load_state_dict(checkpoint['normalizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	if args.scheduler == 'MultiStepLR':
		scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.5)
	elif args.scheduler == 'ReduceLROnPlateau':
		scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.8, verbose=True)

	train_error_vec_per_epoch = []
	val_error_vec_per_epoch = []
	train_loss_vec_per_epoch = []
	val_loss_vec_per_epoch = []
	train_loss_list = []
	train_error_list = []
	val_loss_list = []
	val_error_list = []
	for epoch in range(args.start_epoch, args.epochs):
		# train for one epoch
		[train_error_vec, train_loss_vec] = train(train_loader, model, criterion, optimizer,
							epoch, normalizer, n_p, properties_loss_weight,
							print_checkpoints=print_checkpoints)

		train_loss, train_error = torch.mean(train_loss_vec.avg).item(),\
										torch.mean(train_error_vec.avg).item()
		print('Training Error: %0.3f  Loss: %0.3f' % (train_error, train_loss))
		train_loss_list.append(train_loss)
		train_error_list.append(train_error)

		# evaluate on validation set
		[error, val_error_vec, val_loss_vec] = validate(val_loader, model, criterion,
										normalizer, n_p, properties_loss_weight,
										print_checkpoints=print_checkpoints)

		val_loss, val_error = torch.mean(val_loss_vec.avg).item(),\
										torch.mean(val_error_vec.avg).item()
		val_loss_list.append(val_loss)
		val_error_list.append(val_error)

		if error != error:
			print('Exit due to NaN')
			sys.exit(1)

		if args.scheduler == 'MultiStepLR':
			scheduler.step()
		elif args.scheduler == 'ReduceLROnPlateau': 
			scheduler.step(error)

		# store the error values from previous iteration - useful for plotting
		train_error_vec_per_epoch.append(train_error_vec.avg.cpu().numpy().squeeze())
		val_error_vec_per_epoch.append(val_error_vec.avg.cpu().numpy().squeeze())
		# store the loss values from previous iteration - useful for plotting
		train_loss_vec_per_epoch.append(train_loss_vec.avg.cpu().numpy().squeeze())
		val_loss_vec_per_epoch.append(val_loss_vec.avg.cpu().numpy().squeeze())

		# remember the best error and possibly save checkpoint
		is_best = error < best_error
		if is_best:
			best_error_vec = val_error_vec.avg.squeeze()
		best_error = min(error, best_error)
		
		if save_checkpoints:
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_error': best_error,
				'optimizer': optimizer.state_dict(),
				'normalizer': normalizer.state_dict(),
				'args': vars(args)
			}, is_best)

	# Draw some meaningful plots
	
	if save_checkpoints:
		# Plot1: individual property error vs epoch for all properties
		plotMultiGraph(np.array(train_error_vec_per_epoch), np.array(val_error_vec_per_epoch),
					path=args.save_dir, name='train_val_err_vs_epoch')
		np.savetxt(args.save_dir + 'train_error.txt', np.array(train_error_vec_per_epoch))
		np.savetxt(args.save_dir + 'val_error.txt', np.array(val_error_vec_per_epoch))

		# Plot2: individual property loss vs epoch for all properties
		plotMultiGraph(np.array(train_loss_vec_per_epoch), np.array(val_loss_vec_per_epoch),
					path=args.save_dir, name='train_val_loss_vs_epoch')
		np.savetxt(args.save_dir + 'train_loss.txt', np.array(train_loss_vec_per_epoch))
		np.savetxt(args.save_dir + 'val_loss.txt', np.array(val_loss_vec_per_epoch))

		# Plot3: average loss vs epoch
		plotGraph(train_loss_list, val_loss_list, path=args.save_dir, name='train_val_loss_avg_vs_epoch')
		np.savetxt(args.save_dir + 'train_loss_avg.txt', np.array(train_loss_list))
		np.savetxt(args.save_dir + 'val_loss_avg.txt', np.array(val_loss_list))

		# Plot4: error vs epoch overall
		plotGraph(train_error_list, val_error_list, path=args.save_dir, name='train_val_err_avg_vs_epoch')
		np.savetxt(args.save_dir + 'train_error_avg.txt', np.array(train_error_list))
		np.savetxt(args.save_dir + 'val_error_avg.txt', np.array(val_error_list))

	# test best model using saved checkpoints
	if save_checkpoints:
		print('---------Evaluate Model on Test Set---------------')
		best_checkpoint = torch.load(args.save_dir + 'model_best.pth.tar')
		model.load_state_dict(best_checkpoint['state_dict'])
		[test_error, test_error_vec, test_loss_vec] = validate(test_loader, model, criterion, normalizer, n_p,
									properties_loss_weight, test=True, print_checkpoints=print_checkpoints)
		return best_error.item(), test_error.item(), test_error_vec.avg.cpu().numpy().squeeze(),\
					test_loss_vec.avg.cpu().numpy().squeeze()

	return best_error.item(), None, best_error_vec.cpu().numpy(), None


def train(train_loader, model, criterion, optimizer, epoch,
		normalizer, n_p, properties_loss_weight, print_checkpoints=False):

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	avg_errors = AverageMeter()

	# error_vector is an average error we see per property. Its dim is (1,n_p)
	error_vector = AverageMeter(is_tensor=True, dimensions=[1,n_p])
	# loss_vector is an average loss we see per property. Its dim is (1,n_p)
	loss_vector = AverageMeter(is_tensor=True, dimensions=[1,n_p])

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, targets, _) in enumerate(train_loader):
		# NOTE: here targets is Torch.FloatTensor of dim (batch_size, n_p)

		batch_size = targets.shape[0]
		
		# measure data loading time
		data_time.update(time.time() - end)

		if args.cuda:
			input_var = (Variable(input[0].cuda(async=True)),
						 Variable(input[1].cuda(async=True)),
						 input[2].cuda(async=True),
						 [crys_idx.cuda(async=True) for crys_idx in input[3]])
			targets = targets.cuda(async=True)
		else:
			input_var = (Variable(input[0]),
						 Variable(input[1]),
						 input[2],
						 input[3])
		# normalize target
		targets_normed = normalizer.norm(targets)
		if args.cuda:
			targets_var = Variable(targets_normed.cuda(async=True))
			properties_loss_weight = properties_loss_weight.cuda(async=True)
		else:
			targets_var = Variable(targets_normed)

		# compute output
		output, _ = model(*input_var)

		# Here, both output and targets_var are torch.FloatTensor of dim (batch_size, n_p).
		# Using this, we calculate the loss value per property (each column) and add the losses
		# one by one. This ensures we are able to properly backpropagate the loss in the network.
		# Also, note that, there might be cases where the targets_var contains
		# float('inf') for some missing property value. For all such cases, we just set loss to zero
		# and ignore the predicted output (we have no way to do a loss prediction because we actually
		# have missing data).

		if USE_WEIGHTED_LOSS:
			mse_loss = [properties_loss_weight[i]*criterion(output[:,i], targets_var[:,i]) for i in range(n_p)]
			loss = np.sum(mse_loss) / n_p
		else:
			mse_loss = [criterion(output[:,i], targets_var[:,i]) for i in range(n_p)]# for individual properties
			loss = np.sum(mse_loss) / n_p
		mse_vec = torch.stack(mse_loss).detach()

		# measure accuracy and record loss
		if args.metric == 'mae':
			error = mae(normalizer.denorm(output.data), targets)
		elif args.metric == 'rmse':
			error = torch.sqrt(FloatTensor(mse_loss))
		error_vector.update(error, batch_size)
		loss_vector.update(mse_vec, batch_size)
		avg_error = torch.mean(error)
		losses.update(loss.data.item(), batch_size)
		avg_errors.update(avg_error, batch_size)
		
		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if print_checkpoints:
			if i % args.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'ERR {avg_errors.val:.3f} ({avg_errors.avg:.3f})'.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, avg_errors=avg_errors)
					)
	
	return error_vector, loss_vector


def validate(val_loader, model, criterion, normalizer, n_p, properties_loss_weight,
		test=False, print_checkpoints=False):
	batch_time = AverageMeter()
	losses = AverageMeter()
	avg_errors = AverageMeter()

	# error_vector is an average error we see per property. Its dim is (1,n_p)
	error_vector = AverageMeter(is_tensor=True, dimensions=[1,n_p])
	# loss_vector is an average loss we see per property. Its dim is (1,n_p)
	loss_vector = AverageMeter(is_tensor=True, dimensions=[1,n_p])
	if test:
		test_targets = []
		test_preds = []
		test_cif_ids = []

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input, targets, batch_cif_ids) in enumerate(val_loader):
		batch_size = targets.shape[0]
		with torch.no_grad():
			if args.cuda:
				input_var = (Variable(input[0].cuda(async=True)),
							 Variable(input[1].cuda(async=True)),
							 input[2].cuda(async=True),
							 [crys_idx.cuda(async=True) for crys_idx in input[3]])
				targets = targets.cuda(async=True)
			else:
				input_var = (Variable(input[0]),
							 Variable(input[1]),
							 input[2],
							 input[3])
		targets_normed = normalizer.norm(targets)
		with torch.no_grad():
			if args.cuda:
				targets_var = Variable(targets_normed.cuda(async=True))
				properties_loss_weight = properties_loss_weight.cuda(async=True)
			else:
				targets_var = Variable(targets_normed)

		# compute output
		output, _ = model(*input_var)
		if USE_WEIGHTED_LOSS:
			mse_loss = [properties_loss_weight[i] * criterion(output[:,i], targets_var[:,i]) for i in range(n_p)]
			loss = np.sum(mse_loss) / n_p
		else:
			mse_loss = [criterion(output[:,i], targets_var[:,i]) for i in range(n_p)]	# for individual properties
			loss = np.sum(mse_loss) / n_p
		mse_vec = torch.stack(mse_loss).detach()

		# measure accuracy and record loss
		if args.metric == 'mae':
			error = mae(normalizer.denorm(output.data), targets)
		elif args.metric == 'rmse':
			error = torch.sqrt(FloatTensor(mse_loss))
		error_vector.update(error, batch_size)
		loss_vector.update(mse_vec, batch_size)
		avg_error = torch.mean(error)
		losses.update(loss.data.item(), batch_size)
		avg_errors.update(avg_error, batch_size)
		if test:
			test_pred = normalizer.denorm(output.data)
			test_target = targets
			test_preds += test_pred.tolist()
			test_targets += test_target.tolist()
			test_cif_ids += batch_cif_ids

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if print_checkpoints:
			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'ERR {avg_errors.val:.3f} ({avg_errors.avg:.3f})'.format(
					i, len(val_loader), batch_time=batch_time, loss=losses,
					avg_errors=avg_errors))

	if test:
		star_label = '**'
		with open(args.save_dir + 'test_results.csv', 'w') as f:
			writer = csv.writer(f)
			for cif_id, targets, preds in zip(test_cif_ids, test_targets,
											test_preds):
				writer.writerow((cif_id, targets, preds))
		print('Test error per property:', error_vector.avg.cpu().numpy().squeeze())
		print('Test loss per property:', loss_vector.avg.cpu().numpy().squeeze())
	else:
		star_label = '*'
	print(' {star} ERR {avg_errors.avg:.3f} LOSS {avg_loss.avg:.3f}'.format(star=star_label,
													avg_errors=avg_errors, avg_loss=losses))
	return avg_errors.avg, error_vector, loss_vector


class Normalizer(object):
	"""Normalize a Tensor and restore it later."""
	def __init__(self, tensor):
		"""
		Tensor is taken as a sample to calculate the mean and std.
		The tensor is of dim (N, n_p) where N is the sample size each with n_p columns
		and the normalization is done across a column of values. So, mean is a tensor
		of dim (n_p)
		"""
		self.columns = tensor.shape[1]		# =n_i
		self.mean = FloatTensor([torch.mean(tensor[:,i]) for i in range(self.columns)])
		self.std = FloatTensor([torch.std(tensor[:,i]) for i in range(self.columns)])

	def norm(self, tensor):
		return (tensor - self.mean) / self.std

	def denorm(self, normed_tensor):
		return normed_tensor * self.std + self.mean

	def state_dict(self):
		return {'mean': self.mean,
				'std': self.std}

	def load_state_dict(self, state_dict):
		self.mean = state_dict['mean']
		self.std = state_dict['std']


def sanitize(input_var, reference, return_diff=False):
	"""
	Given two tensor/Variable vectors [dim (k)], sanitize the input_var and reference to zero out the indices
	where the reference has inf. If there are no inf values, return the vectors as is.
	The return_diff is basically a lazy hack to vectorize the mae calculation (Nothing fancy).
	"""
	# find indices where float('inf') is present. To facilitate this, clamp to some high values first
	reference = torch.clamp(reference, max=UNDEFINED_INF)
	idx = (reference == UNDEFINED_INF).nonzero()
	idx = idx.view(-1)
	# if idx is valid (i.e. there is some inf values),
	# then replace these indices with zero in both the tensors
	input_var_c = input_var.clone()
	if len(idx):
		reference.index_fill_(0, idx, 0)
		input_var_c.index_fill_(0, idx, 0)		
	if return_diff:
		return input_var_c - reference
	else:
		return input_var_c, reference


def mae(prediction, target):
	"""
	Computes the mean absolute error between prediction and target. If target has float('inf')
	then modifies the calculation accordingly to avoid that

	Parameters
	----------

	prediction: torch.Tensor (N, n_p)
	target: torch.Tensor (N, n_p)

	Returns
	-------
	torch.Tensor (n_p)
	"""
	n_p = target.shape[1]
	return FloatTensor([torch.sum(torch.abs(sanitize(prediction[:,i], target[:,i], return_diff=True)))\
				for i in range(n_p)])/target.shape[0]


class ModifiedMSELoss(torch.nn.Module):
	"""
	Given the output and target Variables (each of dim (N, 1)) this finds the
	MSE of these variables.
	"""
	# The modification is to ignore the specific rows which
	# have float('inf'). For those rows, just set the value to be zero
	
	def __init__(self):
		super(ModifiedMSELoss,self).__init__()
		
	def forward(self, output, target):
		[output_c, target] = sanitize(output, target)
		# now return the MSE Loss
		loss = nn.MSELoss()
		return loss(output_c, target)


class AverageMeter(object):
	"""
	Computes and stores the average and current value. Accomodates both numbers and tensors.
	If the input to be monitored is a tensor, also need the dimensions/shape of the tensor.
	Also, for tensors, it keeps a column wise count for average, sum etc.
	"""
	def __init__(self, is_tensor=False, dimensions=None):
		if is_tensor and dimensions is None:
			print("Bad definition of AverageMeter!")
			sys.exit(1)
		self.is_tensor = is_tensor
		self.dimensions = dimensions
		self.reset()

	def reset(self):
		self.count = 0
		if self.is_tensor:
			self.val = torch.zeros(self.dimensions).type(FloatTensor)
			self.avg = torch.zeros(self.dimensions).type(FloatTensor)
			self.sum = torch.zeros(self.dimensions).type(FloatTensor)
		else:
			self.val = 0
			self.avg = 0
			self.sum = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, args.save_dir + filename)
	if is_best:
		print('Saving best parameters...')
		shutil.copyfile(args.save_dir + filename, args.save_dir + 'model_best.pth.tar')
