from main import main
from utils import Logger
import math, time, datetime, os, sys, csv, argparse

def init(root_dir, save_dir, disable_cuda=False, epochs=30, workers=0, scheduler='MultiStepLR',
				start_epoch=0, batch_size=64, lr=0.01, lr_milestones=[200, 400, 600, 800], momentum=0.9,
				weight_decay=0.0, print_freq=10, resume='', train_size=None, val_size=1000,
				test_size=1000, optimizer='SGD', atom_fea_len=64, h_fea_len=32, n_conv=3, n_hp=1,
				seed=123, metric='mae', weights=None, dropout=0):
	return main(root_dir, save_dir, disable_cuda=disable_cuda, workers=workers, epochs=epochs, metric = metric,
		start_epoch=start_epoch, batch_size=batch_size, lr=lr, lr_milestones=lr_milestones, seed=seed,
		momentum=momentum, weight_decay=weight_decay, print_freq=print_freq, resume=resume,
		train_size=train_size, val_size=val_size, test_size=test_size, optimizer=optimizer, weights=weights,
		atom_fea_len=atom_fea_len, h_fea_len=h_fea_len, n_conv=n_conv, n_hp=n_hp,
		print_checkpoints=True, save_checkpoints=True, scheduler=scheduler, dropout=dropout)

if __name__ == '__main__':
	SAVE_RESULTS = True

	FILEPATH = 'data/sample/'
	# NOTE: the path should be valid i.e it should be '..results/' and not 'results'. Slashes are important these days
	SAVEPATH = FILEPATH + 'results/'

	parser = argparse.ArgumentParser(description='Input integer')
	parser.add_argument('--idx', type=str, help='A counter for the program. This helps in running multiple runs of the experiment with the\
			same input data and yet store the results in different folders', default='0')
	args = parser.parse_args()
	# This is done so that we have a provision to save results uniquely for independent runs
	SAVEPATH = SAVEPATH + args.idx + '/'
	if not os.path.exists(SAVEPATH):
		os.makedirs(SAVEPATH)

	# Wtires to file and also to terminal
	sys.stdout = Logger(SAVEPATH)

	file = open(FILEPATH + '/id_prop.csv', 'r')
	TOTAL_SIZE = sum(1 for line in file)
	file.seek(0)
	num_properties = len(file.readline().rstrip().split(',')) - 1	# Number of properties being predicted
	print('Length of dataset: ', TOTAL_SIZE)

	train_size = math.floor(0.6 * TOTAL_SIZE)
	val_size = math.floor(0.2 * TOTAL_SIZE)
	test_size = TOTAL_SIZE - train_size - val_size

	start = time.time()
	print(datetime.datetime.now())

	RANDOM_SPLITS = [887, 212, 136, 998, 975]

	# Set the params here
	step = 0.01
	decay = 0
	conv_layer = 1
	n_hp = 1
	epochs = 30
	optimizer = 'Adam'
	workers = 0
	metric = 'mae'
	scheduler = 'MultiStepLR'
	batch_size = 256
	atom_fea_len = 64
	h_fea_len = 32
	weights = None
	dropout = 0
	seed = RANDOM_SPLITS[int(args.idx) % 5] # random.randint(0,1000)

	# scheduler can be None, ReduceLROnPlateau, MultiStepLR
	best_error, test_error, test_error_vec, test_loss_vec = init(FILEPATH, SAVEPATH,
					train_size=train_size, epochs=epochs, val_size=val_size, test_size=test_size,
					optimizer=optimizer, n_conv=conv_layer, n_hp=n_hp, lr=step, weight_decay=decay,
					workers=workers, metric=metric, scheduler=scheduler, batch_size=batch_size,
					atom_fea_len=atom_fea_len, h_fea_len=h_fea_len, seed=seed, weights=weights, dropout=dropout)
	if SAVE_RESULTS:
		with open(FILEPATH + "test_results.log", 'a') as f:
				writer = csv.writer(f)
				if num_properties > 1:
					writer.writerow((test_error, '-'.join(map(str, test_error_vec)),\
							 '-'.join(map(str, test_loss_vec))))
				else:
					writer.writerow((test_error, str(test_error_vec), str(test_loss_vec)))
	print("Best Error is: ", best_error)
	finish = time.time()
	print(datetime.datetime.now())
	print("Time taken ", finish - start )
