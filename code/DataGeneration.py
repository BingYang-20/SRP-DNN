""" 
    Function:   Generate data
	Usage: 		Need to specify stage(etrain, train, test), gpu-id, sources, source_state, data-op, data-val
"""

import os
import argparse


parser = argparse.ArgumentParser(description='Generating multi-channel audio signals')
parser.add_argument('--stage', type=str, default='train', metavar='Stage', help='stage that generated data used for (default: Pretrain)') # ['pretrain', 'preval', 'train', 'val', 'test']
parser.add_argument('--gpu-id', type=str, default='7', metavar='GPU', help='GPU ID (default: 7)')
parser.add_argument('--sources', type=int, nargs='+', default=[1, 2], metavar='Sources', help='number of sources (default: 1, 2)')
parser.add_argument('--source-state', type=str, default='mobile', metavar='SourceState', help='state of sources (default: Mobile)')
parser.add_argument('--data-op', type=str, default='save_sig', metavar='DataOp', help='operation for generated data (default: Save signal)') # ['save_sig', 'save_RIR', 'read_sig', 'read_RIR']
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

from OptSRPDNN import opt
from utils import set_seed, save_file, load_file
from Dataset import Parameter
import Dataset as at_dataset

opts = opt()
dirs = opts.dir()

if (args.data_op == 'save_sig') | (args.data_op == 'save_RIR'):
	# %% Dataset
	if args.stage == 'train':
		data_num = 102400
		set_seed(10000)

	elif args.stage == 'val':
		data_num = 5120
		set_seed(10001)

	elif args.stage == 'test':
		data_num = 5120
		set_seed(10002)

	else:
		raise Exception('Stage unrecognized!')

	speed = 343.0	
	fs = 16000
	T = 4.112  # Trajectory length (s) 2.064
	if args.source_state == 'static':
		traj_points = 1 # number of RIRs per trajectory
	elif args.source_state == 'mobile':
		traj_points = int(10*T) # number of RIRs per trajectory
	else:
		raise Exception('Source state mode unrecognized~')

	# Array
	array = '12ch'
	if array == '2ch':
		array_setup = at_dataset.dualch_array_setup
	elif array == '12ch':
		array_setup = at_dataset.benchmark2_array_setup

	# Source signal
	sourceDataset = at_dataset.LibriSpeechDataset(
		path = dirs['sousig_'+args.stage], 
		T = T, 
		fs = fs, 
		num_source = max(args.sources), 
		return_vad = True, 
		clean_silence = True)

	# Noise signal
	noiseDataset = at_dataset.NoiseDataset(
		T = T, 
		fs = fs, 
		nmic = array_setup.mic_pos.shape[0], 
		noise_type = Parameter(['diffuse'], discrete=True), 
		noise_path = dirs['noisig_'+args.stage], 
		c = speed)

	# Room acoustics
	dataset = at_dataset.RandomMicSigDataset( 
		sourceDataset = sourceDataset,
		num_source = Parameter(args.sources, discrete=True),
		source_state = args.source_state,
		room_sz = Parameter([3,3,2.5], [10,8,6]),
		T60 = Parameter(0.1, 1.0),
		abs_weights = Parameter([0.5]*6, [1.0]*6),
		array_setup = array_setup,
		array_pos = Parameter([0.1,0.1,0.1], [0.9,0.9,0.5]),
		noiseDataset = noiseDataset,
		SNR = Parameter(0, 25),
		nb_points = traj_points, 
		dataset_sz = data_num,
		c = speed, 
		transforms = None,
		return_acoustic_scene = True,
		save_src_noi = False,
		)

	# Data generation
	if args.data_op == 'save_sig':
		save_dir = dirs['sensig_'+args.stage]
		exist_temp = os.path.exists(save_dir)
		if exist_temp==False:
			os.makedirs(save_dir)
			print('make dir: ' + save_dir)
		for idx in range(data_num):
			if idx%1024==0:
				print(int(idx/1024), 'K')
			mic_signals, acoustic_scene = dataset[idx]  
			sig_path = save_dir + '/' + str(idx) + '.wav'
			acous_path = save_dir + '/' + str(idx) + '.npz'
			save_file(mic_signals, acoustic_scene, sig_path, acous_path)

	# elif args.data_op == 'save_RIR':
	# 	save_dir = dirs['rir_'+args.stage]
	# 	exist_temp = os.path.exists(save_dir)
	# 	if exist_temp==False:
	# 		os.makedirs(save_dir)
	# 		print('make dir: ' + save_dir)
	# 	for idx in range(data_num):
	# 		if idx%1024==0:
	# 			print(int(idx/1024), 'K')
	# 		mic_signals, acoustic_scene = dataset[idx]   
	# 		delattr(acoustic_scene, 'SNR')
	# 		delattr(acoustic_scene, 'noise_signal')
	# 		delattr(acoustic_scene, 'source_signal')
	# 		delattr(acoustic_scene, 'timestamps')
	# 		delattr(acoustic_scene, 't')
	# 		delattr(acoustic_scene, 'trajectory')
	# 		acous_path = save_dir + '/' + str(idx) + '.npz'
	# 		save_file(mic_signals, acoustic_scene, sig_path=None, acous_path=acous_path)
		

elif (args.data_op == 'read_sig') | ( args.data_op == 'read_RIR'):
	acoustic_scene = at_dataset.AcousticScene(	
				room_sz = [],
				T60 = [],
				beta = [],
				RIR = [],
				array_setup = [],
				mic_pos = [],
				array_pos = [],
				noise_signal = [],
				SNR = [],
				source_signal = [],
				fs = [],
				timestamps = [],
				traj_pts = [],
				t = [],
				trajectory = [],
				# DOA = [],
				c = []
			)

	if args.data_op == 'read_sig':
		sig_path = dirs['sensig_val'] + '/' + '0.wav'
		acous_path = dirs['sensig_val'] + '/' + '0.npz'
		mic_signal, acoustic_scene = load_file(acoustic_scene, sig_path, acous_path)
		for i in acoustic_scene.__dict__.keys():
			print(i, acoustic_scene.__dict__[i])
		print(acoustic_scene.RIR[0].shape)
	
	# elif args.data_op == 'read_RIR':
	# 	delattr(acoustic_scene, 'SNR')
	# 	delattr(acoustic_scene, 'noise_signal')
	# 	delattr(acoustic_scene, 'source_signal')
	# 	delattr(acoustic_scene, 'timestamps')
	# 	delattr(acoustic_scene, 't')
	# 	delattr(acoustic_scene, 'trajectory')
	# 	acous_path = dirs['rir_train'] + '/' + '0.npz'
	# 	acoustic_scene = load_file(acoustic_scene, sig_path=None, acous_path=acous_path)
	# 	for i in acoustic_scene.__dict__.keys():
	# 		print(i, acoustic_scene.__dict__[i])
	# 	print(acoustic_scene.RIR[0].shape)
	

# if __name__ == '__main__':
    # pass

