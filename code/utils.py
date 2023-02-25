import torch
import numpy as np
import torch
import random 
import pickle
import soundfile 
from copy import deepcopy

## for spherical coordinates

def cart2sph(cart, include_r=False):
	""" Cartesian coordinates to spherical coordinates conversion.
	Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
	where the radius is optional according to the include_r argument.
	"""
	r = torch.sqrt(torch.sum(torch.pow(cart, 2), dim=-1))
	theta = torch.acos(cart[..., 2] / r)
	phi = torch.atan2(cart[..., 1], cart[..., 0])
	if include_r:
		sph = torch.stack((theta, phi, r), dim=-1)
	else:
		sph = torch.stack((theta, phi), dim=-1)
	return sph


def sph2cart(sph):
	""" Spherical coordinates to cartesian coordinates conversion.
	Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
	where the radius is supposed to be 1 if it is not included.
	"""
	if sph.shape[-1] == 2: sph = torch.cat((sph, torch.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
	x = sph[..., 2] * torch.sin(sph[..., 0]) * torch.cos(sph[..., 1])
	y = sph[..., 2] * torch.sin(sph[..., 0]) * torch.sin(sph[..., 1])
	z = sph[..., 2] * torch.cos(sph[..., 0])
	return torch.stack((x, y, z), dim=-1)


## for training process 

def set_seed(seed):
	""" Function: fix random seed.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False # avoid-CUDNN_STATUS_NOT_SUPPORTED #(commont if use cpu??)

	np.random.seed(seed)
	random.seed(seed)

def set_random_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

def get_learning_rate(optimizer):
    """ Function: get learning rates from optimizer
    """ 
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def set_learning_rate(epoch, lr_init, step, gamma):
    """ Function: adjust learning rates 
    """ 
    lr = lr_init*pow(gamma, int(epoch/step))
    return lr

## for data number

def detect_infnan(data, mode='torch'):
    """ Function: check whether there is inf/nan in the element of data or not
    """ 
    if mode == 'troch':
        inf_flag = torch.isinf(data)
        nan_flag = torch.isnan(data)
    elif mode == 'np':
        inf_flag = np.isinf(data)
        nan_flag = np.isnan(data)
    else:
        raise Exception('Detect infnan mode unrecognized')
    if (True in inf_flag):
        raise Exception('INF exists in data')
    if (True in nan_flag):
        raise Exception('NAN exists in data')


## for room acoustic data saving and reading 

def save_file(mic_signal, acoustic_scene, sig_path, acous_path):
    
    if sig_path is not None:
        soundfile.write(sig_path, mic_signal, acoustic_scene.fs)

    if acous_path is not None:
        file = open(acous_path,'wb')
        file.write(pickle.dumps(acoustic_scene.__dict__))
        file.close()

def load_file(acoustic_scene, sig_path, acous_path):

    if sig_path is not None:
        mic_signal, fs = soundfile.read(sig_path)

    if acous_path is not None:
        file = open(acous_path,'rb')
        dataPickle = file.read()
        file.close()
        acoustic_scene.__dict__ = pickle.loads(dataPickle)

    if (sig_path is not None) & (acous_path is not None):
        return mic_signal, acoustic_scene
    elif (sig_path is not None) & (acous_path is None):
        return mic_signal
    elif (sig_path is None) & (acous_path is not None):
        return acoustic_scene

def forgetting_norm(input, num_frame_set=None):
    """
        Function: Using the mean value of the near frames to normalization
        Args:
            input: feature [B, C, F, T]
            num_frame_set: length of the training time frames, used for calculating smooth factor
        Returns:
            normed feature
        Ref: Online Monaural Speech Enhancement using Delayed Subband LSTM, INTERSPEECH, 2020
    """
    assert input.ndim == 4
    batch_size, num_channels, num_freqs, num_frames = input.size()
    input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

    if num_frame_set == None:
        num_frame_set = deepcopy(num_frames)

    mu = 0
    mu_list = []
    for frame_idx in range(num_frames):
        if num_frames<=num_frame_set:
            alpha = (frame_idx - 1) / (frame_idx + 1)
        else:
            alpha = (num_frame_set - 1) / (num_frame_set + 1)
        current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(batch_size, 1) # [B, 1]
        mu = alpha * mu + (1 - alpha) * current_frame_mu
        mu_list.append(mu)
    mu = torch.stack(mu_list, dim=-1) # [B, 1, T]
    output = mu.reshape(batch_size, 1, 1, num_frames)

    return output
