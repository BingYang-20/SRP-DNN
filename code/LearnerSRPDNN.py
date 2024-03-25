import numpy as np
import torch
from copy import deepcopy
from BaseLearner import Learner
import Module as at_module
from utils import forgetting_norm

class SourceTrackingFromSTFTLearner(Learner):
	""" Learner for models which use STFTs of multiple channels as input
	"""
	def __init__(self, model, win_len, win_shift_ratio, nfft, fre_used_ratio, nele, nazi, rn, fs, ch_mode, tar_useVAD, localize_mode, c=343.0): #, arrayType='planar', cat_maxCoor=False, apply_vad=False):
		""" 
		fre_used_ratio - the ratio between used frequency and valid frequency
		"""
		super().__init__(model)

		self.nele = nele
		self.nazi = nazi

		self.nfft = nfft
		if fre_used_ratio == 1:
			self.fre_range_used = range(1, int(self.nfft/2*fre_used_ratio)+1, 1)
		elif fre_used_ratio == 0.5:
			self.fre_range_used = range(0, int(self.nfft/2*fre_used_ratio), 1)
		else:
			raise Exception('Prameter fre_used_ratio unexpected')

		self.dostft = at_module.STFT(win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft)
		fre_max = fs / 2
		self.ch_mode = ch_mode
		self.gerdpipd = at_module.DPIPD(ndoa_candidate=[nele, nazi], mic_location=rn, nf=int(self.nfft/2) + 1, fre_max=fre_max, 
										ch_mode=self.ch_mode, speed=c)
		self.tar_useVAD = tar_useVAD
		self.addbatch = at_module.AddChToBatch(ch_mode=self.ch_mode)
		self.removebatch = at_module.RemoveChFromBatch(ch_mode=self.ch_mode)
		self.sourcelocalize = at_module.SourceDetectLocalize(max_num_sources=int(localize_mode[2]), source_num_mode=localize_mode[1], meth_mode=localize_mode[0])
		
		self.getmetric = at_module.getMetric(source_mode='multiple')

	def data_preprocess(self, mic_sig_batch=None, gt_batch=None, vad_batch=None, eps=1e-6):

		data = []
		if mic_sig_batch is not None:
			mic_sig_batch = mic_sig_batch.to(self.device)
			
			stft = self.dostft(signal=mic_sig_batch) # (nb,nf,nt,nch)
			stft = stft.permute(0, 3, 1, 2)  # (nb,nch,nf,nt)

			nor_flag = 'online'
			if nor_flag=='offline':
				## offline normalization
				mag1 = torch.abs(stft[:, 0:1, :, :])
				mean_value = torch.mean(mag1.reshape(mag1.shape[0], -1), dim=1)
				mean_value = mean_value[:, np.newaxis, np.newaxis, np.newaxis].expand(mag1.shape)
				stft = stft/(mean_value+eps)
			elif nor_flag=='online':
				## online normalization
				mag = torch.abs(stft)
				mean_value = forgetting_norm(mag, num_frame_set=256) # 256 is number of time frames used for training
				mean_value = mean_value.expand(mag.shape)
				stft = stft/(mean_value+eps)

			# change batch (nb,nch,nf,nt)→(nb*(nch-1),2,nf,nt)/(nb*(nch-1)*nch/2,2,nf,nt)
			stft_rebatch = self.addbatch(stft)

			# prepare model input
			mag = torch.abs(stft_rebatch)
			mag = torch.log10(mag+eps)
			phase = torch.angle(stft_rebatch) # [-pi/2, pi/2]
			
			magphase_batch = torch.cat((mag, phase), dim=1) # (nb*(nch-1),4,nf,nt)
			data += [magphase_batch[:,:,self.fre_range_used,:]]

		if gt_batch is not None:
			DOAw_batch = gt_batch['doa']
			vad_batch = gt_batch['vad_sources']

			source_doa = DOAw_batch.cpu().numpy()  
			_, ipd_batch, _ = self.gerdpipd(source_doa=source_doa)
			ipd_batch = np.concatenate((ipd_batch.real[:,:,self.fre_range_used,:,:], ipd_batch.imag[:,:,self.fre_range_used,:,:]), axis=2).astype(np.float32) # (nb, ntime, 2nf, nmic-1, nsource)
			ipd_batch = torch.from_numpy(ipd_batch)

			vad_batch = vad_batch.mean(axis=2).float() # (nb,nseg,nsource) # s>2/3 

			DOAw_batch = DOAw_batch.to(self.device) # (nb,nseg,2,nsource)
			ipd_batch = ipd_batch.to(self.device)
			vad_batch = vad_batch.to(self.device)

			if self.tar_useVAD:
				nb, nt, nf, nmic, num_source = ipd_batch.shape
				th = 0
				vad_batch_copy = deepcopy(vad_batch)
				vad_batch_copy[vad_batch_copy<th] = th
				vad_batch_expand = vad_batch_copy[:, :, np.newaxis, np.newaxis, :].expand(nb, nt, nf, nmic, num_source)
				ipd_batch = ipd_batch * vad_batch_expand
			ipd_batch = torch.sum(ipd_batch, dim=-1)  # (nb,nseg,2nf,nmic-1)

			gt_batch['doa'] = DOAw_batch
			gt_batch['ipd'] = ipd_batch
			gt_batch['vad_sources'] = vad_batch

			data += [gt_batch]

		return data 

	def loss(self, pred_batch=None, gt_batch=None):
		"""	Function: Traning loss
			Args:
				pred_batch: ipd
				gt_batch: dict{'ipd'}
			Returns:
				loss
        """
		pred_ipd = pred_batch
		gt_ipd = gt_batch['ipd']
		nb, _, _, _ = gt_ipd.shape # (nb, nt, nf, nmic)

		pred_ipd_rebatch = self.removebatch(pred_ipd, nb).permute(0, 2, 3, 1)

		loss = torch.nn.functional.mse_loss(pred_ipd_rebatch.contiguous(), gt_ipd.contiguous())

		return loss

	def predgt2DOA(self, pred_batch=None, gt_batch=None, time_pool_size=None):
		"""	Function: Conert IPD vector to DOA
			Args:
				pred_batch: ipd
				gt_batch: dict{'doa', 'vad_sources', 'ipd'}
			Returns:
				pred_batch: dict{'doa', 'spatial_spectrum'}
				gt_batch: dict{'doa', 'vad_sources', 'ipd'}
	    """

		if pred_batch is not None:
			
			pred_ipd = pred_batch.detach()
			dpipd_template, _, doa_candidate = self.gerdpipd( ) # (nele, nazi, nf, nmic)

			_, _, _, nmic = dpipd_template.shape
			nbnmic, nt, nf = pred_ipd.shape
			nb = int(nbnmic/nmic)

			dpipd_template = np.concatenate((dpipd_template.real[:,:,self.fre_range_used,:], dpipd_template.imag[:,:,self.fre_range_used,:]), axis=2).astype(np.float32) # (nele, nazi, 2nf, nmic-1)
			dpipd_template = torch.from_numpy(dpipd_template).to(self.device) # (nele, nazi, 2nf, nmic)

			# for azimuth estimation in half horizontal plane only
			# nele, nazi, _, _ = dpipd_template.shape
			# dpipd_template = dpipd_template[int((nele-1)/2):int((nele-1)/2)+1, int((nazi-1)/2):nazi, :, :]
			# doa_candidate[0] = np.linspace(np.pi/2, np.pi/2, 1)
			# doa_candidate[1] = np.linspace(0, np.pi, 37)

			# rebatch from (nb*nmic, nt, 2nf) to (nb, nt, 2nf, nmic)
			pred_ipd_rebatch = self.removebatch(pred_ipd, nb).permute(0, 2, 3, 1) # (nb, nt, 2nf, nmic)
			if time_pool_size is not None:
				nt_pool = int(nt / time_pool_size)
				ipd_pool_rebatch = torch.zeros((nb, nt_pool, nf, nmic), dtype=torch.float32, requires_grad=False).to(self.device)  # (nb, nt_pool, 2nf, nmic-1)
				for t_idx in range(nt_pool):
					ipd_pool_rebatch[:, t_idx, :, :]  = torch.mean(
					pred_ipd_rebatch[:, t_idx*time_pool_size: (t_idx+1)*time_pool_size, :, :], dim=1)
				pred_ipd_rebatch = deepcopy(ipd_pool_rebatch)
				nt = deepcopy(nt_pool)
			
			pred_DOAs, pred_VADs, pred_ss = self.sourcelocalize(pred_ipd=pred_ipd_rebatch, dpipd_template=dpipd_template, doa_candidate=doa_candidate)
			pred_batch = {}
			pred_batch['doa'] = pred_DOAs
			pred_batch['vad_sources'] = pred_VADs
			pred_batch['spatial_spectrum'] = pred_ss

		if gt_batch is not None: 
			for key in gt_batch.keys():
				gt_batch[key] = gt_batch[key].detach()

		return pred_batch, gt_batch 

	def evaluate(self, pred, gt, metric_setting={'ae_mode':['azi'], 'ae_TH':30, 'useVAD':True, 'vad_TH':[2/3, 0.3], 'metric_unfold':False} ):
		""" Function: Evaluate DOA estimation results
			Args:
				pred 	- dict{'doa', 'vad_sources'}
				gt 		- dict{'doa', 'vad_sources'}
							doa (nb, nt, 2, nsources) in radians
							vad (nb, nt, nsources) binary values
			Returns:
				metric	- dict or list
        """
		doa_gt = gt['doa'] * 180 / np.pi 
		doa_pred = pred['doa'] * 180 / np.pi 
		vad_gt = gt['vad_sources']  
		vad_pred = pred['vad_sources'] 

		# single source 
		# metric = self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode = ae_mode, ae_TH=ae_TH, useVAD=False, vad_TH=vad_TH, metric_unfold=Falsemetric_unfold)

		# multiple source
		metric = \
			self.getmetric(doa_gt, vad_gt, doa_pred, vad_pred, 
				ae_mode = metric_setting['ae_mode'], ae_TH=metric_setting['ae_TH'], 
				useVAD=metric_setting['useVAD'], vad_TH=metric_setting['vad_TH'], 
				metric_unfold=metric_setting['metric_unfold'])

		return metric