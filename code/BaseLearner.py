import os
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from abc import ABC, abstractmethod
from tqdm import tqdm, trange


class Learner(ABC):
	def __init__(self, model):
		self.model = model
		self.max_score = -np.inf
		self.use_amp = False
		self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
		self.start_epoch = 1
		super().__init__()

	def mul_gpu(self):
		""" Function: Use multiple GPUs.
		"""
		self.model = torch.nn.DataParallel(self.model) 
		# When multiple gpus are used, 'module.' is added to the name of model parameters. 
		# So whether using one gpu or multiple gpus should be consistent for model traning and checkpoints loading.

	def cuda(self):
		""" Function: Move the model to the GPU and perform the training and inference there.
		"""
		self.model.cuda()
		self.device = "cuda"

	def cpu(self):
		""" Function: Move the model back to the CPU and perform the training and inference here.
		"""
		self.model.cpu()
		self.device = "cpu"

	def amp(self):
		""" Function: Use Automatic Mixed Precision to train network.
		"""
		self.use_amp = True
		self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

	@abstractmethod
	def data_preprocess(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
		""" To be implemented in each learner according to input of their models
		"""
		pass

	@abstractmethod
	def predgt2DOA(self, pred_batch=None, gt_batch=None):
		""" To be implemented in each learner according to output of their models
	    """
		pass

	@abstractmethod
	def loss(self, pred_batch, gt_batch):
		""" To be implemented in each learner according to output of their models
        """
		pass

	@abstractmethod
	def evaluate(self, pred, gt):
		""" To be implemented in each learner according to output of their models
        """
		pass

	def train_epoch(self, dataset, lr=0.0001, epoch=None, return_metric=False):
		""" Function: Train the model with an epoch of the dataset.
		"""

		avg_loss = 0
		avg_beta = 0.99

		self.model.train() 
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		loss = 0
		if return_metric: 
			metric = {}

		optimizer.zero_grad()
		pbar = tqdm(enumerate(dataset), total=len(dataset), leave=False)

		for batch_idx, (mic_sig_batch, gt_batch) in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch))

			in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

			with torch.cuda.amp.autocast(enabled=self.use_amp):
				pred_batch = self.model(in_batch)
				loss_batch = self.loss(pred_batch = pred_batch, gt_batch = gt_batch)

			# add up gradients until optimizer.zero_grad(), multiply a scale to gurantee the gradients equal to that when trajectories_per_gpu_call = trajectories_per_batch
			if self.use_amp:
				self.scaler.scale(loss_batch).backward()
				self.scaler.step(optimizer)
				self.scaler.update()
			else:
				loss_batch.backward()
				optimizer.step()

			optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss_batch.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (batch_idx + 1)))
			# pbar.set_postfix(loss=loss.item())
			pbar.update()

			loss += loss_batch.item()

			if return_metric: 
				pred_batch, gt_batch = self.predgt2DOA(pred_batch = pred_batch, gt_batch = gt_batch)
				metric_batch = self.evaluate(pred=pred_batch, gt=gt_batch)
				if batch_idx==0:
					for m in metric_batch.keys():
						metric[m] = 0
				for m in metric_batch.keys():
					metric[m] += metric_batch[m].item()

		loss /= len(pbar)
		if return_metric: 
			for m in metric_batch.keys():
				metric[m] /= len(pbar)

		if return_metric: 
			return loss, metric
		else:
			return loss
	
	def test_epoch(self, dataset, return_metric=False):
		""" Function: Test the model with an epoch of the dataset.
		"""
		self.model.eval()
		with torch.no_grad():
			loss = 0
			idx = 0
			if return_metric: 
				metric = {}

			for mic_sig_batch, gt_batch in dataset:

				in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

				with torch.cuda.amp.autocast(enabled=self.use_amp):
					pred_batch = self.model(in_batch)
					loss_batch = self.loss(pred_batch=pred_batch, gt_batch=gt_batch)

				loss += loss_batch.item()

				if return_metric: 
					pred_batch, gt_batch = self.predgt2DOA(pred_batch=pred_batch, gt_batch=gt_batch)
					metric_batch = self.evaluate(pred=pred_batch, gt=gt_batch)
					if idx==0:
						for m in metric_batch.keys():
							metric[m] = 0
					for m in metric_batch.keys():
						metric[m] += metric_batch[m].item()
					idx = idx+1

			loss /= len(dataset)
			if return_metric: 
				for m in metric_batch.keys():
					metric[m] /= len(dataset)

			if return_metric: 
				return loss, metric
			else:
				return loss

	def predict_batch(self, gt_batch, mic_sig_batch, wDNN=True):
		""" Function: Predict 
		    Args:
			    mic_sig_batch   - (nb, nsample, nch)
			    gt_batch        - dict{}
		    Returns:
			    pred_batch		- dict{}
			    gt_batch		- dict{}
			    mic_sig_batch	- (nb, nsample, nch)
		"""
		self.model.eval()
		with torch.no_grad():
			
			mic_sig_batch = mic_sig_batch.to(self.device)
			in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

			if wDNN:
				with torch.cuda.amp.autocast(enabled=self.use_amp):
					pred_batch = self.model(in_batch)
				pred_batch, gt_batch = self.predgt2DOA(pred_batch=pred_batch, gt_batch=gt_batch)
			else:
				nt_ori = in_batch.shape[-1]
				nt_pool = gt_batch['doa'].shape[1]
				time_pool_size = int(nt_ori/nt_pool)
				phase = in_batch[:, int(in_batch.shape[1]/2):, :, :].detach() # (nb*nmic_pair, 2, nf, nt)
				phased = phase[:,0,:,:] - phase[:,1,:,:]
				pred_batch = torch.cat((torch.cos(phased), torch.sin(phased)), dim=1).permute(0, 2, 1) # (nb*nmic_pair, nt, 2nf)
				pred_batch, gt_batch = self.predgt2DOA(pred_batch=pred_batch, gt_batch=gt_batch, time_pool_size=time_pool_size)

			return pred_batch, gt_batch, mic_sig_batch


	def predict(self, dataset, wDNN=True, return_predgt=False, metric_setting=None):
		""" Function: Predict 
		    Args:
			    dataset
			    wDNN
			    return_predgt
			    metric_setting 
		    Returns:
			    data
		"""
		data = []
			
		self.model.eval()
		with torch.no_grad():
			idx = 0

			if return_predgt:
				pred = []
				gt = []
				mic_sig = []
			if metric_setting is not None:
				metric = {}

			for mic_sig_batch, gt_batch in dataset:
				print('Dataloading: ' + str(idx+1))
				pred_batch, gt_batch, mic_sig_batch = self.predict_batch(gt_batch, mic_sig_batch, wDNN)

				if (metric_setting is not None):
					metric_batch = self.evaluate(pred=pred_batch, gt=gt_batch)
				if return_predgt:
					pred += [pred_batch]
					gt += [gt_batch]
					mic_sig += [mic_sig_batch]
				if metric_setting is not None:
					for m in metric_batch.keys():
						if idx==0:
							metric[m] = deepcopy(metric_batch[m])
						else:
							metric[m] = torch.cat((metric[m], metric_batch[m]), axis=0)

				idx = idx+1
				
			if return_predgt:
				data += [pred, gt]
				data += [mic_sig]
			if metric_setting is not None:
				data += [metric]
			return data

	def is_best_epoch(self, current_score):
		""" Function: Check if the current model got the best metric score
        """
		if current_score >= self.max_score:
			self.max_score = current_score
			is_best_epoch = True
		else:
			is_best_epoch = False

		return is_best_epoch

	def save_checkpoint(self, epoch, checkpoints_dir, is_best_epoch = False):
		""" Function: Save checkpoint to "checkpoints_dir" directory, which consists of:
            - the epoch number
            - the best metric score in history
            - the optimizer parameters
            - the model parameters
        """
        
		print(f"\t Saving {epoch} epoch model checkpoint...")
		if self.use_amp:
			state_dict = {
				"epoch": epoch,
				"max_score": self.max_score,
				# "optimizer": self.optimizer.state_dict(),
				"scalar": self.scaler.state_dict(), 
				"model": self.model.state_dict()
			}
		else:
			state_dict = {
				"epoch": epoch,
				"max_score": self.max_score,
				# "optimizer": self.optimizer.state_dict(),
				"model": self.model.state_dict()
			}

		torch.save(state_dict, checkpoints_dir + "/latest_model.tar")
		torch.save(state_dict, checkpoints_dir + "/model"+str(epoch)+".tar")

		if is_best_epoch:
			print(f"\t Found a max score in the {epoch} epoch, saving...")
			torch.save(state_dict, checkpoints_dir + "/best_model.tar")


	def resume_checkpoint(self, checkpoints_dir, from_latest = True):
		""" Function: Resume from the latest/best checkpoint.
		"""

		if from_latest:

			latest_model_path = checkpoints_dir + "/latest_model.tar"

			assert os.path.exists(latest_model_path), f"{latest_model_path} does not exist, can not load latest checkpoint."

			# self.dist.barrier()  # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work

			# device = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
			checkpoint = torch.load(latest_model_path, map_location=self.device)

			self.start_epoch = checkpoint["epoch"] + 1
			self.max_score = checkpoint["max_score"]
			# self.optimizer.load_state_dict(checkpoint["optimizer"])
			if self.use_amp:
				self.scaler.load_state_dict(checkpoint["scalar"])
			self.model.load_state_dict(checkpoint["model"])

			# if self.rank == 0:
			print(f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

		else:
			best_model_path = checkpoints_dir + "/best_model.tar"

			assert os.path.exists(best_model_path), f"{best_model_path} does not exist, can not load best model."

			# self.dist.barrier()  # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work

			# device = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
			checkpoint = torch.load(best_model_path, map_location=self.device)

			epoch = checkpoint["epoch"]
			# self.max_score = checkpoint["max_score"]
			# self.optimizer.load_state_dict(checkpoint["optimizer"])
			# self.scaler.load_state_dict(checkpoint["scaler"])
			if self.device == "cuda":
				self.model.load_state_dict(checkpoint["model"])
			elif self.device == "cpu":
				ck = {}
				for key in checkpoint["model"]:
					key_ = key.replace('module.','')
					ck[key_] = checkpoint["model"][key]
				self.model.load_state_dict(ck)
			else:
				raise Exception('device is not specified~')

			# if self.rank == 0:
			print(f"Best model at {epoch} epoch loaded.")