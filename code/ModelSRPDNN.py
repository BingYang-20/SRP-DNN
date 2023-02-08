import torch
import torch.nn as nn


class CausCnnBlock1x1(nn.Module):
	# expansion = 1
	def __init__(self, inplanes, planes, kernel=(1,1), stride=(1,1), padding=(0,0)):
		super(CausCnnBlock1x1, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)

	def forward(self, x):
		out = self.conv1(x)

		return out

class CausCnnBlock(nn.Module): 
	""" Function: Basic causal convolutional block
    """
	# expansion = 1
	def __init__(self, inplanes, planes, kernel=(3,3), stride=(1,1), padding=(1,2), use_res=True, downsample=None):
		super(CausCnnBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.downsample = downsample
		self.stride = stride
		self.pad = padding
		self.use_res = use_res

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		if self.pad[1] !=0:
			out = out[:,:,:,:-self.pad[1]]

		out = self.conv2(out)
		out = self.bn2(out)
		if self.pad[1] != 0:
			out = out[:, :, :, :-self.pad[1]]

		if self.use_res == True:
			if self.downsample is not None:
				residual = self.downsample(x)
			out += residual

		out = self.relu(out)

		return out


class CRNN(nn.Module):
	""" Proposed CRNN model
	"""
	def __init__(self, ):
		super(CRNN, self).__init__()

		cnn_in_dim = 4
		cnn_dim = 64
		res_flag = False
		self.cnn = nn.Sequential(
                CausCnnBlock(cnn_in_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,2), use_res=res_flag),
                nn.MaxPool2d(kernel_size=(4, 1)),
				CausCnnBlock(cnn_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,2), use_res=res_flag),
				nn.MaxPool2d(kernel_size=(2, 1)),
				CausCnnBlock(cnn_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,2), use_res=res_flag),
				nn.MaxPool2d(kernel_size=(2, 2)),
				CausCnnBlock(cnn_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,2), use_res=res_flag),
				nn.MaxPool2d(kernel_size=(2, 2)),
				CausCnnBlock(cnn_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,2), use_res=res_flag),
				nn.MaxPool2d(kernel_size=(2, 3)),
            )

		ratio = 2
		rnn_in_dim = 256
		rnn_hid_dim = 256
		rnn_out_dim = 128*2*ratio
		rnn_bdflag = False
		if rnn_bdflag:
			rnn_ndirection = 2
		else:
			rnn_ndirection = 1
		self.rnn_bdflag = rnn_bdflag
		# self.rnn = torch.nn.LSTM(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=2, batch_first=True,
		#                                  bias=True, dropout=0.4, bidirectional=rnn_bdflag)
		self.rnn = torch.nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=1,
								batch_first=True, bias=True, dropout=0.4, bidirectional=rnn_bdflag)

		self.rnn_fc = nn.Sequential(
			# nn.Dropout(0.2),
			torch.nn.Linear(in_features=rnn_ndirection * rnn_hid_dim, out_features=rnn_out_dim),  # ,bias=False
			nn.Tanh(),
		)

	def forward(self, x):
		fea = x
		nb, _, nf, nt = fea.shape # (66,4,256,1249)

		fea_cnn = self.cnn(fea)  # (nb, nch, nf, nt) 
		fea_rnn_in = fea_cnn.view(nb, -1, fea_cnn.size(3))  # (nb, nch*nf,nt), nt = 1
		fea_rnn_in = fea_rnn_in.permute(0, 2, 1)  # (nb, nt, nfea)

		fea_rnn, _ = self.rnn(fea_rnn_in)
		fea_rnn_fc = self.rnn_fc(fea_rnn)*2 # (nb, nt, 2nf) 66,104,512
		# fea_rnn_fc = torch.zeros((132,104,512)).to(fea.device)

		return fea_rnn_fc



if __name__ == "__main__":
	import torch
	input = torch.randn((2,512)).cuda()
	net = CRNN()
	ouput = net(input)
	print('# parameters:', sum(param.numel() for param in net.parameters()))
