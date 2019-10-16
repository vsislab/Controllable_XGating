#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_io import *
import myopts
from CaptionModel import CaptionModel

def to_contiguous(tensor):
	if tensor.is_contiguous():
		return tensor
	else:
		return tensor.contiguous()


# gate functiuon
class Gate(nn.Module):
	def __init__(self, seed, source_size, target_size, drop_lm, simple=True):
		super(Gate, self).__init__()
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.source_size = source_size
		self.middle_size = 2 * self.source_size
		self.target_size = target_size
		self.drop_prob_lm = drop_lm

		if simple:
			self.gate = nn.Sequential(nn.Linear(self.source_size, self.target_size),
									  nn.ReLU(),
									  nn.Dropout(self.drop_prob_lm),
									  )
		else:
			self.gate = nn.Sequential(nn.Linear(self.source_size, self.middle_size),
									  nn.ReLU(),
									  nn.Dropout(self.drop_prob_lm),
									  nn.Linear(self.middle_size, self.target_size),
									  nn.ReLU(),
									  nn.Dropout(self.drop_prob_lm)
									  )

	def forward(self, source, target):
		''' 使用 source 生成 gate 控制 target '''
		gate = self.gate(source)
		new_target = gate * target + target
		# new_target = gate * target
		return new_target


# fusion function
class Fusion(nn.Module):
	''' concate tow features and use MLP to fuse them. '''
	def __init__(self, seed, feat_size1, feat_size2, fusion_size, drop_lm=0.5, activity=None):
		super(Fusion, self).__init__()
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.feat_size1 = feat_size1
		self.feat_size2 = feat_size2
		self.fusion_size = fusion_size
		self.drop_prob_lm = drop_lm
		self.activity = activity

		self.late_fusion = nn.Sequential(nn.Linear(self.feat_size1 + self.feat_size2, self.fusion_size),
										 getattr(nn, self.activity)(),
										 nn.Dropout(self.drop_prob_lm)
										 )

	def forward(self, feats1, feats2, feat_mask=None):  # feats (m ,28, featsize1) feat_mask(m, 28)
		assert feats1.shape[0] == feats2.shape[0] and feats1.shape[1] == feats2.shape[1], 'Size Error: sizes of feats1 and feats2 are not match.'
		feats = to_contiguous(torch.cat([feats1, feats2], -1))  # (m, 28, feat_size+featsize2)
		feats = self.late_fusion(feats)
		return feats


##############
## encoders ##
##############
class EncoderLstm_two_fc(nn.Module):
	''' generally, take RGB and OF fc_features as input:
		1. visual embedding layer turn them as the embeding size;
		2. encoder LSTMCell exploits the temporal information;
		3. output of LSTMCell <gate> with each other;
		4. <late fusion> the two kind gate outputs. '''
	def __init__(self, opt):
		super(EncoderLstm_two_fc, self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)

		# self.flag == 'temporal'
		self.feat_size_rgb = opt.feat_size
		self.feat_size_opfl = opt.feat_size2

		self.embed_size = opt.rnn_size
		self.rnn_size = opt.rnn_size
		self.drop_prob_lm = opt.drop_prob_lm

		self.visual_emb_rgb = nn.Sequential(nn.Linear(self.feat_size_rgb, self.embed_size),
											nn.BatchNorm1d(self.embed_size),
											nn.ReLU(True),
											)  # (m, 28, 512)
		self.visual_emb_opfl = nn.Sequential(nn.Linear(self.feat_size_opfl, self.embed_size),
											 nn.BatchNorm1d(self.embed_size),
											 nn.ReLU(True),
											 )  # (m, 28, 512)
		self.drop_out = nn.Dropout(self.drop_prob_lm)
		self.lstmcell_rgb = nn.LSTMCell(self.embed_size, self.rnn_size)
		self.lstmcell_opfl = nn.LSTMCell(self.embed_size, self.rnn_size)
		self.gate_rgb = Gate(opt.seed, opt.rnn_size, opt.rnn_size, opt.drop_prob_lm)
		self.gate_opfl = Gate(opt.seed, opt.rnn_size, opt.rnn_size, opt.drop_prob_lm)
		self.fusion = Fusion(opt.seed, opt.rnn_size, opt.rnn_size, opt.rnn_size, opt.drop_prob_lm, opt.fusion_activity)

	def init_hidden(self, batch_size):
		h_size = (batch_size, self.rnn_size)
		h_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		c_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		return (h_0, c_0)

	def forward(self, feats_rgb, feats_opfl, feats_mask ):  # feats:(m, 28, 1536), feats_mask:(m,28)
		batch_size, feat_len = feats_rgb.size(0), feats_rgb.size(1)  # 因为rgb和optical flow在数据数据输入阶段规定了只采样30帧，因此二者shape是完全一样的。
		# for rgb features
		emb_rgb = self.visual_emb_rgb(feats_rgb.view(-1, feats_rgb.size(-1)))
		emb_rgb = emb_rgb.view(batch_size, feat_len, -1)
		emb_rgb = self.drop_out(emb_rgb) * feats_mask.unsqueeze(-1)
		state_rgb = self.init_hidden(batch_size)
		# for optical flow features
		emb_opfl = self.visual_emb_opfl(feats_opfl.view(-1, feats_opfl.size(-1)))
		emb_opfl = emb_opfl.view(batch_size, feat_len, -1)
		emb_opfl = self.drop_out(emb_opfl) * feats_mask.unsqueeze(-1)
		state_opfl = self.init_hidden(batch_size)

		out_feats_rgb, out_feats_opfl = [], []
		for i in range(feat_len):
			input_rgb = emb_rgb[:, i, :]
			input_opfl = emb_opfl[:, i, :]

			# for rgb features
			mask_rgb = feats_mask[:, i]
			state_h_rgb, state_c_rgb = self.lstmcell_rgb(input_rgb, state_rgb)
			state_h_rgb = state_h_rgb * mask_rgb.unsqueeze(-1)
			state_c_rgb = state_c_rgb * mask_rgb.unsqueeze(-1)
			state_rgb = (state_h_rgb, state_c_rgb)

			# for optical flow features
			mask_opfl = feats_mask[:, i]
			state_h_opfl, state_c_opfl = self.lstmcell_opfl(input_opfl, state_opfl)
			state_h_opfl = state_h_opfl * mask_opfl.unsqueeze(-1)
			state_c_opfl = state_c_opfl * mask_opfl.unsqueeze(-1)
			state_opfl = (state_h_opfl, state_c_opfl)

			# inner fuse hidden of rgb and optical flow
			state_h_rgb_new = self.gate_rgb(state_h_opfl, state_h_rgb)
			state_h_opfl_new = self.gate_opfl(state_h_rgb, state_h_opfl)
			out_feats_rgb.append(state_h_rgb_new)
			out_feats_opfl.append(state_h_opfl_new)
		output_rgb = torch.cat([x.unsqueeze(1) for x in out_feats_rgb], dim=1)  # (m, nframes, rnn_size)
		output_opfl = torch.cat([x.unsqueeze(1) for x in out_feats_opfl], dim=1)  # (m, nframes, rnn_size)
		# fusion rgb and optical flow hidden state together
		output = self.fusion(output_rgb, output_opfl, feats_mask)  # (m, nframes, rnn_size)
		return output


class EncoderLstm_one_fc(nn.Module):
	''' generally, take RGB or OF fc_features as input:
		1. visual embedding layer turn them as the embeding size;
		2. encoder LSTMCell exploits the temporal information; '''
	def __init__(self, opt):
		super(EncoderLstm_one_fc, self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)

		self.feat_size = opt.feat_size

		self.embed_size = opt.rnn_size
		self.rnn_size = opt.rnn_size
		self.drop_prob_lm = opt.drop_prob_lm
		self.visual_emb = nn.Sequential(nn.Linear(self.feat_size, self.embed_size),
										nn.BatchNorm1d(self.embed_size),
										nn.ReLU(True))

		self.drop_out = nn.Dropout(self.drop_prob_lm)
		self.lstmcell = nn.LSTMCell(self.embed_size, self.rnn_size)

	def init_hidden(self, batch_size):
		h_size = (batch_size, self.rnn_size)
		h_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		c_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		return (h_0, c_0)

	def forward(self, feats, feats_mask ):  # feats:(m, 28, 1536), feats_mask:(m,28)
		batch_size, feat_len = feats.size(0), feats.size(1)
		emb = self.visual_emb(feats.view(-1, feats.size(-1)))
		emb = emb.view(batch_size, feat_len, -1)
		emb = self.drop_out(emb) * feats_mask.unsqueeze(-1)
		state = self.init_hidden(batch_size)

		out_feats = []
		for i in range(feat_len):
			input = emb[:, i, :]
			mask = feats_mask[:, i]

			state_h, state_c = self.lstmcell(input, state)
			state_h = state_h * mask.unsqueeze(-1)
			state_c = state_c * mask.unsqueeze(-1)
			state = (state_h, state_c)
			out_feats.append(state_h)
		output = torch.cat([x.unsqueeze(1) for x in out_feats], dim=1)
		return output


class EncoderLstm_two_spatial(nn.Module):
	''' Generally, take RGB and OF pool_feature as input:
		1. for each data, pool feature are spatial summarized;
		2. summarized RGB and OF pool feature <gate> each other;
		3. visual embedding layer turn Gated summarized features as the embedding size;
		4. encoder LSTMCell exploits the temporal information;
		5. output of encoder LSTMCell <gate> with each other;
		6. <late fusion> two kind output of gate.'''
	def __init__(self, opt):
		super(EncoderLstm_two_spatial, self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)

		self.feat_depth_rgb = opt.feat_depth
		self.HxW_rgb = opt.HxW

		self.feat_depth_opfl = opt.feat_depth2
		self.HxW_opfl = opt.HxW2

		self.embed_size = opt.rnn_size
		self.rnn_size = opt.rnn_size
		self.drop_prob_lm = opt.drop_prob_lm

		self.atten_rgb = Spatial_Atten(opt, self.feat_depth_rgb, self.HxW_rgb)
		self.atten_opfl = Spatial_Atten(opt, self.feat_depth_opfl, self.HxW_opfl)

		self.lstm_rgb = Lstm_spatial(opt, self.feat_depth_rgb)
		self.lstm_opfl = Lstm_spatial(opt, self.feat_depth_opfl)

		self.gate_rgb_front = Gate(opt.seed, self.feat_depth_opfl, self.feat_depth_rgb, self.drop_prob_lm)
		self.gate_opfl_front = Gate(opt.seed, self.feat_depth_rgb, self.feat_depth_opfl, self.drop_prob_lm)
		self.gate_rgb_late = Gate(opt.seed, self.rnn_size, self.rnn_size, self.drop_prob_lm)
		self.gate_opfl_late = Gate(opt.seed, self.rnn_size, self.rnn_size, self.drop_prob_lm)
		self.fusion = Fusion(opt.seed, self.rnn_size, self.rnn_size, self.rnn_size, self.drop_prob_lm, opt.fusion_activity)

	def init_hidden(self, feats, feats_mask):
		''' feats: (m, K, HxW, depth), feats_mask: (m, K)'''
		batch_size, nfeats, HxW, depth = feats.shape
		h_size = (batch_size, self.rnn_size)
		h_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		c_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		return (h_0, c_0)

	def forward(self, feats_rgb, feats_opfl, feats_mask):
		''' 输入的特征是 (m, K, H, W, depth), mask 是 (m, k)'''
		assert len(feats_rgb.shape) == 5 and len(feats_opfl.shape) == 5, "Size Error: Shape of batch-feats should be 5."
		assert feats_rgb.shape[:2] == feats_opfl.shape[:2]
		batch_size, feat_len, height_rgb, weight_rgb, depth_rgb = feats_rgb.shape
		batch_size, feat_len, height_opfl, weight_opfl, depth_opfl = feats_opfl.shape
		assert self.HxW_rgb == height_rgb * weight_rgb and self.feat_depth_rgb == depth_rgb
		assert self.HxW_opfl == height_opfl * weight_opfl and self.feat_depth_opfl == depth_opfl

		feats_rgb = feats_rgb.view(batch_size, feat_len, self.HxW_rgb, self.feat_depth_rgb)  # (m, K, HxW_rgb, depth_rgb)
		feats_opfl = feats_opfl.view(batch_size, feat_len, self.HxW_opfl, self.feat_depth_opfl)
		state_rgb = self.init_hidden(feats_rgb, feats_mask)
		state_opfl = self.init_hidden(feats_opfl, feats_mask)
		outfeats_rgb, outfeats_opfl = [], []
		for i in range(feat_len):
			mask = feats_mask[:, 1]  # (m,)
			i_feats_rgb = feats_rgb[:, i, :, :]  # (m, HxW, depth)
			i_feats_opfl = feats_opfl[:, i, :, :]
			# soft attention
			input_rgb, alpha_rgb = self.atten_rgb(i_feats_rgb, mask, state_rgb[0])  # (m, depth_rgb),(m, HxW_rgb)
			input_opfl, alpha_opfl = self.atten_opfl(i_feats_opfl, mask, state_opfl[0])
			# gate for encoding
			input_rgb_new = self.gate_rgb_front(input_opfl, input_rgb)  # (m, depth_rgb)
			input_opfl_new = self.gate_opfl_front(input_rgb, input_opfl)  # (m, depth_opfl)
			# encoding
			outfeat_rgb, state_rgb = self.lstm_rgb(input_rgb_new, mask, state_rgb)  # (m, rnn_size)
			outfeat_opfl, state_opfl = self.lstm_opfl(input_opfl_new, mask, state_opfl)
			# collecting rgb and opfl outputs
			outfeats_rgb.append(outfeat_rgb)
			outfeats_opfl.append(outfeat_opfl)
		outfeats_rgb = torch.stack(outfeats_rgb, dim=1)  # (m, K, rnn_size)
		outfeats_opfl = torch.stack(outfeats_opfl, dim=1)  # (m, K, rnn_size)
		# gate on outputs
		outfeats_rgb_new = self.gate_rgb_late(outfeats_opfl, outfeats_rgb)  # (m, K, rnn_size)
		outfeats_opfl_new = self.gate_opfl_late(outfeats_rgb, outfeats_opfl)
		output = self.fusion(outfeats_rgb_new, outfeats_opfl_new)  # (m, K, rnn_size)
		return output


class EncoderLstm_two_spatial_nogate(nn.Module):
	''' Generally, take RGB and OF pool_feature as input:
		1. for each data, pool feature are spatial summarized;
		3. visual embedding layer turn summarized features as the embedding size;
		4. encoder LSTMCell exploits the temporal information;
		6. <late fusion> two kind output of encoder LSTMCell.'''
	def __init__(self, opt):
		super(EncoderLstm_two_spatial_nogate, self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)

		self.feat_depth_rgb = opt.feat_depth
		self.HxW_rgb = opt.HxW

		self.feat_depth_opfl = opt.feat_depth2
		self.HxW_opfl = opt.HxW2

		self.embed_size = opt.rnn_size
		self.rnn_size = opt.rnn_size
		self.drop_prob_lm = opt.drop_prob_lm

		self.atten_rgb = Spatial_Atten(opt, self.feat_depth_rgb, self.HxW_rgb)
		self.atten_opfl = Spatial_Atten(opt, self.feat_depth_opfl, self.HxW_opfl)

		self.lstm_rgb = Lstm_spatial(opt, self.feat_depth_rgb)
		self.lstm_opfl = Lstm_spatial(opt, self.feat_depth_opfl)

		# self.gate_rgb_front = Gate(opt.seed, self.feat_depth_opfl, self.feat_depth_rgb, self.drop_prob_lm)
		# self.gate_opfl_front = Gate(opt.seed, self.feat_depth_rgb, self.feat_depth_opfl, self.drop_prob_lm)
		# self.gate_rgb_late = Gate(opt.seed, self.rnn_size, self.rnn_size, self.drop_prob_lm)
		# self.gate_opfl_late = Gate(opt.seed, self.rnn_size, self.rnn_size, self.drop_prob_lm)
		self.fusion = Fusion(opt.seed, self.rnn_size, self.rnn_size, self.rnn_size, self.drop_prob_lm, opt.fusion_activity)

	def init_hidden(self, feats, feats_mask):
		''' feats: (m, K, HxW, depth), feats_mask: (m, K)'''
		batch_size, nfeats, HxW, depth = feats.shape
		h_size = (batch_size, self.rnn_size)
		h_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		c_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		return (h_0, c_0)

	def forward(self, feats_rgb, feats_opfl, feats_mask):
		''' 输入的特征是 (m, K, H, W, depth), mask 是 (m, k)'''
		assert len(feats_rgb.shape) == 5 and len(feats_opfl.shape) == 5, "Size Error: Shape of batch-feats should be 5."
		assert feats_rgb.shape[:2] == feats_opfl.shape[:2]
		batch_size, feat_len, height_rgb, weight_rgb, depth_rgb = feats_rgb.shape
		batch_size, feat_len, height_opfl, weight_opfl, depth_opfl = feats_opfl.shape
		assert self.HxW_rgb == height_rgb * weight_rgb and self.feat_depth_rgb == depth_rgb
		assert self.HxW_opfl == height_opfl * weight_opfl and self.feat_depth_opfl == depth_opfl

		feats_rgb = feats_rgb.view(batch_size, feat_len, self.HxW_rgb, self.feat_depth_rgb)  # (m, K, HxW_rgb, depth_rgb)
		feats_opfl = feats_opfl.view(batch_size, feat_len, self.HxW_opfl, self.feat_depth_opfl)
		state_rgb = self.init_hidden(feats_rgb, feats_mask)
		state_opfl = self.init_hidden(feats_opfl, feats_mask)
		outfeats_rgb, outfeats_opfl = [], []
		for i in range(feat_len):
			mask = feats_mask[:, 1]  # (m,)
			i_feats_rgb = feats_rgb[:, i, :, :]  # (m, HxW, depth)
			i_feats_opfl = feats_opfl[:, i, :, :]
			# soft attention
			input_rgb, alpha_rgb = self.atten_rgb(i_feats_rgb, mask, state_rgb[0])  # (m, depth_rgb),(m, HxW_rgb)
			input_opfl, alpha_opfl = self.atten_opfl(i_feats_opfl, mask, state_opfl[0])
			# gate for encoding
			# input_rgb_new = self.gate_rgb_front(input_opfl, input_rgb)  # (m, depth_rgb)
			# input_opfl_new = self.gate_opfl_front(input_rgb, input_opfl)  # (m, depth_opfl)
			# encoding
			outfeat_rgb, state_rgb = self.lstm_rgb(input_rgb, mask, state_rgb)  # (m, rnn_size)
			outfeat_opfl, state_opfl = self.lstm_opfl(input_opfl, mask, state_opfl)
			# collecting rgb and opfl outputs
			outfeats_rgb.append(outfeat_rgb)
			outfeats_opfl.append(outfeat_opfl)
		outfeats_rgb = torch.stack(outfeats_rgb, dim=1)  # (m, K, rnn_size)
		outfeats_opfl = torch.stack(outfeats_opfl, dim=1)  # (m, K, rnn_size)
		# gate on outputs
		# outfeats_rgb_new = self.gate_rgb_late(outfeats_opfl, outfeats_rgb)  # (m, K, rnn_size)
		# outfeats_opfl_new = self.gate_opfl_late(outfeats_rgb, outfeats_opfl)
		output = self.fusion(outfeats_rgb, outfeats_opfl)  # (m, K, rnn_size)
		return output


class EncoderLstm_one_spatial(nn.Module):
	''' Generally, take RGB or OF pool_feature as input:
		1. for each data, pool feature are spatial summarized;
		3. visual embedding layer turn summarized features as the embedding size;
		4. encoder LSTMCell exploits the temporal information;'''
	def __init__(self, opt):
		super(EncoderLstm_one_spatial, self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)

		self.feat_depth = opt.feat_depth
		self.HxW = opt.HxW

		self.embed_size = opt.rnn_size
		self.rnn_size = opt.rnn_size
		self.drop_prob_lm = opt.drop_prob_lm

		self.atten= Spatial_Atten(opt, self.feat_depth, self.HxW)

		self.lstmcell = Lstm_spatial(opt, self.feat_depth)

	def init_hidden(self, feats, feats_mask):
		''' feats: (m, K, HxW, depth), feats_mask: (m, K)'''
		batch_size, nfeats, HxW, depth = feats.shape
		h_size = (batch_size, self.rnn_size)
		h_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		c_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		return (h_0, c_0)

	def forward(self, feats, feats_mask):
		''' 输入的特征是 (m, K, H, W, depth), mask 是 (m, k)'''
		assert len(feats.shape) == 5, "Size Error: Shape of batch-feats should be 5."
		batch_size, feat_len, height, weight, depth = feats.shape
		assert self.HxW == height * weight and self.feat_depth == depth

		feats = feats.view(batch_size, feat_len, self.HxW, self.feat_depth)  # (m, K, HxW, depth)
		state = self.init_hidden(feats, feats_mask)
		outfeats = []
		for i in range(feat_len):
			mask = feats_mask[:, 1]  # (m,)
			i_feats = feats[:, i, :, :]  # (m, HxW, depth)
			# soft attention
			input, alpha = self.atten(i_feats, mask, state[0])  # (m, depth_rgb),(m, HxW_rgb)
			# encoding
			outfeat, state = self.lstmcell(input, mask, state)  # (m, rnn_size)
			# collecting rgb and opfl outputs
			outfeats.append(outfeat)
		output = torch.stack(outfeats, dim=1)  # (m, K, rnn_size)
		return output

class EncoderLstm_onespatial_onefc(nn.Module):
	''' Generally, take RGB and OF pool_feature as input:
		1. for each data, pool feature are spatial summarized;
		3. visual embedding layer turn summarized features as the embedding size;
		4. encoder LSTMCell exploits the temporal information of summarized spatial feature and fc, respronlly;
		6. <late fusion> two kind output of encoder LSTMCell.'''
	def __init__(self, opt):
		super(EncoderLstm_onespatial_onefc, self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)
		# fc
		self.feat_size = opt.feat_size
		# local
		self.feat_depth = opt.feat_depth
		self.HxW = opt.HxW

		self.embed_size = opt.rnn_size
		self.rnn_size = opt.rnn_size
		self.drop_prob_lm = opt.drop_prob_lm

		self.atten_local = Spatial_Atten(opt, self.feat_depth, self.HxW)
		self.lstm_fc = Lstm_spatial(opt, self.feat_size)
		self.lstm_local = Lstm_spatial(opt, self.feat_depth)

		self.fusion = Fusion(opt.seed, self.rnn_size, self.rnn_size, self.rnn_size, self.drop_prob_lm, opt.fusion_activity)

	def init_hidden(self, feats, feats_mask):
		''' feats: (m, K, HxW, depth), feats_mask: (m, K)'''
		batch_size = feats.size(0)
		h_size = (batch_size, self.rnn_size)
		h_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		c_0 = Variable(torch.FloatTensor(*h_size).zero_(), requires_grad=False).cuda()
		return (h_0, c_0)

	def forward(self, feats_fc, feats_local, feats_mask):
		''' 输入的特征是(m, K, feat_size) 和 (m, K, H, W, depth), mask 是 (m, k)'''
		assert len(feats_fc.shape) == 3 and len(feats_local.shape) == 5, "Size Error"
		assert feats_fc.shape[:2] == feats_local.shape[:2]
		batch_size, feat_len, feat_size_fc = feats_fc.shape
		batch_size, feat_len, height, weight, depth_local = feats_local.shape
		assert self.HxW == height * weight and self.feat_depth == depth_local

		feats_local = feats_local.view(batch_size, feat_len, self.HxW, self.feat_depth)  # (m, K, HxW, depth)
		state_local = self.init_hidden(feats_local, feats_mask)
		state_fc = self.init_hidden(feats_fc, feats_mask)
		outfeats_local, outfeats_fc = [], []
		for i in range(feat_len):
			mask = feats_mask[:, 1]  # (m,)
			i_feats_local = feats_local[:, i, :, :]  # (m, HxW, depth)
			i_feats_fc = feats_fc[:, i, :]  # (m, feat_size)
			# soft attention
			input_local, alpha_local = self.atten_local(i_feats_local, mask, state_local[0])  # (m, depth_rgb),(m, HxW_rgb)
			input_fc = i_feats_fc
			# encoding
			outfeat_local, state_local = self.lstm_local(input_local, mask, state_local)  # (m, rnn_size)
			outfeat_fc, state_fc = self.lstm_fc(input_fc, mask, state_fc)
			# collecting rgb and opfl outputs
			outfeats_local.append(outfeat_local)
			outfeats_fc.append(outfeat_fc)
		outfeats_local = torch.stack(outfeats_local, dim=1)  # (m, K, rnn_size)
		outfeats_fc = torch.stack(outfeats_fc, dim=1)  # (m, K, rnn_size)
		# late fusion
		output = self.fusion(outfeats_fc, outfeats_local)  # (m, K, rnn_size)
		return output

##########################################
## important subsubmodules in ENCODERS. ##
##########################################
class Spatial_Atten(nn.Module):
	''' take RGB or OF pool_features as inputs, also the driver state,
		make the spatial attention on the pool_features. '''
	def __init__(self, opt, feat_depth, HxW,):
		super(Spatial_Atten, self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)
		self.feat_depth = feat_depth
		self.HxW = HxW
		self.rnn_size = opt.rnn_size
		self.mid_size = int(0.5 * (self.rnn_size + self.feat_depth))
		self.drop_prob_lm_attn = 0.1

		self.spatial_h = nn.Linear(self.rnn_size, self.mid_size)
		self.spatial_f = nn.Linear(self.feat_depth, self.mid_size)
		self.spatial_a = nn.Linear(self.mid_size, 1)
		self.dropout_attn = nn.Dropout(self.drop_prob_lm_attn)

	def forward(self, feats, feats_mask, state_h):
		''' feats: (m, HxW, depth), feats_mask: (m,), state_h:(m,rnn_size) '''
		assert self.feat_depth == feats.shape[-1] and self.HxW == feats.shape[1]
		alpha = F.tanh(self.spatial_h(state_h).unsqueeze(1) + self.spatial_f(feats))  # (m, HxW, midsize)
		alpha = self.dropout_attn(alpha)
		alpha = F.softmax(self.spatial_a(alpha), dim=1)  # (m, HxW, 1)
		alpha = self.dropout_attn(alpha)
		output = torch.bmm(alpha.transpose(1, 2), feats)  # (m, 1, depth)
		output = output.squeeze(1)  # (m, depth)
		return output, alpha.squeeze(-1)


class Lstm_spatial(nn.Module):
	''' This function is the temporal encoding processing in spatial encoder.
		It is convenient to write it separably. '''
	def __init__(self, opt, feat_depth):
		super(Lstm_spatial, self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)
		self.feat_depth = feat_depth
		self.embed_size = opt.rnn_size
		self.rnn_size = opt.rnn_size
		self.drop_prob_lm = opt.drop_prob_lm

		self.visual_emb = nn.Sequential(nn.Linear(self.feat_depth, self.embed_size),
										nn.BatchNorm1d(self.embed_size),
										nn.ReLU(True),
										)
		self.drop_out = nn.Dropout(self.drop_prob_lm)
		self.lstmcell = nn.LSTMCell(self.embed_size, self.rnn_size)
	def forward(self, input, mask, state):
		''' input:(m, depth), mask:(m,) '''
		assert self.feat_depth == input.shape[-1]
		emb = self.visual_emb(input)  # (m, embed_size)
		emb = self.drop_out(emb) * mask.unsqueeze(-1)

		state_h, state_c = self.lstmcell(emb, state)
		state_h = state_h * mask.unsqueeze(-1)
		state_h = self.drop_out(state_h)
		state_c = state_c * mask.unsqueeze(-1)
		state = (state_h, state_c)
		output = state_h
		return output, state


##############
## decoders ##
##############
class LSTMCore_one_layer(nn.Module):
	''' One-layer decoder:
		1. <soft attention> all features;
		2. LSTM generates a word. '''
	def __init__(self,opt):
		super(LSTMCore_one_layer,self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)
		#  word and rnn size
		self.input_encoding_size = opt.input_encoding_size  #  468
		self.rnn_size = opt.rnn_size  #  1000 or 3518
		#  feat and attention size
		self.feat_size = opt.feat_size  #  1536
		self.visual_size = opt.rnn_size
		self.att_size = opt.att_size  #
		#  drop out
		self.drop_prob_lm = opt.drop_prob_lm  #  0.5
		#  Build a LSTM
		self.lstmcell = two_inputs_lstmcell(self.input_encoding_size, self.visual_size, self.rnn_size, self.drop_prob_lm)
		#  Build the soft attention
		self.v2a = nn.Linear(self.visual_size, self.att_size)  #  Uv  (1536,att_size)
		self.h2a = nn.Linear(self.rnn_size, self.att_size)  #  Wh  (1000,1536)
		self.a2w = nn.Linear(self.att_size, 1)  #  Wtanh(Wh+Uv+b)  (1536,1)

	def forward(self, xt, xt_mask, V, state): # xt:(m,input_encoding_size) xt_mask:(m,1) V:(m,28,visual_size)
		#  attention
		alpha = self.h2a(state[0][-1]).unsqueeze(1) + self.v2a(V)  # [(m,rnn_size)x(rnn_size,att_size)].unsqueeze(1)+(m,28,visual_size)x(visual_size,att_size)=>(m,28,att_size)
		alpha = F.tanh( alpha )
		alpha = self.a2w(alpha)  #  (m,28,1)
		alpha = alpha.transpose(1,0)  #  (28,m,1)
		alpha = F.softmax( alpha, dim=0 )  #  (28,m,1)
		alpha = alpha.transpose(1,0)  #  (m,28,1)
		af = torch.sum(alpha*V, dim=1)  #  (m,self.feat_size)

		# lstm unit
		output, state = self.lstmcell(xt, af, state, xt_mask)
		return output, state


class LSTMCore_two_layer(nn.Module):
	''' two-layer decoder:
		1. <soft attention> all input features;
		2. 双层LSTM, 第一层仅输入tokens，第二层输入第一层的状态和attented特征'''
	def __init__(self,opt):
	# def __init__(self,input_encoding_size,rnn_size,drop_prob_lm,feat_size,att_size):
		super(LSTMCore_two_layer,self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)
		#  word and rnn size
		self.input_encoding_size = opt.input_encoding_size  #  468
		self.rnn_size = opt.rnn_size  #  1000 or 3518
		#  feat and attention size
		self.visual_size = opt.rnn_size
		self.att_size = opt.att_size  #
		self.globalpos_size = opt.rnn_size
		#  drop out
		self.drop_prob_lm = opt.drop_prob_lm  #  0.5
		#  The first lstm layer
		# self.lstm_1 = one_input_lstmcell(self.input_encoding_size, self.rnn_size, self.drop_prob_lm)
		self.lstm_1 = two_inputs_lstmcell(self.input_encoding_size, self.globalpos_size, self.rnn_size, self.drop_prob_lm)
		#  The second lstm layer
		self.lstm_2 = two_inputs_lstmcell(self.rnn_size, self.visual_size, self.rnn_size, self.drop_prob_lm)
		#  Build a soft attention
		self.dropout = nn.Dropout(self.drop_prob_lm)
		self.v2a = nn.Linear(self.visual_size, self.att_size)  #  Uv  (1536,att_size)
		self.h2a = nn.Linear(self.rnn_size + self.rnn_size, self.att_size)  #  Wh  (1000,1536)
		self.a2w = nn.Linear(self.att_size, 1)  #  Wtanh(Wh+Uv+b)  (1536,1)

	def forward(self, xt, xt_mask, V, pos_feat, state): # xt:(m,input_encoding_size) xt_mask:(m,1) V:(m,28,visual_size) pos_feat:(m, pos_size)
		'''state should be a list: [state_layer1, state_layer2]'''
		assert len(state) == 2, "input parameters 'state' expect a list with 2 elements"
		# the first LSTM only take "xt" as input
		state1, state2 = state[0], state[1]
		# attention
		alpha = self.h2a(torch.cat([state1[0][-1], state2[0][-1]], dim=1)).unsqueeze(1) + self.v2a(V)  # (m, feat_K, att_size)
		alpha = self.a2w(F.tanh(alpha))  # (m, feat_K, 1)
		alpha = F.softmax(alpha.transpose(1, 0), dim=0).transpose(1, 0)  # (m, feat_K, 1)
		af = torch.sum(alpha * V, dim=1)  # (m, visual_size)
		# tow-layer lstm
		output_1, state1 = self.lstm_1(xt, pos_feat, state1, xt_mask)
		output, state2 = self.lstm_2(output_1, af, state2, xt_mask)
		# states：
		state = [state1, state2]
		return output, state

class LSTMCore_two_layer_gate(nn.Module):
	''' two-layer decoder:
		1. <soft attention> all input features;
		2. 双层LSTM, 第一层仅输入tokens，第二层输入第一层的状态和attented特征'''
	def __init__(self,opt):
	# def __init__(self,input_encoding_size,rnn_size,drop_prob_lm,feat_size,att_size):
		super(LSTMCore_two_layer_gate,self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)
		#  word and rnn size
		self.input_encoding_size = opt.input_encoding_size  #  468
		self.rnn_size = opt.rnn_size  #  1000 or 3518
		#  feat and attention size
		self.visual_size = opt.rnn_size
		self.att_size = opt.att_size  #
		self.globalpos_size = opt.rnn_size
		#  drop out
		self.drop_prob_lm = opt.drop_prob_lm  #  0.5
		#  The first lstm layer
		# self.lstm_1 = one_input_lstmcell(self.input_encoding_size, self.rnn_size, self.drop_prob_lm)
		self.gate = Gate(opt.seed, self.input_encoding_size, self.globalpos_size, self.drop_prob_lm)
		self.lstm_1 = two_inputs_lstmcell(self.input_encoding_size, self.globalpos_size, self.rnn_size, self.drop_prob_lm)
		#  The second lstm layer
		self.lstm_2 = two_inputs_lstmcell(self.rnn_size, self.visual_size, self.rnn_size, self.drop_prob_lm)
		#  Build a soft attention
		self.dropout = nn.Dropout(self.drop_prob_lm)
		self.v2a = nn.Linear(self.visual_size, self.att_size)  #  Uv  (1536,att_size)
		self.h2a = nn.Linear(self.rnn_size + self.rnn_size, self.att_size)  #  Wh  (1000,1536)
		self.a2w = nn.Linear(self.att_size, 1)  #  Wtanh(Wh+Uv+b)  (1536,1)

	def forward(self, xt, xt_mask, V, pos_feat, state): # xt:(m,input_encoding_size) xt_mask:(m,1) V:(m,28,visual_size) pos_feat:(m, pos_size)
		'''state should be a list: [state_layer1, state_layer2]'''
		assert len(state) == 2, "input parameters 'state' expect a list with 2 elements"
		# the first LSTM only take "xt" as input
		state1, state2 = state[0], state[1]
		# attention
		alpha = self.h2a(torch.cat([state1[0][-1], state2[0][-1]], dim=1)).unsqueeze(1) + self.v2a(V)  # (m, feat_K, att_size)
		alpha = self.a2w(F.tanh(alpha))  # (m, feat_K, 1)
		alpha = F.softmax(alpha.transpose(1, 0), dim=0).transpose(1, 0)  # (m, feat_K, 1)
		af = torch.sum(alpha * V, dim=1)  # (m, visual_size)
		# tow-layer lstm
		gated_posfeat = self.gate(xt, pos_feat)
		output_1, state1 = self.lstm_1(xt, gated_posfeat, state1, xt_mask)
		output, state2 = self.lstm_2(output_1, af, state2, xt_mask)
		# states：
		state = [state1, state2]
		return output, state

##########################################
## important subsubmodules in DECODERS. ##
##########################################
class one_input_lstmcell(nn.Module):
	''' rewrite the lstmcell with two inputs '''
	def __init__(self,input_size, rnn_size, drop_lm=0.5):
		super(one_input_lstmcell, self).__init__()
		self.input_size = input_size
		self.rnn_size = rnn_size
		self.drop_lm = drop_lm
		# input
		self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
		# previous hidden
		self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
		# dropout
		if self.drop_lm is not None:
			self.dropout = nn.Dropout(self.drop_lm)

	def forward(self, input, state, mask=None):
		''' 要求输入的state形状是 ((1, m, rnn_size),(1, m, rnn_size)) '''
		all_input_sums = self.i2h(input) + self.h2h(state[0][-1])
		sigmoid_chunk = all_input_sums.narrow(dimension=1, start=0, length=3*self.rnn_size)
		sigmoid_chunk = F.sigmoid(sigmoid_chunk)
		in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)  # (m, rnn_size)
		forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
		out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
		tanh_chunk = all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size)
		in_transform = F.tanh(tanh_chunk)
		state_c = forget_gate * state[1][-1] + in_gate * in_transform
		if mask is not None:
			state_c = state_c * mask + state[1][-1] * (1. - mask)

		state_h = out_gate * F.tanh(state_c)
		if mask is not None:
			state_h = state_h * mask + state[0][-1] * (1. - mask)
		if self.drop_lm is not None:
			state_h = self.dropout(state_h)

		output = state_h  # (m, rnn_size)
		# state returned has the same shape as the input state
		return output, (state_h.unsqueeze(0), state_c.unsqueeze(0))


class two_inputs_lstmcell(nn.Module):
	''' rewrite the lstmcell with two inputs '''
	def __init__(self,input_size, visual_size, rnn_size, drop_lm=0.5):
		super(two_inputs_lstmcell, self).__init__()
		self.input_size = input_size
		self.rnn_size = rnn_size
		self.visual_size = visual_size
		self.drop_lm = drop_lm
		# input1
		self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
		# input2
		self.a2h = nn.Linear(self.visual_size, 4 * self.rnn_size)
		# previous hidden
		self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
		# dropout
		if self.drop_lm is not None:
			self.dropout = nn.Dropout(self.drop_lm)

	def forward(self, input1, input2, state, mask=None):
		''' 要求输入的state形状是 ((1, m, rnn_size),(1, m, rnn_size)) '''
		all_input_sums = self.i2h(input1) + self.a2h(input2) + self.h2h(state[0][-1])
		sigmoid_chunk = all_input_sums.narrow(dimension=1, start=0, length=3*self.rnn_size)
		sigmoid_chunk = F.sigmoid(sigmoid_chunk)
		in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)  # (m, rnn_size)
		forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
		out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
		tanh_chunk = all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size)
		in_transform = F.tanh(tanh_chunk)
		state_c = forget_gate * state[1][-1] + in_gate * in_transform
		if mask is not None:
			state_c = state_c * mask + state[1][-1] * (1. - mask)
		state_h = out_gate * F.tanh(state_c)
		if mask is not None:
			state_h = state_h * mask + state[0][-1] * (1. - mask)
		if self.drop_lm is not None:
			state_h = self.dropout(state_h)
		output = state_h  # (m, rnn_size)
		# state returned has the same shape as the input state
		return output, (state_h.unsqueeze(0), state_c.unsqueeze(0))

