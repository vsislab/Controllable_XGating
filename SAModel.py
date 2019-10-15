#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_io import *
from sub_modules import *
import myopts
from CaptionModel import CaptionModel
from RecModel import RecModel


# entire model framework
class SAModel(CaptionModel):
	def __init__(self, opt):
		super(SAModel,self).__init__()
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)

		self.vocab_size = opt.vocab_size
		self.category_size = opt.category_size
		self.input_encoding_size = opt.input_encoding_size  # 468
		self.rnn_size = opt.rnn_size  # 512
		self.visual_size = opt.rnn_size
		self.num_layers = opt.num_layers  # 1
		self.drop_prob_lm = opt.drop_prob_lm
		self.seq_length = opt.seq_length
		self.ss_prob = 0.0  # Schedule sampling probability
		# self.img_embed = nn.Linear(self.feat_size, self.input_encoding_size)  # (1536,468)
		#============== insert encoder lstm ======================
		# self.encoder = EncoderLstm(opt)
		#self.two_spatial_encoder = EncoderLstm_two_spatial(opt)
		self.two_spatial_encoder = EncoderLstm_two_fc(opt)
		# self.two_spatial_encoder = EncoderLstm_two_spatial_nogate(opt)
		# self.fc_encoder = EncoderLstm_one_fc(opt)
		# self.one_spatial_encoder = EncoderLstm_one_spatial(opt)
		# self.TS_fusion = Fusion(opt.seed, opt.rnn_size, opt.feat_depth, opt.rnn_size, opt.drop_prob_lm, opt.fusion_activity)

		self.img_embed_h_1 = nn.Linear(self.visual_size, self.rnn_size)  # (rnn_size, rnn_size)
		self.img_embed_c_1 = nn.Linear(self.visual_size, self.rnn_size)  # (rnn_size, rnn_size)
		self.img_embed_h_2 = nn.Linear(self.visual_size, self.rnn_size)  # (rnn_size, rnn_size)
		self.img_embed_c_2 = nn.Linear(self.visual_size, self.rnn_size)  # (rnn_size, rnn_size)
		#self.lstmcore = LSTMCore_two_layer(opt)  # for decoder
		self.lstmcore = LSTMCore_two_layer_gate(opt)
		self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)  # (20000,468)
		self.logit = nn.Linear(self.rnn_size, self.vocab_size)  # (1000,20000)
		self.classifer = nn.Sequential(nn.Linear(self.rnn_size, 128),
		                               nn.ReLU(),
		                               nn.Dropout(self.drop_prob_lm),
		                               nn.Linear(128, self.category_size))
		self.reconstruct = opt.rec_strategy
		if self.reconstruct is not None:
			self.RecNet = RecModel(opt)
		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.embed.weight.data.uniform_(-initrange, initrange)
		self.logit.bias.data.fill_(0)
		self.logit.weight.data.uniform_(-initrange, initrange)

	# def init_hidden(self, batch_size):
	#     weight = next(self.parameters()).data
	#     return ( Variable( weight.new(self.num_layers, batch_size, self.rnn_size).zero_() ),
	#                 Variable( weight.new(self.num_layers, batch_size, self.rnn_size).zero_() ) ) # ( (1,m,1000),(1,m,1000) )
	def init_hidden(self, feat, feat_mask): # feat(m,28,1536),feat_mask:(m,28)
		feat_ = torch.from_numpy( np.sum(feat.cpu().data.numpy(),axis=1,dtype=np.float32) )  #(m,visual_size)
		mask_ = torch.from_numpy( np.sum(feat_mask.cpu().data.numpy(),axis=1,dtype=np.float32) )  # (m,)
		feat_mean = ( feat_ / mask_.unsqueeze(-1) ).unsqueeze(0)  # (1,m,visual_size)
		feat_mean = Variable(feat_mean,requires_grad=False).cuda()
		state1 = (self.img_embed_h_1(feat_mean), self.img_embed_c_1(feat_mean))
		state2 = (self.img_embed_h_2(feat_mean), self.img_embed_c_2(feat_mean))
		return [state1, state2] #( (1,m,rnn_size),(1,m,rnn_size) )

	def forward(self, feats_rgb, feats_opfl, feats_rgb_pool, feats_opfl_pool, feat_mask, pos_feats, seq, seq_mask):
		''' feats_rgb and feats_opfl: (m, K, feat_size)
			feats_rgb_pool and feats_opfl_pool: (m, K, H, W, Depth)
			pos_feats: (m, rnn_size)
			seq: (m,seq_len+1)
			seq_mask:(m,seq_len+1)'''
		'''
		1. rgb和opfl特征由 Temporal Encoder 编码嵌入 temporal 信息。
		'''
		# ===== here insert encoder lstm operation =====
		# feats_temporal = self.encoder(feats_rgb, feats_opfl, feat_mask)  # (m,28,visual_size)
		# ===== 进行 spatial attention =====
		feats = self.two_spatial_encoder(feats_rgb_pool, feats_opfl_pool, feat_mask)  # (m, K, depth)
		# feats = self.fc_encoder(feats_rgb, feat_mask)
		# feats = self.one_spatial_encoder(feats_rgb_pool, feat_mask)
		# ===== 进行 temporal 和 spatial 级别的融合
		# feats = self.TS_fusion(feats_temporal, feats_spatial)
		# =============== atten + decoding ===========
		batch_size = feats.size(0)
		state = self.init_hidden(feats, feat_mask) # ( (1,m,rnn_size), (1,m,rnn_size) )
		outputs_hidden = [] # before use log_softmax
		outputs, categories = [], [] # after use log_softmax
		for i in range(seq.size(1)):
			if self.training and i >= 1 and self.ss_prob > 0.0: # otherwise no need to sample
				sample_prob = feats.data.new(batch_size).uniform_(0, 1)
				sample_mask = sample_prob < self.ss_prob
				if sample_mask.sum() == 0:
					it = seq[:, i].clone()
				else:
					sample_ind = sample_mask.nonzero().view(-1)
					it = seq[:, i].data.clone()
					prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
					it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
					it = Variable(it, requires_grad=False)
			else:
				it = seq[:, i].clone()  # <BOS>
			# break if all the sequences end
			if i >= 1 and seq[:, i].data.sum() == 0:
				break
			xt = self.embed(it)
			xt_mask = seq_mask[:,i].unsqueeze(1)
			output, state = self.lstmcore(xt,xt_mask,feats,pos_feats, state)  # output:(m,1000)
			outputs_hidden.append(output) # [(m,1000),(m,1000),...], total: seq_len+1 propare for RecNet
			output_word = F.log_softmax(self.logit(output), dim=1)
			output_category = F.log_softmax(self.classifer(output), dim=1)
			outputs.append(output_word)  # [ (m,nwords),(m,nwords), ... ], total: seq_len+1
			categories.append(output_category)

		# here insert RecNet======================
		if self.training and self.reconstruct is not None:
			dh = torch.cat([_.unsqueeze(1) for _ in outputs_hidden], 1).contiguous() # Variable (m,seq_len+1,1000)
			rec_feats = self.RecNet(dh, seq_mask) # (m, seq_len+1, feat_size)
			return torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous(), rec_feats
		#=========================================
		return torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous(), \
		       torch.cat([_.unsqueeze(1) for _ in categories], 1).contiguous()# (m,seq_len+1,nwords)

	def get_logprobs_state(self, it, feats, pos_feats, state): # just obtain the logprobs and state for one only word
		# 'it' is Variable contraining a word index
		batch_size = it.size(0)
		xt = self.embed(it)
		xt_mask = Variable( torch.ones([batch_size,1]).float(), requires_grad=False ).cuda()

		# def forward(self, xt, xt_mask, V, state)
		output, state = self.lstmcore(xt,xt_mask,feats,pos_feats, state)
		logprobs = F.log_softmax(self.logit(output),dim=1)

		return logprobs, state

	def sample_beam(self, feats, feat_masks, pos_feats, opt={}): #feats:(m,28,visual_size),feat_mask(m,28), pos_feats:(m,rnn_size)
		beam_size = opt.get('beam_size', 5)
		print('sampling with beam search ( beam_size = {} )'.format(beam_size))
		batch_size = feats.size(0)  # (m,28,1000)

		assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
		seq = torch.LongTensor( self.seq_length, batch_size).zero_()
		seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
		# lets process every image independently for now, for simplicity

		self.done_beams = [[] for _ in range(batch_size)]  # for every data, we create a empty list for it.
		for k in range(batch_size):  # for each data
			feat = feats[k,:,:]  #(28,1536)
			feat = feat.expand(beam_size,feat.size(0),feat.size(1)) # (beam_size,28,1536)
			feat_mask = feat_masks[k] #(28,)
			feat_mask = feat_mask.expand(beam_size,feat_mask.size(0)) # (beam_size,28)
			pos_feat = pos_feats[k]  # (512,)
			pos_feat = pos_feat.expand(beam_size, pos_feat.size(0))  # (beam_size, 512)
			state = self.init_hidden(feat,feat_mask) # state:( (1,beam_size,1000),(1,beam_size,1000) )

			# the first input, <bos>
			it = feats.data.new(beam_size).long().zero_()  # (beam_size,)
			xt = self.embed(Variable( it,requires_grad=False ))  # (beam_size,468)
			xt_mask = Variable( torch.ones([beam_size,1]).float(),requires_grad=False ).cuda()
			output,state = self.lstmcore( xt,xt_mask,feat, pos_feat, state )  # output: (beam_size,rnn_size)
			logprobs = F.log_softmax(self.logit(output), dim=1)  # (beam_size,n_words)

			# other inputs
			self.done_beams[k] = self.beam_search(state, logprobs, feats[k], pos_feats[k], opt=opt)  # beam_search() ???
			seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
			seqLogprobs[:, k] = self.done_beams[k][0]['logps']
		# return the samples and their log likelihoods
		return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)  # seq/seqLogprobs: (batch_size,seq_length)

	def sample(self, feats_rgb, feats_opfl, feats_rgb_pool, feats_opfl_pool, feat_mask, pos_feats, opt={}):
		sample_max = opt.get('sample_max', 1)
		beam_size = opt.get('beam_size', 1)
		temperature = opt.get('temperature', 1.0)
		# =========== encoder lstm on feats =================
		# feats_temporal = self.encoder(feats_rgb, feats_opfl, feat_mask)  # (m,28,rnn_size)
		feats = self.two_spatial_encoder(feats_rgb_pool, feats_opfl_pool, feat_mask)
		# feats = self.fc_encoder(feats_rgb, feat_mask)
		# feats = self.one_spatial_encoder(feats_rgb_pool, feat_mask)
		# feats = self.TS_fusion(feats_temporal, feats_spatial)

		if beam_size > 1:   # if that, it turns to beam search
			return self.sample_beam(feats,feat_mask, pos_feats, opt)
		else:
			print('sampling with greedy search')
		batch_size = feats.size(0)
		state = self.init_hidden(feats, feat_mask)
		seq = []
		seqLogprobs = []
		for t in range(self.seq_length + 1):  # seq_length + <bos>
			# if t == 0:
			#     feats_ = np.mean(feats.data.numpy(), axis=1, dtype=np.float32)  # (m,28,1536) --> (m,1536)
			#     feats_ = Variable(torch.from_numpy(feats_))  # (m,1536)
			#     xt = self.img_embed(feats_)
			# else:
			if t == 0: # input <bos>
				it = feats.data.new(batch_size).long().zero_()
			elif sample_max:  # greedy search
				sampleLogprobs, it = torch.max(logprobs.data, 1)
				it = it.view(-1).long()
			else:
				if temperature == 1.0:
					prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
				else:
					# scale logprobs by temperature
					prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
				it = torch.multinomial(prob_prev, 1).cuda()
				sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
				it = it.view(-1).long() # and flatten indices for downstream processing

			xt = self.embed(Variable(it, requires_grad=False))

			if t >= 1:
				# stop when all finished
				if t == 1:
					unfinished = it > 0
				else:
					unfinished = unfinished * (it > 0)
				if unfinished.sum() == 0:
					break
				it = it * unfinished.type_as(it)
				seq.append(it)   # at last, len(seq) is seq_length
				seqLogprobs.append(sampleLogprobs.view(-1))
			# using the mask of sequence
			if t == 0:
				xt_mask = Variable( torch.ones([batch_size,1]).float(),requires_grad=False ).cuda()
			else:
				xt_mask = Variable( unfinished.unsqueeze(-1).float(),requires_grad=False).cuda()  # (m,1)
			output, state = self.lstmcore(xt,xt_mask,feats,pos_feats, state)
			logprobs = F.log_softmax(self.logit(output), dim=1)  # the Probability distributions of the predicted word

		return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

class LanguageModelCriterion(nn.Module):  # compute and return mean loss for each word
	def __init__(self):
		super(LanguageModelCriterion, self).__init__()

	def forward(self, input, target, mask):  # input:(m,seq_len+1,n_words),target:(m,seq_len+1),mask:(m,seq_len+1)
		# truncate to the same size
		input = to_contiguous(input).view(-1,input.size(2))  #( m*(seq_len+1),n_words )
		target = torch.cat((target[:,1:],target[:,0].unsqueeze(1)),dim=1) # (m,seq_len+1)
		target = to_contiguous(target).view(-1,1)  # ( m*(seq_len+1),1 )
		mask = to_contiguous(mask).view(-1,1)  # ( m*(seq_len+1),1 )

		output = -1. * input.gather(1,target) * mask
		output = torch.sum(output) / torch.sum(mask)
		return output
		# target = target[:, :input.size(1)]   # (m,len_of_seq)
		# mask =  mask[:, :input.size(1)]   # (m,len_of_seq)
		# input = to_contiguous(input).view(-1, input.size(2))  # (m*len_of_seq, n_words)
		# target = to_contiguous(target).view(-1, 1)   # (m*len_of_seq, 1)
		# mask = to_contiguous(mask).view(-1, 1)   #  (m*len_of_seq, 1)
		# output = - input.gather(1, target) * mask   # (m*len_of_seq,1)*(m*len_of_seq,1)=>(m*len_of_seq,1)
		# output = torch.sum(output) / torch.sum(mask)  # a scalar, the mean probability for each word
		# return output

class ClassiferCriterion(nn.Module):
	''' Compute and return mean classifer loss '''
	def __init__(self):
		super(ClassiferCriterion, self).__init__()

	def forward(self, input, target, mask, class_mask=None):
		''' input:(m, seq_len+1, n_classes), target:(m, seq_len+1), mask:(m, seq_len+1) '''
		input = to_contiguous(input).view(-1, input.size(2))  # (m*(seq_len+1), n_classes)
		target = to_contiguous(target).view(-1, 1)  # (m*(seq_len+1), 1)
		mask = to_contiguous(mask).view(-1, 1)  # (m*(seq_len+1), 1)
		output = -1. * input.gather(1, target) * mask
		if class_mask is None:
			output = torch.sum(output) / torch.sum(mask)
		else:
			class_mask = to_contiguous(class_mask).view(-1, 1)  # (m*(seq_len+1), 1)
			output = output * class_mask
			output = torch.sum(output) / torch.sum(mask * class_mask)
		return output

class RewardCriterion(nn.Module):
	def __init__(self):
		super(RewardCriterion,self).__init__()

	def forward(self, input, seq, reward ): # input: sample logprobs; seq: sample results;
		input = to_contiguous(input).view(-1)  # inpupt:(mxseq_length,)
		reward = to_contiguous(reward).view(-1)  # reward:(mxseq_length,)
		mask = (seq > 0).float()  # mask:(m,seq_length)
		# mask = to_contiguous(mask).view(-1)  # (mxseq_length,)
		mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1) #(m,seq_length)=>(mxseq_length,)
		output = - input * reward * Variable(mask)
		output = torch.sum(output) / torch.sum(mask)
		return output

def RecnetLoss(rec_feats, feats, feats_mask, rec_strategy):
	"""
	:param rec_feats: Variable (m,seq_len+1,1536) from reconstructor
					  or (m, feat_k, 1536)
	:param feats: Variable (m, feat_k, 1536)
	:param feats_mask: Variable (m,feat_k)
	:return:
	"""
	if rec_strategy == 'global':
		rec_feats_mean = torch.sum(rec_feats, 1) / rec_feats.size(1)  # Variable (m,1536)
		feats_mean = torch.sum(feats, 1) / torch.sum(feats_mask, 1, keepdim=True) # Variable (m,1536)
		Ed = torch.sqrt( torch.sum( (rec_feats_mean - feats_mean) ** 2, 1) )  # Variable (m)
		return torch.mean(Ed)
	elif rec_strategy == 'local':
		Eds = torch.sqrt( torch.sum( ((rec_feats - feats) * feats_mask.unsqueeze(-1)) ** 2, -1) ) # (m, feat_K)
		return torch.sum(Eds) / torch.sum(feats_mask)
	elif rec_strategy == 'both':
		rec_feats_mean = torch.sum(rec_feats, 1) / rec_feats.size(1)  # Variable (m,1536)
		feats_mean = torch.sum(feats, 1) / torch.sum(feats_mask, 1, keepdim=True) # Variable (m,1536)
		Ed = torch.sqrt( torch.sum( (rec_feats_mean - feats_mean) ** 2, 1) )  # Variable (m)
		
	Eds = torch.sqrt( torch.sum( ((rec_feats - feats) * feats_mask.unsqueeze(-1)) ** 2, -1) ) # (m, feat_K)
	return torch.mean(Ed), torch.sum(Eds) / torch.sum(feats_mask)
		

if __name__ == '__main__':
	opt = myopts.parse_opt()

	mydset_valid = custom_dset_valid()
	myloader_valid = torch.utils.data.DataLoader(mydset_valid, batch_size=2, collate_fn=collate_fn)
	model = SAModel(opt=opt)
	model.cuda()
	model.train()
	crit = LanguageModelCriterion()
	i = 0
	for data,cap,cap_mask,feat,feat_mask,lens,gts in myloader_valid:
		cap = Variable(cap, requires_grad=False).cuda()
		cap_mask = Variable(cap_mask, requires_grad=False).cuda()
		feat = Variable(feat, requires_grad=False).cuda()
		feat_mask = Variable(feat_mask, requires_grad=False).cuda()
		out = model(feat,feat_mask,cap,cap_mask)
		seq,seqlogprob = model.sample(feat,feat_mask,{'sample_max':0})
		# seq,seqlogprob = model.sample_beam(feat,feat_mask)
		print(seq)
		break
