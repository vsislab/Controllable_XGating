# from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import data_io

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
from collections import OrderedDict
import myutils

def eval_split(model, crit, classify_crit, my_dset, eval_kwargs={}): # eval_kwargs is dict type of opts
	verbose = eval_kwargs.get('verbose', True)
	lang_eval = eval_kwargs.get('language_eval', 0)
	beam_size = eval_kwargs.get('beam_size', 1)
	weight_class = eval_kwargs.get('weight_class', 0.0)

	# Make sure in the evaluation mode
	model.eval()

	loss_sum = 0
	loss_evals = 1e-8
	predictions = []
	gts = []
	caps = data_io.get_caps(eval_kwargs['data_path'])

	myloader = DataLoader(my_dset, batch_size=eval_kwargs.get('batch_size',64), collate_fn=data_io.collate_fn,shuffle=False)
	for data, cap, cap_mask, cap_classes, class_mask, feat1, feat2, feat3, feat4, feat_mask, pos_feat, lens, groundtruth, image_ids in myloader:
		tmp = [cap, cap_mask, cap_classes, class_mask, feat1, feat2, feat3, feat4, feat_mask, pos_feat]
		tmp = [Variable( _ , volatile=True).cuda() for _ in tmp]
		cap, cap_mask, cap_classes, class_mask, feat1, feat2, feat3, feat4, feat_mask, pos_feat = tmp
		# forward the model to get loss
		out, category = model(feat1, feat2, feat3, feat4, feat_mask, pos_feat, cap, cap_mask)
		loss_language = crit(out, cap, cap_mask).data[0]
		loss_category = classify_crit(category, cap_classes, cap_mask, class_mask).data[0]
		#loss = (1. - weight_class) * loss_language + weight_class * loss_category
		loss = loss_language + weight_class * loss_category
		loss_sum = loss_sum + loss
		loss_evals = loss_evals + 1
		# forward the model to also get generated samples for each image
		seq, seqLogprobs = model.sample(feat1, feat2, feat3, feat4, feat_mask, pos_feat, eval_kwargs)
		if 'cuda' in str(type(seq)):
			seq = seq.cpu()
		if 'cuda' in str(type(seqLogprobs)):
			seqLogprobs = seqLogprobs.cpu()
		# set_trace()
		itow = my_dset.get_itow()
		sents = myutils.decode_sequence(itow, seq) # list

		for k, (sent,seqLogprob) in enumerate(zip(sents, seqLogprobs)):
			entry = {'image_id': image_ids[k], 'caption': sent, 'seqLogprob': seqLogprob.numpy(), }
			predictions.append(entry)
			gts.append(caps[image_ids[k]])
			# if verbose:
			#     print('image %s: %s' % (entry['image_id'], entry['caption']))
	if verbose:
		for x in predictions[:10]:
			print('image %s: %s' % (x['image_id'], x['caption']))

	lang_stats = None
	if lang_eval == 1:
		predictions_ = []
		for x in predictions:
			predictions_.append(x['caption'])
		gts_ = []
		for x in gts:
			gts_tmp = []
			for i in x:
				try:
					#i['tokenized'].encode()
					gts_tmp.append(i['tokenized'])
				except:
					continue
			gts_.append(gts_tmp)

		lang_stats = language_eval(predictions_,gts_) # lang_states = scores

	# Switch back to training mode
	model.train()
	return loss_sum / loss_evals, predictions, lang_stats

def language_eval(sample_seqs, gt_seqs):# sample_seqs:list[[x,x],[x,x],...], gt_seqs:list[[list1,list2,...],[list1,list2,...],...]
	import sys
	#sys.path.append("caption-eval")
        sys.path.append("coco-caption/pycocoevalcap/")
	from bleu.bleu import Bleu
	from cider.cider import Cider
	from meteor.meteor import Meteor
	from rouge.rouge import Rouge

	assert len(sample_seqs) == len(gt_seqs),"number of eval data is different"
	res = OrderedDict()  # res: {0:[xx],1:[xx],...}
	for i in range(len(sample_seqs)): # for each data(sent)
		res[i] = [sample_seqs[i]]

	gts = OrderedDict() # gts: {0:[sent1,sent2,...],1:[sent1,sent2,...], ...}
	for i in range(len(gt_seqs)):
		gts[i] = [gt_seqs[i][j] for j in range(len(gt_seqs[i]))]

	res = {i: res[i] for i in range(len(sample_seqs))}
	gts = {i: gts[i] for i in range(len(gt_seqs))}

	avg_bleu_score, bleu_scores = Bleu(4).compute_score(gts, res)
	avg_cider_score, cider_scores = Cider().compute_score(gts, res)
	avg_meteor_score, meteor_scores = Meteor().compute_score(gts, res)
	avg_rouge_score, rouge_scores = Rouge().compute_score(gts, res)

	print(" BLEU1:{}\n BLEU2:{}\n BLEU3:{}\n BLEU4:{}\n METEOR:{}\n ROUGE:{}\n CIDEr:{}\n"\
		.format(avg_bleu_score[0], avg_bleu_score[1], avg_bleu_score[2], avg_bleu_score[3], \
				avg_meteor_score, avg_rouge_score, avg_cider_score))

	return {'BLEU':avg_bleu_score, 'METEOR':avg_meteor_score, 'ROUGE':avg_rouge_score, 'CIDEr':avg_cider_score}

if __name__ == "__main__":
	language_eval()
	# encoder.FLOAT_REPR = lambda o: format(o, '.3f')
