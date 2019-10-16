from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

#import myopts
import data_io
from data_io import *
from SAModel import *
#import myopts
import cPickle
import eval_utils
import argparse
import os


# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
				help='path to model to evaluate')
parser.add_argument('--infos_path', type=str, default='',
				help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
				help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
				help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
				help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
				help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=3,
				help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
				help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# misc
parser.add_argument('--id', type=str, default='',
				help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

opt = parser.parse_args()


# Load infos
with open(opt.infos_path) as f:
	infos = cPickle.load(f)

if opt.batch_size == 0:
	opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
	opt.id = infos['opt'].id

ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
for k in vars(infos['opt']).keys():
	if k not in ignore:
		if k in vars(opt):
			assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
		else:
			vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model
print(opt)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
# load train/valid/test data
# mytrain_dset, myvalid_dset, mytest_dset = loaddset(opt)
mytest_dset = test_dataio(opt)
# set my model
model = SAModel(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = LanguageModelCriterion()
classify_crit = ClassiferCriterion()

print ("testing starts ...")
test_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, classify_crit, mytest_dset, vars(opt), True)

test_result = {}
test_result['test_loss'] = test_loss
test_result['predictions'] = predictions
test_result['scores'] = lang_stats
#with open(os.path.join(opt.checkpoint_path,'test_result.pkl'), 'wb') as f:
#	cPickle.dump(test_result, f)
#print ("testing finish !\n")


print('loss: ', test_loss)
if lang_stats:
	print(lang_stats)
