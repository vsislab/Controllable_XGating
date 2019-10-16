import collections
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
import os
import sys
import time
#sys.path.append("cider")
#sys.path.append("caption-eval")
sys.path.append("coco-caption/pycocoevalcap")
#from pyciderevalcap.ciderD.ciderD import CiderD
from bleu.bleu import Bleu
from cider.cider import Cider
from meteor.meteor import Meteor
from rouge.rouge import Rouge

CiderD_scorer = None
#CiderD_scorer = CiderD(df='corpus')
def init_cider_scorer(reward_type):
    global CiderD_scorer
    # CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    if reward_type == 'BLEU':
        CiderD_scorer = CiderD_scorer or Bleu()
    elif reward_type == 'METEOR':
        CiderD_scorer = CiderD_scorer or Meteor()
    elif reward_type == 'ROUGE':
        CiderD_scorer = CiderD_scorer or Rouge()
    elif reward_type == 'CIDEr':
        CiderD_scorer = CiderD_scorer or Cider()
    # if reward_type == 'MIX':
    #     CiderD_scorer = for

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def array_to_str(arr): # arr:(x,), turn an arr to a sentence
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, feat1, feat2, feat_mask, groundtruth, gen_result):  # gen_result:(m,seq_length), groundtruth is a list, elem of which is tensor
    batch_size = gen_result.size(0)
    # seq_per_img = batch_size // len(data['gts'])  # what the true meaning of seq_per_img?

    # get greedy decoding baseline
    greedy_res, _ = model.sample(Variable(feat1.data, volatile=True),
                                 Variable(feat2.data, volatile=True),
                                 Variable(feat_mask.data, volatile=True)) # sample_max=1

    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()  # got by sampling
    greedy_res = greedy_res.cpu().numpy()  # got by greedy
    for i in range(batch_size):  # res:0~batch_size-1 is sample results, batch_size~end is greedy results
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(batch_size): # for each data  ==> tensor (x,y)
        gts[i] = [ array_to_str(groundtruth[i][j]) for j in range(len(groundtruth[i]))] # for each data, we got the token ground truth

    # res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}
    # gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    print('Cider scores:', _)
    # print type(scores)
    # print ('sleep')
    # time.sleep(100)
    
    if str(CiderD_scorer).split('.')[1] == 'bleu':
        scores = scores[-1]
    scores = np.array(scores)  # for meteor. if Cider, commond it !
    scores = scores[:batch_size] - scores[batch_size:]  # R(sample)-R(greedy)
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)  # rewards:(m,seq_length)
    return rewards

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

# Input: seq, NxD numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N): # for each data
        txt = ''
        for j in range(D): # for each word in a data
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix]
            else:
                break
        out.append(txt)
    return out

if __name__ == '__main__':
    CiderD_scorer = CiderD(df='coco-val')
    print CiderD_scorer
