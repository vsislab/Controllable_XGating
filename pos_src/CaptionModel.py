# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_search(self, state, logprobs, feat, *args, **kwargs): # logprobs:(beam_size,n_words),feat:(28,1536)
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity  (beam_size, n_words)
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams  (seq_length,beam_size)
            #beam_seq_logprobs: tensor contanining the beam logprobs  (seq_length,beam_size)
            #beam_logprobs_sum: tensor contanining joint logprobs  (beam_size)
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True) # descending sort, ys:(beam_size,nwords), ix:(beam_size,nwords)
            candidates = []
            cols = min(beam_size, ys.size(1))  # ys.size(1) = n_words
            rows = beam_size
            if t == 0: # from <BOS>
                rows = 1
            for c in range(cols): # for each column (word, essentially) --which word
                for q in range(rows): # for each beam expansion --which beam
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]  # state:((1,beam_size,1536),(1,beam_size,1536))
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1: # save the previous beam_seq and beam_seq_logporbs
            # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()   # beam_seq_prev (t,beam_size)
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()   # beam_seq_logprobs_prev (t,beam_size)
            for vix in range(beam_size):  # for each beam
                v = candidates[vix]  # take the top beam_size information, as the topper, the better
                #fork beam index q into index vix
                if t >= 1: # update previous beam_seq
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)): # for state_ix in range(2):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation.
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # start beam search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 5)

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()  # (seq_length,beam_size) (30,5)
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_() # (seq_length,beam_size)
        beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam, it decides the best sequence
        done_beams = []

        for t in range(self.seq_length):  # for each word, for t in range(30):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float() # lets go to CPU for more efficiency in indexing operations (beam_size,n_words)
            # suppress UNK tokens in the decoding, the index of 'UNK' is 1. So the probs of 'UNK' are extremely low
            logprobsf[:,1] =  logprobsf[:,1] - 1000  # logprobs:(beam_size,n_words)

            beam_seq,\
            beam_seq_logprobs,\
            beam_logprobs_sum,\
            state,\
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        state)

            for vix in range(beam_size):   # for each beam
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(), 
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]  # (beam_size,)
            feat_ = feat.expand( it.size(0), feat.size(0), feat.size(1) ) # (beam_size,28,1536)
            # feat_ = np.mean(feat.data.numpy(), axis=1, dtype=np.float32)  # feat:(28,1536)
            logprobs, state = self.get_logprobs_state(Variable(it.cuda()), feat_, *(args + (state,)))  # the ',' in (state,) ensure it's tuple

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size] # choose the top beam_size results
        return done_beams
