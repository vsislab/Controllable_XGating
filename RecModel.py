import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# from data_io import *
# import myopts
# from CaptionModel import CaptionModel

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RecCore(nn.Module):
    def __init__(self, opt):
        super(RecCore,self).__init__()
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        # basic setting
        self.rec_strategy = opt.rec_strategy
        self.feat_K = opt.feat_K
        # rnn input and output size
        self.input_size = opt.rnn_size
        self.output_size = opt.feat_size
        # drop out
        self.drop_prob_lm = opt.drop_prob_lm
        # LSTM
        self.i2h = nn.Linear(self.input_size, 4 * self.output_size)
        self.h2h = nn.Linear(self.output_size, 4 * self.output_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        if self.rec_strategy == 'global':
            self.d2h = nn.Linear(self.input_size, 4 * self.output_size)
        # Attention
        if self.rec_strategy == 'local' or self.rec_strategy == 'both':
            self.att_size = opt.att_size
            self.h2a = nn.Linear(self.output_size, self.att_size)  # (feat_size,atten_size)
            self.v2a = nn.Linear(self.input_size, self.att_size)  # (rnn_size, atten_size)
            self.a2w = nn.Linear(self.att_size, 1)  # (atten_size, 1)

    def forward(self, dh, cap_mask, state):
        """
        :param dh: Variable (m, seq_len+1, rnn_size)
        :param cap_mask: Variable, (m, seq_len+1)
        :param state: Variable ((1,m,feats_size),(1,m,feats_size)), the init states
        :return:
        """
        outputs = []
        if self.rec_strategy == 'global':
            dh_mean = self.mean_dh(dh, cap_mask)  # Variable (m, rnn_size)
            nsteps = dh.size(1)  # seq_len+1
            for i in range(nsteps):
                all_input_sums = self.i2h(dh[:,i,:]) + self.h2h(state[0][-1]) + self.d2h(dh_mean) # (m,4*feat_size)
                sigmoid_chunk = all_input_sums.narrow(dimension=1, start=0, length=3 * self.output_size)
                sigmoid_chunk = F.sigmoid(sigmoid_chunk)
                in_gate = sigmoid_chunk.narrow(1, 0, self.output_size)
                forget_gate = sigmoid_chunk.narrow(1,self.output_size, self.output_size)
                out_gate = sigmoid_chunk.narrow(1, 2 * self.output_size, self.output_size)

                tanh_chunk = all_input_sums.narrow(1, 3*self.output_size, self.output_size)
                in_transform = F.tanh(tanh_chunk)
                next_c = forget_gate * state[1][-1] + in_gate * in_transform # (m,feat_size)
                next_h = out_gate * F.tanh(next_c)  # Variable (m,feat_size)
                next_h = self.dropout(next_h)

                outputs.append(next_h)
                state = (next_h.unsqueeze(0), next_c.unsqueeze(0)) # ((1,m,feat_size),(1,m,feat_size))
            # in the end, force on outputs
            return torch.cat( [ _.unsqueeze(1) for _ in outputs], 1).contiguous() # Variable (m,seq_len+1,feat_size)
        elif self.rec_strategy == 'local' or self.rec_strategy == 'both':
            # dh_mean = self.mean_dh(dh, cap_mask)  # Variable (m, rnn_size)
            nsteps = self.feat_K  # seq_len+1
            for i in range(nsteps):
                alpha = self.h2a(state[0][-1]).unsqueeze(1) + self.v2a(dh)  #(m,att_size)+(m,seq_len+1,att_size)
                alpha = self.a2w( F.tanh(alpha)) # (m, seq_len+1, 1)
                alpha = F.softmax(alpha, dim=1) # (m, seq_len+1, 1)
                dh_att = torch.sum(dh * alpha, dim=1)  # (m, rnn_size)

                all_input_sums = self.i2h(dh_att) + self.h2h(state[0][-1])  # (m, 4*output_size)
                sigmoid_chunk = all_input_sums.narrow(1, 0, 3*self.output_size)
                sigmoid_chunk = F.sigmoid(sigmoid_chunk)
                in_gate = sigmoid_chunk.narrow(1, 0, self.output_size)
                forget_gate = sigmoid_chunk.narrow(1, self.output_size, self.output_size)
                out_gate = sigmoid_chunk.narrow(1, 2*self.output_size, self.output_size)

                tanh_chunk = all_input_sums.narrow(1, 3*self.output_size, self.output_size)
                in_transform = F.tanh(tanh_chunk)

                next_c = forget_gate * state[1][-1] + in_gate * in_transform  # (m, output_size)
                next_h = out_gate * F.tanh(next_c)  # (m, output_size)
                next_h = self.dropout(next_h)

                outputs.append(next_h)
                state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous() # Variable (m, feat_k, output_size)




    def mean_dh(self,dh,cap_mask):
        '''
        similar to init_hidden
        '''
        dh_ = torch.from_numpy( np.sum( dh.cpu().data.numpy(), axis=1, dtype=np.float32)) # (m, rnn_size)
        mask_ = torch.from_numpy( np.sum( cap_mask.cpu().data.numpy(),axis=1,dtype=np.float32))
        dh_mean = (dh_ / mask_.unsqueeze(-1)) # Tensor (m,rnn_size)
        return Variable(dh_mean, requires_grad=False).cuda()


class RecModel(nn.Module):
    def __init__(self,opt):
        super(RecModel,self).__init__()
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        self.rec_strategy = opt.rec_strategy
        self.feat_K = opt.feat_K  # default=28
        self.input_size = opt.rnn_size  # 1000
        self.output_size = opt.feat_size  # 1536
        self.rec_embed_h = nn.Linear(self.input_size, self.output_size) #(1000. 1536)
        self.rec_embed_c = nn.Linear(self.input_size, self.output_size)
        self.reccore = RecCore(opt)
        self.rec_mlp = nn.Linear(self.output_size, self.output_size)
        self.init_weight()

    def init_weight(self):
        initrange = 0.1
        self.rec_mlp.bias.data.fill_(0)
        self.rec_mlp.weight.data.uniform_(-initrange,initrange)
    def init_hidden(self, dh, cap_mask):
        """
        inputs:
            dh: Variable, (m, seq_len+1, rnn_size)
            cap_mask: Variable, (m, seq_len+1)
        return:
            a couple of 2 Variables, donates init_h and init_c for LSTM
            ((1, m, feat_size),(1, m, feat_size))
        """
        dh_ = torch.from_numpy( np.sum( dh.cpu().data.numpy(), axis=1, dtype=np.float32)) # Tensor,(m, rnn_size)
        mask_ = torch.from_numpy( np.sum( cap_mask.cpu().data.numpy(), axis=1, dtype=np.float32))# Tensor,(m,)
        dh_mean = (dh_ / mask_.unsqueeze(-1)).unsqueeze(0) #Tensor,(1,m,rnn_size)
        dh_mean = Variable(dh_mean, requires_grad=False).cuda()
        return (self.rec_embed_h(dh_mean), self.rec_embed_c(dh_mean))

    def forward(self, dh, cap_mask ):
        """
        :param dh: Variable,(m, seq_len+1, rnn_size), all hidden state of decoder
        :param cap_mask: Variable, (m, seq_len+1)
        :return:
        """
        state = self.init_hidden(dh, cap_mask) # Variable ((1,m,feats_size),(1,m,feats_size))
        if self.rec_strategy == 'global':
            rec_feats = self.reccore(dh,cap_mask, state) # Variable (m,seq_len+1,feat_size)
        elif self.rec_strategy == 'local' or self.rec_strategy == 'both':
            rec_feats = self.reccore(dh,cap_mask, state) # Variable (m, feat_K, feat_size)

        # use MLP to rec_feats
        tmp0, tmp1, tmp2 = rec_feats.size(0), rec_feats.size(1), rec_feats.size(2)
        rec_feats = self.rec_mlp(rec_feats.view(-1, tmp2)) # (m x seq_len+1(or feat_K), feat_size)
        return rec_feats.view(tmp0, tmp1, tmp2)
