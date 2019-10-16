# coding:utf-8
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import sys
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import pickle
import numpy as np
import time
import myopts
from collections import OrderedDict
import random
import h5py


def load_pkl(pkl_file):
	f = open(pkl_file,'rb')
	try:
		result = pickle.load(f)
	finally:
		f.close()
	return result

def get_sub_frames(frames, K):
	# from all frames, take K of them, then add end of video frame
	if len(frames) < K:
		# frames_ = np.zeros([K, frames.shape[1]])
		# frames_[:len(frames),:] = frames
		temp_zeros = np.zeros([K-frames.shape[0], frames.shape[1]])
		frames_ = np.concatenate((frames,temp_zeros), axis=0)
	else:
		index = np.linspace(0,len(frames),K,endpoint=False,dtype=int)
		frames_ = frames[index]
	return frames_

def get_sub_pool_frames(frames, K):
	# for pool features
	assert len(frames.shape) == 4, "shape of pool features should be 4 dims"
	if len(frames) < K:
		frames_shape = list(frames.shape)
		temp_zeros = np.zeros([K - frames_shape[0]] + frames_shape[1:])
		frames_ = np.concatenate((frames, temp_zeros), axis=0)
	else:
		index = np.linspace(0, len(frames), K, endpoint=False, dtype=int)
		frames_ = frames[index]
	return frames_

def filt_word_category(cate_pkl, words):
	# load the category file
	category_words = load_pkl(cate_pkl)  # {NN:[cat, dog, pig]}
	# make word and category conpends  {cat:NN, take:VB, ...}
	words_category = {}
	for category, wordlist in category_words.items():
		for word in wordlist:
			words_category[word] = category
	# give each category a ID
	category_name_un = ['FW', '-LRB-', '-RRB-', 'LS']  # 不明白
	category_name_vb = ['VB', 'VBD', 'VBP', 'VBG', 'VBN', 'VBZ']    # 动词
	category_name_nn = ['NN', 'NNS', 'NNP']    # 名词
	category_name_jj = ['JJ', 'JJR', 'JJS']     # 形容词
	category_name_rb = ['RB', 'RBS', 'RBR', 'WRR', 'EX']     # 副词
	category_name_cc = ['CC']  # 连词
	category_name_pr = ['PRP', 'PRP$', 'WP', 'POS', 'WP$']  # 代词
	category_name_in = ['IN', 'TO']  # 介词
	category_name_dt = ['DT', 'WDT', 'PDT']  # 冠词
	category_name_rp = ['RP', 'MD']  # 助词
	category_name_cd = ['CD']  # 数字
	category_name_sy = ['SYM', ':', '``', '#', '$']  # 符号
	category_name_uh = ['UH']  # 叹词

	all_category = category_words.keys()
	category_id = {}  # {VB:2, VBS:2, NN:3, NNS:3 ...}
	for category in all_category:
		if category in category_name_vb:
			category_id[category] = 2
		elif category in category_name_nn:
			category_id[category] = 3
		elif category in category_name_jj:
			category_id[category] = 4
		elif category in category_name_rb:
			category_id[category] = 5
		elif category in category_name_cc:
			category_id[category] = 6
		elif category in category_name_pr:
			category_id[category] = 7
		elif category in category_name_in:
			category_id[category] = 8
		elif category in category_name_dt:
			category_id[category] = 9
		elif category in category_name_rp:
			category_id[category] = 10
		elif category in category_name_cd:
			category_id[category] = 11
		elif category in category_name_sy:
			category_id[category] = 12
		elif category in category_name_uh:
			category_id[category] = 13
		else:
			category_id[category] = 1
	# turn words' category from str to ID
	all_words_in_category = words_category.keys()
	filted_words_categoryid = {}  # {'<EOS>':0, '<UNK>':1, 'cat':3, 'take':2, 'log_vir':1}
	for key in words:
		if key in all_words_in_category:
			the_key_category = words_category[key]
			filted_words_categoryid[key] = category_id[the_key_category]
		else:
			filted_words_categoryid[key] = 1
	filted_words_categoryid['<EOS>'] = 0
	filted_words_categoryid['<UNK>'] = 1
	# take out the unmasked category ids
	unmasked_categoryid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # VB, NN, JJ, and RB needn't be masked
	return filted_words_categoryid, words_category, category_id, category_words, unmasked_categoryid


def get_caps(path):
	return load_pkl(os.path.join(path, 'CAP.pkl'))

def get_nwords(path):
	return len(load_pkl(os.path.join(path, 'worddict.pkl'))) + 2

def get_nclasses(path):
	return 14 #len(load_pkl(os.path.join(path, 'category.pkl'))) + 2

class custom_dset_train(Dataset):
	def get_itow(self):
		# wtoi = load_pkl( wtoi_path )
		# wtoi['<EOS>'] = 0
		# wtoi['UNK'] = 1
		wtoi = self.wtoi

		itow = {}
		for key,val in wtoi.iteritems():
			itow[val] = key
		return itow

	def __init__(self, train_pkl, cap_pkl, cate_pkl, feat_path1, feat_path2, pos_path, wtoi_path, nwords=10000, K=28, opt=None):
		self.nwords = nwords
		self.K = K
		data_name_list = load_pkl( train_pkl ) #[vid1_0,vid1_2, ...]
		caps = load_pkl( cap_pkl )
		wtoi = load_pkl( wtoi_path )
		wtoi['<EOS>'] = 0
		wtoi['UNK'] = 1  # because 'wtoi_path' start from 2.
		wtoi_keys = wtoi.keys()
		self.wtoi = wtoi
		filted_class, words_class, class_id, class_words, unmasked_classid = filt_word_category(cate_pkl, wtoi)
		self.category = filted_class
		category_keys = self.category.keys()

		temp_cap_list = []
		for i,ID in enumerate(data_name_list):
			vidid, capid = ID.split('_') # vidid='vid1', capid=0
			temp_cap_list.append(caps[vidid][int(capid)])

		data_list = []
		cap_list = []
		for data,cap in zip(data_name_list,temp_cap_list):
			token = cap['tokenized'].split()
			if 0 < len(token) <= opt.seq_length:
				data_list.append(data)
				new_cap = {}
				# new_cap['image_id'] = cap['image_id']
				new_cap['caption'] = cap['caption']
				new_cap['tokenized'] = cap['tokenized']
				new_cap['numbered'] = [ wtoi[w] if w in wtoi_keys else 1 for w in token]
				new_cap['category'] = [self.category[w] if w in category_keys else 1 for w in token]
				new_cap['category_mask'] = [1 if index in unmasked_classid else 0 for index in new_cap['category']]
				cap_list.append( new_cap )

		gts_list = []
		for i,ID in enumerate(data_list):
			sub_gts_list = []
			vidid, _ = ID.split('_')
			for cap in caps[vidid]:
				token = cap['tokenized'].split()
				numbered = [ wtoi[w] if w in wtoi_keys else 1 for w in token ]
				sub_gts_list.append(numbered)
			sub_gts_list.sort(key=lambda x: len(x),reverse=True)
			tmp_gts_arr = np.zeros([len(sub_gts_list),len(sub_gts_list[0])],dtype=int)
			for x in range(len(sub_gts_list)):
				tmp_gts_arr[x,:len(sub_gts_list[x])] = sub_gts_list[x]
			gts_list.append(tmp_gts_arr)

		self.data_list = data_list  #[vid1_0,vid1_2, ...]
		self.cap_list = cap_list    #[{},{},...]
		self.gts_list = gts_list    #[[str,str,...],...]
		self.feat_path1 = feat_path1
		self.feat_path2 = feat_path2
		self.pos_path = pos_path
		print('got %d data and %d labels'%(len(self.data_list),len(self.cap_list)))

	def __getitem__(self, index):
		data = self.data_list[index]
		cap = self.cap_list[index]['numbered']
		cap_class = self.cap_list[index]['category']
		class_mask = self.cap_list[index]['category_mask']
		gts = self.gts_list[index]

		# feat = np.load(self.feat_path +'train/' + data.split('_')[0] + '.npy')
		feat1 = self.feat_path1[data.split('_')[0]][:]
		feat1 = get_sub_frames(feat1, self.K)
		feat1 = torch.from_numpy(feat1).float()   # turn numpy data to Tensor

		feat2 = self.feat_path2[data.split('_')[0]][:]
		feat2 = get_sub_frames(feat2, self.K)
		feat2 = torch.from_numpy(feat2).float()
		
                # feat_mask = (torch.sum(feat, dim=1, keepdim=True) != 0).float().transpose(1,0) # for fc features
		feat_mask = (torch.sum(feat1.view(feat1.size(0), -1), dim=1, keepdim=True) != 0).float().transpose(1,0)

		pos_feat = self.pos_path[data.split('_')[0]]['states'][:] 
		pos_feat = pos_feat[-1]
		pos_feat = torch.from_numpy(pos_feat).float()

		return data,cap,cap_class,class_mask,feat1, feat2, feat_mask, pos_feat, gts

	def __len__(self):
		return len(self.cap_list)


class custom_dset_test(Dataset):
	def get_itow(self):
		# wtoi = load_pkl( wtoi_path )
		# wtoi['<EOS>'] = 0
		# wtoi['UNK'] = 1
		wtoi = self.wtoi

		itow = {}
		for key,val in wtoi.iteritems():
			itow[val] = key
		return itow

	def __init__(self, test_pkl, cap_pkl, cate_pkl, feat_path1, feat_path2, pos_path, wtoi_path, nwords=10000, K=28, opt=None):
		self.nwords = nwords
		self.K = K
		data_name_list = load_pkl( test_pkl ) #[vid1_0,vid1_2, ...]
		caps = load_pkl( cap_pkl )
		wtoi = load_pkl( wtoi_path )
		wtoi['<EOS>'] = 0
		wtoi['UNK'] = 1  # because 'wtoi_path' start from 2.
		wtoi_keys = wtoi.keys()
		self.wtoi = wtoi
		filted_class, words_class, class_id, class_words, unmasked_classid = filt_word_category(cate_pkl, wtoi)
		self.category = filted_class
		category_keys = self.category.keys()

		temp_cap_list = []
		for i,ID in enumerate(data_name_list):
			vidid, capid = ID.split('_') # vidid='vid1', capid=0
			temp_cap_list.append(caps[vidid][int(capid)])

		data_list = []
		cap_list = []
		for data,cap in zip(data_name_list,temp_cap_list):
			token = cap['tokenized'].split()
			if 0 < len(token) <= opt.seq_length:
				data_list.append(data)
				new_cap = {}
				new_cap['caption'] = cap['caption']
				new_cap['tokenized'] = cap['tokenized']
				new_cap['numbered'] = [ wtoi[w] if w in wtoi_keys else 1 for w in token]
				new_cap['category'] = [self.category[w] if w in category_keys else 1 for w in token]
				new_cap['category_mask'] = [1 if index in unmasked_classid else 0 for index in new_cap['category']]
				cap_list.append(new_cap)

		tmp_vid = []
		tmp_vidname = []
		tmp_cap = []
		for data, cap in zip(data_list, cap_list):
			if data.split('_')[0] not in tmp_vid:
				tmp_vid.append(data.split('_')[0])
				tmp_vidname.append(data)
				tmp_cap.append(cap)
		data_list = tmp_vidname
		cap_list = tmp_cap

		gts_list = []
		for i, ID in enumerate(data_list):
			sub_gts_list = []
			vidid, _ = ID.split('_')
			for cap in caps[vidid]:
				token = cap['tokenized'].split()
				numbered = [wtoi[w] if w in wtoi_keys else 1 for w in token]
				sub_gts_list.append(numbered)
			sub_gts_list.sort(key=lambda x: len(x), reverse=True)
			tmp_gts_arr = np.zeros([len(sub_gts_list), len(sub_gts_list[0])], dtype=int)
			for x in range(len(sub_gts_list)):
				tmp_gts_arr[x, :len(sub_gts_list[x])] = sub_gts_list[x]
			gts_list.append(tmp_gts_arr)

		self.data_list = data_list  #[vid1_0,vid1_2, ...]
		self.cap_list = cap_list    #[{},{},...]
		self.gts_list = gts_list
		self.feat_path1 = feat_path1
		self.feat_path2 = feat_path2
		self.pos_path = pos_path
		print('got %d data and %d labels'%(len(self.data_list),len(self.cap_list)))

	def __getitem__(self, index):
		data = self.data_list[index]
		cap = self.cap_list[index]['numbered']
		cap_class = self.cap_list[index]['category']
		class_mask = self.cap_list[index]['category_mask']
		gts = self.gts_list[index]

		# feat1 = np.load(self.feat_path1 +'test/' + data.split('_')[0] + '.npy')
		feat1 = self.feat_path1[data.split('_')[0]][:]
		feat1 = get_sub_frames(feat1, self.K)
		feat1 = torch.from_numpy(feat1).float()  # turn numpy data to Tensor

		# feat2 = np.load(self.feat_path2 + 'test/' + data.split('_')[0] + '.npy')
		feat2 = self.feat_path2[data.split('_')[0]][:]
		feat2 = get_sub_frames(feat2, self.K)
		feat2 = torch.from_numpy(feat2).float()
		# feat_mask = (torch.sum(feat, dim=1, keepdim=True) != 0).float().transpose(1,0)
		feat_mask = (torch.sum(feat1.view(feat1.size(0), -1), dim=1, keepdim=True) != 0).float().transpose(1, 0)

		pos_feat = self.pos_path[data.split('_')[0]]['states'][:] 
		pos_feat = pos_feat[-1] 
		pos_feat = torch.from_numpy(pos_feat).float()
		return data,cap,cap_class,class_mask,feat1,feat2,feat_mask, pos_feat,gts

	def __len__(self):
		return len(self.cap_list)

def collate_fn(batch): # batch: ( data, cap, feat)
	batch.sort(key=lambda x:len(x[1]), reverse=True)
	data, cap, cap_class, class_mask, feat1, feat2, feat_mask, pos_feat,gts = zip(*batch)  # gts:

	max_len = len(cap[0])  # the first captions must has be the longest
	feats1 = torch.stack(feat1, dim=0)
	feats2 = torch.stack(feat2, dim=0)
	feat_mask = torch.cat(feat_mask,dim=0)
	pos_feat = torch.stack(pos_feat, dim=0)  # (m, rnn_size)

	caps = []
	lens = []
	caps_mask = torch.zeros([len(cap),max_len+1])  # (m, max_len+1)
	for i in range(len(cap)):  #  for each data in the batch:
		temp_cap = [0]*(max_len+1)
		temp_cap[1:len(cap[i])+1] = cap[i]  # here print the original and temp_cap, for compared
		caps.append(temp_cap)
		caps_mask[i,:len(cap[i])+1] = 1
		lens.append(len(cap[i]))
	caps = torch.LongTensor(caps)

	# collect word category
	cap_classes = []
	class_masks = []
	# class_mask = torch.zeros([len(cap_classes), max_len+1])  # (m, max_len+1)
	# print(len(cap_class), len(class_mask))
	for i in range(len(cap_class)): # for each data in batch
		temp_cap_class = [0] * (max_len+1)
		temp_cap_class[0:len(cap_class[i])] = cap_class[i]
		cap_classes.append(temp_cap_class)
		temp_class_mask = [0] * (max_len + 1)
		temp_class_mask[0:len(class_mask[i])] = class_mask[i]
		temp_class_mask[len(class_mask[i])] = 1
		class_masks.append(temp_class_mask)
		# class_mask[i, 0:len(class_mask[i])] = torch.from_numpy(class_mask[i])
		# class_mask[i, len(class_mask[i])] = 1
	cap_classes = torch.LongTensor(cap_classes)  # (m, max_len+1)
	class_masks = torch.FloatTensor(class_masks)


	gts = [torch.from_numpy(x).long() for x in gts]
	# batch_feat = torch.cat(batch_feat,dim=0)
	image_id = [i.split('_')[0] for i in data]

	return data,caps,caps_mask,cap_classes,class_masks, feats1,feats2,feat_mask,pos_feat,lens,gts,image_id  # feat:(m,28,1536)

def loaddset(opt):
	train_pkl = os.path.join(opt.data_path, 'valid.pkl')#os.path.join(opt.data_path, 'train.pkl')
	valid_pkl = os.path.join(opt.data_path, 'valid.pkl')
	test_pkl =  os.path.join(opt.data_path, 'valid.pkl')
	cap_pkl = os.path.join(opt.data_path, 'CAP.pkl')
	cate_pkl = os.path.join(opt.data_path, 'category.pkl')
	wtoi_path = os.path.join(opt.data_path, 'worddict.pkl')
	# feature files, npy or hdf5
	feat_path1 = os.path.join(opt.data_path, 'feats.hdf5')
	feat_path2 = os.path.join(opt.data_path2, 'feats.hdf5')
	posseq_path = os.path.join(opt.data_path, 'postagsequence.hdf5')
	file1 = h5py.File(feat_path1, 'r')
	file2 = h5py.File(feat_path2, 'r')
	posfile = h5py.File(posseq_path, 'r')

	mytrain_dset = custom_dset_train(train_pkl, cap_pkl, cate_pkl, file1, file2, posfile, wtoi_path, opt.vocab_size, opt.feat_K, opt)
	myvalid_dset = custom_dset_test(valid_pkl, cap_pkl, cate_pkl, file1, file2, posfile, wtoi_path, opt.vocab_size, opt.feat_K, opt)
	mytest_dset = custom_dset_test(test_pkl, cap_pkl, cate_pkl, file1, file2, posfile, wtoi_path, opt.vocab_size, opt.feat_K, opt)
	return mytrain_dset, myvalid_dset, mytest_dset

def test_dataio(opt):
	test_pkl = os.path.join(opt.data_path, 'test.pkl')
	cap_pkl = os.path.join(opt.data_path, 'CAP.pkl')
	cate_pkl = os.path.join(opt.data_path, 'category.pkl')
	wtoi_path = os.path.join(opt.data_path, 'worddict.pkl')
	# feature files, npy or hdf5
	# file = os.path.join(opt.data_path, 'feats/')
	feat_path1 = os.path.join(opt.data_path, 'feats.hdf5')
	feat_path2 = os.path.join(opt.data_path2, 'feats.hdf5')
	posseq_path = os.path.join(opt.data_path, 'postagsequence.hdf5')
	file1 = h5py.File(feat_path1, 'r')
	file2 = h5py.File(feat_path2, 'r')
	posfile = h5py.File(posseq_path, 'r')
	mytest_dset = custom_dset_test(test_pkl, cap_pkl, cate_pkl, file1, file2, posfile, wtoi_path, opt.vocab_size, opt.feat_K, opt)
	return mytest_dset


if __name__ == '__main__':
	opt = myopts.parse_opt()
	opt.data_path = '../datas/msrvtt_i3d_rgb/'
	opt.data_path2 = '../datas/msrvtt_i3d_flow/'
	opt.data_path_pool = '../datas/msrvtt_i3d_rgb_pool/'
	opt.data_path_pool2 = '../datas/msrvtt_i3d_flow_pool/'
	dset = test_dataio(opt)
	loader = DataLoader(dset, batch_size=2, collate_fn=collate_fn)
	count = 0
	for data, cap, cap_mask, cap_class, class_mask, feat1, feat2, feat3, feat4, feat_mask, pos_feat, lens, gts, image_id in loader:
		print('data:',data)
		print('image_id:',image_id)
		print('cap:', len(cap), cap)
		print('cap_maks:', cap_mask)
		print('cap_class:', len(cap_class), cap_class)
		print('class_mask:', class_mask)
		print('pso_feat: ', pos_feat)
		break
	# 	if count >= 1:
	# 		break
	# 	else:
	# 		count += 1
