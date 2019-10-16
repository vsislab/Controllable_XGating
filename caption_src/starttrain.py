import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import cPickle
import time
import os
import data_io
from data_io import *
from SAModel import *
import myopts
import myutils
import eval_utils


def train(opt):
	# load train/valid/test data
	opt.vocab_size = get_nwords(opt.data_path)
	opt.category_size = get_nclasses(opt.data_path)
	mytrain_dset, myvalid_dset, mytest_dset = loaddset(opt)

	writer = SummaryWriter(opt.checkpoint_path)
	# init or load training infos
	infos = {}
	histories = {}
	if opt.start_from is not None:
		# open old infos and check if models are compatible
		with open(os.path.join(opt.start_from, 'infos_' + opt.id + '-best.pkl')) as f:
			infos = cPickle.load(f)
			saved_model_opt = infos['opt']
			need_be_same = ["rnn_size", "num_layers"]  # optim needn't same
			for checkme in need_be_same:
				assert vars(saved_model_opt)[checkme] == vars(opt)[
					checkme], "Command line argument and saved model disagree on '%s' " % checkme

		if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '-best.pkl')):
			with open(os.path.join(opt.start_from, 'histories_' + opt.id + '-best.pkl')) as f:
				histories = cPickle.load(f)
		# random seed must be inherited if didn't assign it.
		if opt.seed == 0:
			opt.seed = infos['opt'].seed

	iteration = infos.get('iter', 0) + 1
	epoch = infos.get('epoch', 0)

	val_result_history = histories.get('val_result_history', {})
	loss_history = histories.get('loss_history', {})
	lr_history = histories.get('lr_history', {})
	ss_prob_history = histories.get('ss_prob_history', {})

	if opt.load_best_score == 1:
		best_val_score = infos.get('best_val_score', None)
	else:
		best_val_score = None

	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)
	model = SAModel(opt)

	if opt.start_from is not None:
		# check if all necessary files exist
		assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
		model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best.pth')), strict=True)
	model.cuda()
	model.train()

	crit = LanguageModelCriterion()
	classify_crit = ClassiferCriterion()
	rl_crit = RewardCriterion()

	# select optimizer
	if opt.optim == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
	elif opt.optim == 'adadelta':
		optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=opt.weight_decay)
		opt.learning_rate_decay_start = -1

	# training start
	tmp_patience = 0
	# each epoch
	while True:
		update_lr_flag = True  # when a new epoch start, set update_lr_flag to True
		if update_lr_flag:
				# Assign the learning rate
			if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 and opt.optim != 'adadelta':
				frac = int( (epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every )
				decay_factor = opt.learning_rate_decay_rate  ** frac
				opt.current_lr = opt.learning_rate * decay_factor
				myutils.set_lr(optimizer, opt.current_lr) # set the decayed rate
				#print('epoch {}, lr_decay_start {}, cur_lr {}'.format(epoch, opt.learning_rate_decay_start, opt.current_lr))
			else:
				opt.current_lr = opt.learning_rate
			# Assign the scheduled sampling prob
			if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
				frac = int( (epoch - opt.scheduled_sampling_start) / opt.scheduled_sampling_increase_every )
				opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
				model.ss_prob = opt.ss_prob
			# If start self critical training
			if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
				sc_flag = True
				myutils.init_cider_scorer(opt.reward_type)
			else:
				sc_flag = False
			update_lr_flag = False

		#loading train data
		myloader_train = DataLoader(mytrain_dset, batch_size=opt.batch_size, collate_fn=data_io.collate_fn, shuffle=True)
		torch.cuda.synchronize()
		for data, cap, cap_mask, cap_classes, class_mask, feat1, feat2, feat_mask, pos_feat, lens, groundtruth, image_id in myloader_train:
			start = time.time()
			cap = Variable(cap,requires_grad=False).cuda()
			cap_mask = Variable(cap_mask,requires_grad=False).cuda()
			cap_classes = Variable(cap_classes, requires_grad=False).cuda()
			class_mask = Variable(class_mask, requires_grad=False).cuda()
			feat1 = Variable(feat1, requires_grad=False).cuda()
			feat2 = Variable(feat2, requires_grad=False).cuda()
			feat_mask = Variable(feat_mask,requires_grad = False).cuda()
			pos_feat = Variable(pos_feat, requires_grad=False).cuda()

			optimizer.zero_grad()
			if not sc_flag:
				out, category = model(feat1, feat2, feat_mask,pos_feat,cap,cap_mask)  # (m,seq_len+1,n_words),(m, seq_len+1, n_classes)
				loss_language = crit(out, cap, cap_mask)
				loss_classify = classify_crit(category, cap_classes, cap_mask, class_mask)
				# print(loss_language.data[0], loss_classify.data[0])
				loss = loss_language + opt.weight_class * loss_classify
			else:
				gen_result,sample_logprobs = model.sample(feat1, feat2, feat_mask, pos_feat, {'sample_max':0})
				reward = myutils.get_self_critical_reward(model,feat1, feat2, feat_mask, pos_feat, groundtruth,gen_result) # (m,max_length)
				loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(reward).float().cuda(), requires_grad=False))
			loss.backward()
			
			myutils.clip_gradient(optimizer, opt.grad_clip)
			optimizer.step()
			train_loss = loss.data[0]
			torch.cuda.synchronize()
			end = time.time()

			if not sc_flag:
				print("iter {} (epoch {}), train_loss = {:.3f}, loss_lang = {:.3f}, loss_class = {:.3f}, time/batch = {:.3f}".format(iteration, epoch, train_loss, loss_language.data[0], loss_classify.data[0], end - start))
			else:
				print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}".format(iteration, epoch, np.mean(reward[:, 0]), end - start))

			# Write the training loss summary
			if (iteration % opt.losses_log_every == 0):
				writer.add_scalar('train_loss', train_loss, iteration)
                                writer.add_scalar('learning_rate', opt.current_lr, iteration)
                                writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)
                                if sc_flag:
                                        writer.add_scalar('avg_reward', np.mean(reward[:, 0]), iteration)

				loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:, 0])
				lr_history[iteration] = opt.current_lr
				ss_prob_history[iteration] = model.ss_prob

			# make evaluation on validation set, and save model
			if (iteration % opt.save_checkpoint_every == 0):
				# eval model
				print('validation and save the model...')
				time.sleep(3)
				eval_kwargs = {}
				eval_kwargs.update(vars(opt)) # attend vars(opt) into eval_kwargs
				val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, classify_crit, myvalid_dset, eval_kwargs)
				print('validation is finish!')
				time.sleep(3)

				writer.add_scalar('validation loss', val_loss, iteration)
                                if opt.language_eval == 1:
                                        for tag, value in lang_stats.items():
                                                if type(value) is list:
                                                        writer.add_scalar(tag, value[-1], iteration)
                                                else:
                                                        writer.add_scalar(tag, value, iteration)
                                        for tag, value in model.named_parameters():
						try:
                                                	tag = tag.replace('.', '/')
                                               		writer.add_histogram(tag, value.data.cpu().numpy(), iteration)
                                                	writer.add_histogram(tag + '/grad', (value.grad).data.cpu().numpy(), iteration)
						except AttributeError:
							continue

				val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

				# Save model if is improving on validation result
				if opt.language_eval == 1:
					current_score = lang_stats['CIDEr']
				else:
					current_score = - val_loss
				best_flag = False

				if best_val_score is None or current_score > best_val_score:
					best_val_score = current_score
					best_flag = True
					tmp_patience = 0
				else:
					tmp_patience += 1

				if not os.path.exists(opt.checkpoint_path):
					os.mkdir(opt.checkpoint_path)
				checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
				torch.save(model.state_dict(), checkpoint_path)
				print("model saved to {}".format(checkpoint_path))

				# Dump miscalleous informations(current information)
				infos['iter'] = iteration
				infos['epoch'] = epoch
				infos['best_val_score'] = best_val_score
				infos['opt'] = opt
				infos['val_score'] = lang_stats
				infos['val_sents'] = predictions

				histories['val_result_history'] = val_result_history
				histories['loss_history'] = loss_history
				histories['lr_history'] = lr_history
				histories['ss_prob_history'] = ss_prob_history
				with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as f:
					cPickle.dump(infos, f)
				with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'wb') as f:
					cPickle.dump(histories, f)

				if best_flag:
					checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
					torch.save(model.state_dict(), checkpoint_path)
					print("model saved to {}".format(checkpoint_path))
					with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl'), 'wb') as f:
						cPickle.dump(infos, f)
					with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '-best.pkl'), 'wb') as f:
						cPickle.dump(histories, f)

			if tmp_patience >= opt.patience:
				break
			iteration += 1
		if tmp_patience >= opt.patience:
			print("early stop, trianing is finished!")
			break
		if epoch >= opt.max_epochs and opt.max_epochs != -1:
			print("reach max epochs, training is finished!")
			break
		epoch += 1


if __name__ == '__main__':
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	opt = myopts.parse_opt()
	print('start training')
	train(opt=opt)
