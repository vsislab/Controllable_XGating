import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--data_path', type=str, default='../datas/oldi3drgb/',
                        help='the path contains feats/,CAP.pkl and so on. ')
    parser.add_argument('--data_path2', type=str, default='../datas/oldi3dflow/',
                        help='the path contains another feats/,CAP.pkl and so on. ')
    parser.add_argument('--start_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'infos.pkl'         : configuration;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--feat_K', type=int, default=30, help=' the number of feats(that is, frames) take out from a video')

    # Model settings
    parser.add_argument('--vocab_size', type=int, default=20000, help='number of all words')
    parser.add_argument('--category_size', type=int, default=20, help='number of all categories')
    parser.add_argument('--rnn_size', type=int, default=512, help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--seq_length', type=int, default=28, help='the length of the sentence coming from dataset')
    parser.add_argument('--input_encoding_size', type=int, default=468, help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_size', type=int, default=1536, help='the size of the attention MLP; recommend same as feat_size')
    parser.add_argument('--feat_size', type=int, default=1536, help='2048 for resnet, 4096 for vgg, 1536 for Inception-V4')
    parser.add_argument('--feat_size2', type=int, default=1536, help='2048 for resnet, 4096 for vgg, 1536 for Inception-V4')
    parser.add_argument('--fusion_activity', type=str, default='ReLU', help='ReLU, Tanh, Sigmoid for unlinear function in fusion model')
    parser.add_argument('--weight_class', type=float, default=0.0)

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1, help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--reward_type', type=str, default='CIDEr', help='use BLEU/METEOR/ROUGE/CIDEr/MIX as reward')
    parser.add_argument('--beam_size', type=int, default=3, help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam|adadelta')
    parser.add_argument('--learning_rate', type=float, default=4e-4, # 4e-4
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=4,  help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9, help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=-1, help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=1000, help=' to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save', help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=50, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=0, help='Do we load previous best score when resuming training.')

    # misc
    parser.add_argument('--id', type=str, default='', help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--seed', type=int, default=0, help='the random seed.')
    parser.add_argument('--patience', type=int, default='30', help='the early stop threshold(number of validation times')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 , "language_eval in pos generator should be 0"

    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.optim in ['adam','adadelta','rmsprop','sgd','sgdmom','adagrad'], "the optimizer type should be clear"
    assert args.reward_type in ['BLEU','METEOR','ROUGE','CIDEr','MIX'], "check the reward type, which should be 'BLEU','METEOR','ROUGE','CIDEr' or 'MIX'"

    return args
