CUDA_VISIBLE_DEVICES=7 python starttrain_trainpos.py --data_path ../datas/msrvtt_inpRes_rgb/ --data_path2 ../datas/msrvtt_i3d_flow/ --beam_size 1 --learning_rate_decay_start 3 --scheduled_sampling_start -1 --id postag --checkpoint_path ../results/pos_generator/pos_tag/ --feat_size 1536 --feat_size2 1024 --seed 1024 --feat_K 20 --weight_class 0.0 --language_eval 0 --save_checkpoint_every 50