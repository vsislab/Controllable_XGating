# train model by cross entropy loss
CUDA_VISIBLE_DEVICES=7 python starttrain.py --data_path ../datas/msrvtt_inpRes_rgb/ --data_path2 ../datas/msrvtt_i3d_flow/ --beam_size 1 --learning_rate_decay_start 3 --scheduled_sampling_start 3 --id msrvtt_inpRes_i3dflow --checkpoint_path ../results/XE/msrvtt_inpRes_i3dflow/ --feat_size 1536 --feat_size2 1024 --seed 1024 --feat_K 20 --weight_class 0.0

# train model by self-critical sequence training
#CUDA_VISIBLE_DEVICES=7 python starttrain.py --data_path ../datas/msrvtt_inpRes_rgb/ --data_path2 ../datas/msrvtt_i3d_flow/ --beam_size 1 --learning_rate_decay_start 3 --scheduled_sampling_start 3 --id msrvtt_inpRes_i3dflow_RL --checkpoint_path ../results/RL/msrvtt_inpRes_i3dflow_RL/ --feat_size 1536 --feat_size2 1024 --seed 1024 --feat_K 20 --weight_class 0.0 --start_from ../results/XE/msrvtt_inpRes_i3dflow/ --self_critical_after 14 --reward_type CIDEr --max_epochs -1
