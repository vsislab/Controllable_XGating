#CUDA_VISIBLE_DEVICES=7 python starttrain.py --data_path ../datas/msrvtt_inpRes_rgb/ --data_path2 ../datas/msrvtt_i3d_flow/ --data_path_pool ../datas/msrvtt_inpRes_rgb_pool/ --data_path_pool2 ../datas/msrvtt_i3d_flow_pool/ --beam_size 1 --learning_rate_decay_start 3 --scheduled_sampling_start 3 --id debug --checkpoint_path ../results/debug/debug/ --feat_size 1536 --feat_size2 1024 --feat_depth 1536 --feat_depth2 1024 --HxW 64 --HxW2 49 --seed 1024 --feat_K 20 --weight_class 0.0

CUDA_VISIBLE_DEVICES=7 python starttrain.py --data_path ../datas/msrvtt_inpRes_rgb/ --data_path2 ../datas/msrvtt_i3d_flow/ --data_path_pool ../datas/msrvtt_inpRes_rgb_pool/ --data_path_pool2 ../datas/msrvtt_i3d_flow_pool/ --beam_size 1 --learning_rate_decay_start 3 --scheduled_sampling_start 3 --id debug --checkpoint_path ../results/debug/debug_RL/ --feat_size 1536 --feat_size2 1024 --feat_depth 1536 --feat_depth2 1024 --HxW 64 --HxW2 49 --seed 1024 --feat_K 20 --weight_class 0.0 --start_from ../results/debug/debug/ --self_critical_after 0 --reward_type CIDEr --max_epochs -1
