# inference corss entropy model
CUDA_VISIBLE_DEVICES=7 python eval.py --model ../results/msrvtt_Resrgbfc_i3dflowfc_NO14mask_globalpos_dgate/model-best.pth --infos_path ../results/msrvtt_Resrgbfc_i3dflowfc_NO14mask_globalpos_dgate/infos_msrvtt_Resrgbfc_i3dflowfc_14mask_globalpos-best.pkl --beam_size 3

# validate SCST model
#CUDA_VISIBLE_DEVICES=7 python eval.py --model ../results/msrvtt_Resrgbfc_i3dflowfc_NO14mask_globalpos_dgate_RL/model-best.pth --infos_path ../results/msrvtt_Resrgbfc_i3dflowfc_NO14mask_globalpos_dgate_RL/infos_msrvtt_Resrgbfc_i3dflowfc_14mask_globalpos-best.pkl --beam_size 3
