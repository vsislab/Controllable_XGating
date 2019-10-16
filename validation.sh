
CUDA_VISIBLE_DEVICES=2 python eval.py --model $1'model-best.pth' --infos_path $2'-best.pkl' > ./logs/$3'.log-best'&
#CUDA_VISIBLE_DEVICES=0 python eval.py --model $1'model-meteor-best.pth' --infos_path $2'-meteor-best.pkl' > ./logs/$3'.log-meteor-best'&
#CUDA_VISIBLE_DEVICES=0 python eval.py --model $1'model-bleu-best.pth' --infos_path $2'-bleu-best.pkl' > ./logs/$3'.log-bleu-best'&
#CUDA_VISIBLE_DEVICES=1 python eval.py --model $1'model-rouge-best.pth' --infos_path $2'-rouge-best.pkl' > ./logs/$3'.log-rouge-best'&
#CUDA_VISIBLE_DEVICES=1 python eval.py --model $1'model.pth' --infos_path $2'.pkl' > ./logs/$3'.log-current'&

wait


