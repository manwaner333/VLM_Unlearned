export CUDA_HOME=/home/jovyan/.local/conda/envs/tofu1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


CUDA_VISIBLE_DEVICES=0,1,2,3 DS_SKIP_CUDA_CHECK=1  ./inference.py  --eval_file ./dataset/full.json --loss_type full --model_path models/final_ft_2_epochs_lr1e-05_llava-v1.6-vicuna_full/step_320