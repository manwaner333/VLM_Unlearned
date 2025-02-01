export CUDA_HOME=/home/jovyan/.local/conda/envs/tofu1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3


DS_SKIP_CUDA_CHECK=1 accelerate launch \
    --config_file config/accelerate_config.yaml \
    --multi_gpu \
    --num_processes=4 \
    ./finetune.py \

# deepspeed  --num_gpus=4 finetune.py --config_file config/accelerate_config.yaml