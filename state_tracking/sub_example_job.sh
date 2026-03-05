#!/bin/bash -l
#SBATCH --partition=
#SBATCH --account=
#SBATCH --time=2:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o logs/%j.log

export CUDA_VISIBLE_DEVICES=0

conda activate wp

export PYTHONPATH=$PWD

seed=1
task="S3"
model="gla"

numLayers=1
batch=2048
householder=2

trainLen=128
evalLen=256

user="irie"
project="word_problem"

python src/main.py train \
  --group=${task} \
  --k=${trainLen} \
  --k_test=${evalLen} \
  --n_layers=${numLayers} \
  --epochs=100 \
  --allow_neg_eigval=True \
  --num_householder=${householder} \
  --batch_size=${batch} \
  --seed=${seed} \
  --lr=1e-3 \
  --n_heads=8 \
  --use_scheduler=True \
  --model_name=${model} \
  --wandb_user=${user} \
  --project_name=${project}


