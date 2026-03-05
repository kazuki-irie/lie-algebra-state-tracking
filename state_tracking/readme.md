## Instruction

Create the following folder here:
```
mkdir .project-root
```

To install the main packages:

```
pip install -r requirements.txt
```

See also the instruction in the parent directory to install other packages.

NB: we used `gcc/12.2.0` and `cuda/12.4.1`

## Data Generation

The data files need to be generated separately for each task using the following command:

```
python src/generate_data.py --group=S3 --k=128 --samples=500000
```

* where `group` can be replaced by Z2 (C2), Z60 (C60), D8, H3, S4, S5, A4, A5, etc.
* `k` specifies the sequence length. E.g., we need to generate a training set using `k=128` and a test set with `k=256` 

## Word Problems: Training & Evaluation

An example training script is provided in `sub_example_job.sh`.

```
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
```

* where `model` can be chosen from `transformer`, `gla`, `mamba_lstm`, `neg_mamba_lstm`, `aussm` and `deltaproduct`.
* There are some DeltaProduct specific flags---we can just keep/ignore them for other models.
* `task` and `n_layers`, as well as training/eval sequence lengths to be varied.
* `wandb_user` is the wandb user name.

To loop over some of the variables, remove the corresponding variable definition from the script and do something like:
```
for SEED in 1 2 3; do sbatch --export=ALL,seed=$SEED sub_example_job.sh;done
```

## Rotation Tasks: Training & Evaluation

The main file for the rotation task is `src/main_regression.py`.

For the `A5json` argument, provide the path to the file `src/A5.json`.
Other arguments are similar to the word problem setting above.

```
A5json=

python src/main_regression.py train \
  --task=a5_regression \
  --a5_json_path=${A5json} \
  --group=${task} \
  --k=${trainLen} \
  --k_test=${evalLen} \
  --n_layers=${numLayers} \
  --epochs=50 \
  --allow_neg_eigval=True \
  --num_householder=${householder} \
  --batch_size=${batch} \
  --seed=${seed} \
  --lr=${lr} \
  --hidden_size=${hsize} \
  --mamba_expand_rate=4 \
  --n_heads=8 \
  --use_scheduler=True \
  --model_name=${model} \
  --wandb_user=${user} \
  --project_name=${project}
  ```
