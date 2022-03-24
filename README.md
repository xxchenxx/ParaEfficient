# Parameter Efficient Fine-tuning

## Environment
```bash
conda env create -n para -f environment.yml
conda activate para
pip install tensorboardx
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install loralib
```
## BitFit

```bash
cd BitFit
python run_glue.py \
       --output-path output \
       --task-name rte\
       --model-name bert-base-cased\
       --fine-tune-type bitfit\
       --learning-rate 1e-3
```

## P-Tuning v2

```bash
cd P-Tuning-v2
python3 run.py \
  --model_name_or_path bert-base-cased \
  --task_name glue \
  --dataset_name rte \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-2 \
  --num_train_epochs 60 \
  --pre_seq_len 20 \
  --output_dir checkpoints/$DATASET_NAME-bert/ \
  --overwrite_output_dir \
  --hidden_dropout_prob 0.1 \
  --seed 1 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
```


## DiffPruning

```bash
cd DiffPruning
```

### Data

```bash
python download_glue.py
```
### Stage I: Finding Masks
```bash
SEED=0
PER_GPU_TRAIN_BATCH_SIZE=8
GRADIENT_ACC=1
LR=0.00001000
SPARSITY_PEN=0.00000012500
CONCRETE_LOWER=-1.500
CONCRETE_UPPER=1.500
ALPHA_INIT=5
FIX_LAYER=-1


EXP_NAME=cola
BASE_DIR=$(pwd)
LOCAL_DATA_DIR=${BASE_DIR}/glue_data
LOCAL_CKPT_DIR=${BASE_DIR}/logs/${EXP_NAME}
GPU=1
TASK=cola
DATA=CoLA
mkdir -p ${LOCAL_CKPT_DIR}
cd ${BASE_DIR}

CUDA_VISIBLE_DEVICES=${GPU} python ${BASE_DIR}/examples/run_glue_diffpruning.py --model_type bert --model_name_or_path bert-large-cased-whole-word-masking --task_name ${TASK} --output_dir ${LOCAL_CKPT_DIR} --do_train --do_eval --data_dir ${LOCAL_DATA_DIR}/${DATA} --sparsity_pen ${SPARSITY_PEN} --concrete_lower ${CONCRETE_LOWER} --concrete_upper ${CONCRETE_UPPER} --num_train_epochs 3 --save_steps 5000 --seed ${SEED} --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} --learning_rate ${LR}  --gradient_accumulation_steps ${GRADIENT_ACC} --fix_layer ${FIX_LAYER} --max_seq_length 128 --per_gpu_eval_batch_size 8 --overwrite_output_dir --logging_steps 5000 1>${LOCAL_CKPT_DIR}/${EXP_NAME}.out 2>${LOCAL_CKPT_DIR}/${EXP_NAME}.err
```

### Stage II: Pruning

```{bash}
EXP2=${EXP_NAME}_2nd_mag
mkdir -p ${LOCAL_CKPT_DIR}/${EXP2}
EXP3=${EXP_NAME}_3rd_fixmask
EVAL_CHECKPOINT=${BASE_DIR}/logs/${EXP_NAME}/checkpoint-last-info.pt
mkdir -p ${LOCAL_CKPT_DIR}/${EXP3}

CUDA_VISIBLE_DEVICES=${GPU} python ${BASE_DIR}/examples/run_glue_mag.py --model_type bert --model_name_or_path bert-large-cased-whole-word-masking --task_name ${TASK} --output_dir ${LOCAL_CKPT_DIR}/${EXP2} --do_train --do_eval --data_dir ${LOCAL_DATA_DIR}/${DATA} --sparsity_pen 0.000000125 --concrete_lower -1.5 --concrete_upper 1.5 --num_train_epochs 3 --save_steps 5000 --seed ${SEED} --eval_checkpoint ${EVAL_CHECKPOINT} --save_checkpoint ${LOCAL_CKPT_DIR}/mag0.5p.pt --evaluate_during_training --logging_steps 5000 --target_sparsity 0.005 --overwrite_output_dir 1>${LOCAL_CKPT_DIR}/${EXP2}_mag0.5p.out 2>${LOCAL_CKPT_DIR}/${EXP2}_mag0.5p.err

CUDA_VISIBLE_DEVICES=${GPU} python ${BASE_DIR}/examples/run_glue_fixmask_finetune.py --model_type bert --model_name_or_path bert-large-cased-whole-word-masking --task_name ${TASK} --output_dir ${LOCAL_CKPT_DIR}/${EXP3} --do_train --do_eval --data_dir ${LOCAL_DATA_DIR}/${DATA} --sparsity_pen ${SPARSITY_PEN} --concrete_lower -1.5 --concrete_upper 1.5 --num_train_epochs 3 --save_steps 5000 --seed ${SEED} --mask_checkpoint ${LOCAL_CKPT_DIR}/mag0.5p.pt --evaluate_during_training --logging_steps 5000 --overwrite_output_dir --finetune 1 1>${LOCAL_CKPT_DIR}/${EXP3}_mag0.5.out 2>${LOCAL_CKPT_DIR}/${EXP3}_mag0.5.err
```


## LoRA

```{bash}
cd LoRA/examples/NLU
pip install -e .
export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./rte"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-large \
--lora_path ./roberta_large_lora_mnli.bin \
--task_name rte \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 4e-4 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_lora \
--lora_r 8 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.1

```

## P-tuning-v2

```{r}
cd P-tuning-v2
bash run_script/run_rte_bert.sh 
```