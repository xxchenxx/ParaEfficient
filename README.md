# Parameter Efficient Fine-tuning

## Environment
```bash
conda create -n para -f environment.yml
```
## BitFit

```bash
cd BitFit
python run_glue.py 
       --output-path <output_path>\
       --task-name rte\
       --model-name bert-base-cased\
       --fine-tune-type bitfit\
       --learning-rate 1e-3
```

## BitFit

```bash
cd BitFit
python run_glue.py 
       --output-path <output_path>\
       --task-name rte\
       --model-name bert-base-cased\
       --fine-tune-type bitfit\
       --learning-rate 1e-3
```

## P-Tuning v2

```bash
cd P-Tuning-v2
python3 run.py \
  --model_name_or_path bert-large-cased \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
```