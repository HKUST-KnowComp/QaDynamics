# QADYNAMICS: Training Dynamics-Driven Synthetic QA Diagnostic for Zero-Shot Commonsense Question Answering

This is the official code and data repository for the paper published in Findings of EMNLP2023: QADYNAMICS: Training Dynamics-Driven Synthetic QA Diagnostic for Zero-Shot Commonsense Question Answering

## 1. Download Data & Model Checkpoint

We will upload training dynamics, data, and best model checkpoint soon.

## 2. Required Package

Required packages are listed in `requirements.txt`. Install them by running:

```bash
pip install -r requirements.txt
```

## 3. Model Training

Use the following command to train the model at the directory of `src/Training/`.
Turn on the `training_dynamics` argument to decide whether to record the training dynamics.
And use `td_every` to decide the frequency to obtain the training dynamics.

```commandline
CUDA_VISIBLE_DEVICES=1 python run_pretrain.py \
    --model_type deberta \
    --model_name_or_path microsoft/deberta-v3-large \
    --task_name cskg \
    --output_dir ../../output \
    --train_file ../../atomic/train.json \
    --second_train_file ../../cwwv/train.json \
    --dev_file ../../data/ATOMIC/dev_random.jsonl \
    --second_dev_file ../../data/CWWV/dev_random.jsonl \
    --max_seq_length 128 \
    --max_words_to_mask 6 \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --warmup_proportion 0.05 \
    --evaluate_during_training \
    --per_gpu_eval_batch_size 2  \
    --save_steps 2000\
    --margin 1.0 \
    --seed 0 \
    --training_dynamics \
    --td_every 500
```

## 4. Acknowledgement

The authors of this paper were supported by the NSFC Fund (U20B2053) from the NSFC of China, the RIF (R6020-19 and R6021-20), and the GRF (16211520 and 16205322) from RGC of Hong Kong. 
We also thank the support from the UGC Research Matching Grants (RMGS20EG01-D, RMGS20CR11, RMGS20CR12, RMGS20EG19, RMGS20EG21, RMGS23CR05, RMGS23EG08). 