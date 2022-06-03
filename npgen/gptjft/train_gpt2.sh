TRAIN_FILE=../../data/gptjft_proofwiki_gtrefs_ref-pretrain.train.json
VALID_FILE=../../data/gptjft_proofwiki_gtrefs_ref-pretrain.valid.json
MODEL=gpt2-medium
CONFIG=/path/to/Finetune_GPTNEO_GPTJ6B/finetuning_repo/ds_config.json
OUTDIR=/path/to/gpt2
GPU=0

deepspeed --include localhost:"$GPU" --master_port 61000 /path/to/Finetune_GPTNEO_GPTJ6B/finetuning_repo/run_clm.py \
        --deepspeed "$CONFIG" \
        --model_name_or_path "$MODEL" \
        --train_file "$TRAIN_FILE"\
        --validation_file "$VALID_FILE"\
        --do_train \
        --do_eval \
        --fp16 \
        --overwrite_cache \
        --evaluation_strategy="steps" \
        --output_dir "$OUTDIR" \
        --num_train_epochs 10 \
        --eval_steps 50 \
        --logging_steps 50 \
        --logging_first_step \
        --gradient_accumulation_steps 1 \
        --per_device_train_batch_size 16 \
        --use_fast_tokenizer False \
        --learning_rate 5e-06 \
        --warmup_steps 10 \
        --save_total_limit 2 \
        --save_strategy="steps" \
        --save_steps 50 \
        --load_best_model_at_end 1 \
        --block_size 1024 \
        --logging_dir "$OUTDIR" \
        --report_to="tensorboard"
