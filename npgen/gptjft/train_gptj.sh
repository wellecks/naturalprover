TRAIN_FILE=/path/to/gptjft_proofwiki_gtrefs_ref-pretrain.train.json
VALID_FILE=/path/to/gptjft_proofwiki_gtrefs_ref-pretrain.valid.json
MODEL=EleutherAI/gpt-j-6B
CONFIG=/path/to/Finetune_GPTNEO_GPTJ6B/finetuning_repo/ds_config_gptj6b.json
OUTDIR=/path/to/gptj

deepspeed --include localhost:0,1,2,3 /path/to/projects/Finetune_GPTNEO_GPTJ6B/finetuning_repo/run_clm.py \
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
        --per_device_train_batch_size 8 \
        --use_fast_tokenizer False \
        --learning_rate 5e-06 \
        --warmup_steps 10 \
        --save_total_limit 1 \
        --save_strategy="steps" \
        --save_steps 50 \
        --load_best_model_at_end 1 \
        --block_size 1024 \
        --logging_dir "$OUTDIR" \
        --report_to="tensorboard"
