DATA_DIR=../../data/gpt3

# Finetune GPT-3 with no reference grounding
# The model finetuned with this command: curie:ft-academics-uw-2021-12-18-07-51-00
openai api fine_tunes.create -m curie \
    -t ${DATA_DIR}/gpt3ft_proofwiki_norefs.train.jsonl \
    -v ${DATA_DIR}/gpt3ft_proofwiki_norefs.valid.jsonl

# Finetune GPT-3 grounded in retrieved references
# The model finetuned with this command: curie:ft-academics-uw-2022-01-26-00-12-13
openai api fine_tunes.create -m curie \
    -t ${DATA_DIR}/gpt3ft_proofwiki_retrefs_ref-pretrain.train.jsonl \
    -v ${DATA_DIR}/gpt3ft_proofwiki_retrefs_ref-pretrain.valid.jsonl

# Finetune GPT-3 grounded in human-provided references
# The model finetuned with this command: curie:ft-academics-uw-2022-01-30-14-23-47
openai api fine_tunes.create -m curie \
    -t ${DATA_DIR}/gpt3ft_proofwiki_gtrefs_ref-pretrain.train.jsonl \
    -v ${DATA_DIR}/gpt3ft_proofwiki_gtrefs_ref-pretrain.valid.jsonl

