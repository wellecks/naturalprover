ENGINE_BASE=curie:ft-academics-uw-2021-12-18-07-51-00
ENGINE_GT=curie:ft-academics-uw-2022-01-26-00-12-13
MAX_STEPS=25
N=10
NRUNS=3
ALPHA=0.75
TEMP=0.30


# greedy-decoding
python generate_gpt3_stepwise.py --ckpt "$ENGINE_GT" --refs gtrefs --mode full --temperature 0.0 --core-only --n 1 --max-steps "$MAX_STEPS" --num-runs 1

# full proof sampling
python generate_gpt3_stepwise.py --ckpt "$ENGINE_GT" --refs gtrefs --mode full --temperature "$TEMP" --core-only --n "$N" --max-steps "$MAX_STEPS" --num-runs "$NRUNS" --alpha "$ALPHA"

# stepwise beam
python generate_gpt3_stepwise.py --ckpt "$ENGINE_GT" --refs gtrefs --mode stepwise_beam --core-only --expansion multitemp --selection diversity  --temperature "$TEMP" --n "$N" --num-runs "$NRUNS" --max-steps "$MAX_STEPS" --alpha "$ALPHA"

