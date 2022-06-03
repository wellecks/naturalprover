GPU=0
CKPT=/path/to/gptj
OUTDIR=/path/to/output
MAX_STEPS=20
NRUNS=1
ALPHA=0.75

CUDA_VISIBLE_DEVICES="$GPU" python generate.py --name gptj --ckpt "$CKPT" --split valid --mode full --refs gtrefs --temperature 0.0 --n 1 --num-runs "$NRUNS" --alpha "$ALPHA" --outdir "$OUTDIR" --core-only

