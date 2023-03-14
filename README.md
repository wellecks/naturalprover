## NaturalProver: Grounded Mathematical Proof Generation with Language Models

[NaturalProver: Grounded Mathematical Proof Generation with Language Models](https://arxiv.org/pdf/2205.12910.pdf)\
Sean Welleck\*, Jiacheng Liu\*, Ximing Lu, Hannaneh Hajishirzi, Yejin Choi



This repo contains:

- The **NaturalProofs-Gen** datasets.
- **GPT-3, GPT-J, GPT-2** code for training and generation.
- **Automatic evaluation** for proof generation and next-step suggestion.
- **GPT-2** trained model.

Please cite our work if you found the resources in this repository useful:
```
@article{welleck2022naturalprover,
    title={NaturalProver: Grounded Mathematical Proof Generation with Language Models},
    author={Sean Welleck and Jiacheng Liu and Ximing Lu and Hannaneh Hajishirzi and Yejin Choi},
    year={2022},
    eprint={2205.12910},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

**Note**: this repo has been updated following publication. For the version at publication time, see the `neurips2022` branch.

## Quick download
To download and unpack the data, models, and other files discussed below:
```bash
pip install gdown
python download.py --data --model --other --savedir /path/to/savedir
```
This creates the following file structure:
```bash
data/   # NaturalProofs-Gen datasets
model/  # Pretrained NaturalProver models 
other/  # Additional files (e.g. example predictions)
```

## Data

#### Quick download with `gdown`
To download and unpack the `NaturalProofs-Gen` datasets:
```bash
pip install gdown
python download.py --data --savedir ./
```
This creates the following file structure:
```
data/base  # Original proofwiki, NaturalProofs-Gen (unsplit), redirects (used in evaluation).
data/gpt3  # NaturalProofs-Gen formatted for GPT-3 fine-tuning + ref-reconstruction for retrieved & provided settings.
data/gptj  # NaturalProofs-Gen formatted for GPT-2/J fine-tuning (similar to data/gpt3).
```

Within each folder, you will see datasets with varied reference-conditioning (`norefs`, `retrefs`, `gtrefs`) and reference reconstruction (`ref-pretrain`) settings.

#### Download with `git-lfs`
Alternatively, the `NaturalProofs-Gen` datasets are also stored in the `data/` directory using [git-lfs](https://git-lfs.github.com/):

```bash
git-lfs pull
```
This will create the same file structure discussed above.

## GPT3

### Finetuning
See `npgen/gpt3/finetune_gpt3.sh`.

### Generation
For **full-proof generation**,
```bash
cd npgen/gpt3
python generate_gpt3_stepwise.py  # see generate_gpt3_stepwise.sh for example arguments
```
See `npgen/gpt3/generate_gpt3_stepwise.sh` for example commands for greedy decoding, full-proof sample-and-select, and stepwise++ decoding. 
Also see `npgen/gpt3/generate_gpt3.sh` for greedy decoding commands using various models.

For **next-step** generation:
```bash
cd npgen/gpt3
python generate_gpt3.py --ckpt ${CKPT} --refs {no,gt,ret}refs --mode nextstep --core-only
```
See `npgen/gpt3/generate_gpt3.sh` for example commands.

### Evaluate generations
For an example of running automatic metrics for full-proof generation, first download the naturalprover generations:
```
pip install gdown
python download.py --other --savedir /path/to/savedir

==> /path/to/savedir/other/naturalprover_generations
```
Then see this notebook for an example of running the metrics:
```bash
notebooks/evaluation.ipynb
```
The notebook reproduces the GPT-3 automatic metrics in the paper (Table 7).

| name                   |   gleu |    f1 |   kf1 |   ref_precision |   ref_recall |   ref_f1 |   corpus_ref_halluc |
|:-----------------------|-------:|------:|------:|----------------:|-------------:|---------:|--------------------:|
| gpt3                   |  24.4  | 49.96 | 49.3  |           29.93 |        24.73 |    23.69 |               17.92 |
| naturalprover-retrieve |  26.58 | 53.02 | 55.88 |           38.17 |        28.48 |    27.1  |                2.25 |
| naturalprover          |  35.27 | 66    | 90.07 |           93.05 |        86.05 |    87.08 |                1.6  |
| naturalprover++        |  34.49 | 65.61 | 96.39 |           94.66 |        95    |    93.92 |                1.71 |

## GPT2/J

### Finetuning

See `npgen/gptj/train_gpt{j,2}.sh` for example commands. The script uses Deepspeed, and 
is based on [mallorbc/Finetune_GPTNEO_GPTJ6B](https://github.com/mallorbc/Finetune_GPTNEO_GPTJ6B/tree/main/finetuning_repo). 

### GPT2 finetuned model

We provide a GPT-2 model fine-tuned with provided in-context references and reference reconstruction.
```
pip install gdown
python download.py --model --savedir /path/to/savedir

==> /path/to/savedir/model/naturalprover_gpt2
```

### Generation

See commands in:
```
cd npgen/gptjft
bash eval_gpt2.sh
```
For GPT-J, see `npgen/gptjft/eval_gptj.sh`.


### Evaluate generations
For an example of running automatic metrics for full-proof generation, first download the naturalprover generations:
```
pip install gdown
python download.py --other --savedir /path/to/savedir

==> /path/to/savedir/other/naturalprover_generations
```
Then see this notebook for an example of running the metrics:
```bash
notebooks/evaluation.ipynb
```
We provide GPT-2 and GPT-3-curie generations (provided in-context references, greedy decoding) in `/other/naturalprover_generations`, and GPT-J generations with greedy decoding and sample-and-select with the constraint value function (10 samples).

Results on the core validation set:

| name             |   gleu |   ref_f1 |   corpus_ref_halluc |
|:-----------------|-------:|---------:|--------------------:|
| naturalprover-gpt2-greedy      |  32.06 |    65.22 |                6.76 |
| naturalprover-gptj6b-greedy    |  38.58 |    79.19 |                2.96 |
| naturalprover-gptj6b-select10  |  37.83 |    88.80  |                4.84 |
| naturalprover-gpt3curie-greedy |  42.39 |    89.29 |                1.9  |

*The repo does not have stepwise++ or next-step suggestions for GPT-2/J*.







