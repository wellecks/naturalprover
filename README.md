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

## Data
The `NaturalProofs-Gen` datasets are stored in the `data/` directory using [git-lfs](https://git-lfs.github.com/).

```bash
git-lfs pull
```

```
data/base  # Original proofwiki, NaturalProofs-Gen (unsplit), redirects (used in evaluation).
data/gpt3  # NaturalProofs-Gen formatted for GPT-3 fine-tuning + ref-reconstruction for retrieved & provided settings.
data/gptj  # NaturalProofs-Gen formatted for GPT-2/J fine-tuning (similar to data/gpt3).
```

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

We use the [mallorbc/Finetune_GPTNEO_GPTJ6B](https://github.com/mallorbc/Finetune_GPTNEO_GPTJ6B/tree/main/finetuning_repo) repo.
Setup that repo by following its README, then see `npgen/gptj/train_gpt{j,2}.sh` for example commands.

### GPT2 finetuned model

We provide a GPT-2 model fine-tuned with provided in-context references and reference reconstruction.
```
pip install gdown
python download.py --gpt2 --savedir /path/to/savedir

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
We provide GPT-2 and GPT-J6B generations (provided in-context references, greedy decoding) in `/other/naturalprover_generations`.

| name             |   gleu |   ref_f1 |   corpus_ref_halluc |
|:-----------------|-------:|---------:|--------------------:|
| naturalprover-gpt2-greedy      |  32.06 |    65.22 |                6.76 |
| naturalprover-gptj6b-greedy    |  39.14 |    79.23 |                3.51 |
| naturalprover-gpt3curie-greedy |  42.39 |    89.29 |                1.9  |

*Due to a GPT-J6B-specific discrepancy, GPT-J6B achieves better results than in the inital manuscript*

*The repo does not have stepwise++ or next-step suggestions for GPT-2/J*.







