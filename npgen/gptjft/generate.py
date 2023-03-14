import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from npgen.evaluation_proofgen import parse_reference_titles
from collections import Counter
from copy import deepcopy
import numpy as np
import npgen.gptjft.utils as utils
import csv


def truncate_prompt(tokenizer, prompt, max_length):
    assert prompt[-19:] == ' <|extratoken_100|>'
    prompt = prompt[:-19] # get rid of ' <proof>', which is 3 tokens
    
    input_ids = tokenizer(prompt)['input_ids']
    
    # first, try to get rid of references
    while len(input_ids) > max_length and prompt.endswith('</reference>'):
        prompt = prompt[:prompt.rfind('<reference>')-1] # -1 because of the trailing space
        input_ids = tokenizer(prompt)['input_ids']
    
    # if still overflows, truncate the theorem content
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length] + input_ids[-7:] # The last 7 tokens are '</content> </theorem>'
        prompt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
        
    prompt += ' <|extratoken_100|>'
    return prompt


def split_to_steps(proof):
    splits = proof.split('\\n')
    steps = []
    step = ''
    for split in splits:
        if len(split) == 0:
            continue
        if split[0].islower():
            step += '\\n' + split
        else:
            if len(step) > 0:
                steps.append(step)
            step = split
    if len(step) > 0:
        steps.append(step)
    return steps


class ConstraintValueFunction(object):
    def __init__(self, constraints):
        self.unsatisfied = Counter(constraints)
        self.constraints = constraints

    def score(self, state):
        score = 0
        titles = Counter(parse_reference_titles(state))
        unsatisfied = deepcopy(self.unsatisfied)
        for title in unsatisfied:
            if unsatisfied[title] > 0 and title in titles and titles[title] > 0:
                score += 1
                titles[title] -= 1
                unsatisfied[title] -= 1
        return score

    def advance(self, state):
        titles = Counter(parse_reference_titles(state))
        for title in titles:
            self.unsatisfied[title] = self.unsatisfied[title] - titles[title]

    def num_unsatisfied(self):
        return sum([v for k, v in self.unsatisfied.items() if v > 0])


def parse_constraints(prompt):
    import re
    reg = r"<reference>([^<]*)</reference>"
    constraints = re.findall(reg, prompt, re.MULTILINE)
    constraints = [c.strip() for c in constraints]
    return constraints


def generate_sample_and_rerank_constraints(
        model, tokenizer, batch,
        temperature=1.0,
        n=10,
        length_normalize=False,
        print_debug=False,
        alpha=1.0,
        beta=1.0,
        max_length=1000
):
    origs = [b[0] for b in batch]
    items = [b[1] for b in batch]
    prompts = [truncate_prompt(tokenizer, item['prompt'], max_length-100) for item in items]
    completions = utils.generate(
        model=model,
        tokenizer=tokenizer,
        texts=prompts,
        decoder_params=dict(
            max_length=max_length,
            temperature=temperature,
            do_sample=temperature > 0,
            num_return_sequences=n,
        )
    )

    scored = []
    outputs = []
    for i in range(len(completions)):
        total_sampled_tokens = 0
        item = items[i]
        completion = completions[i]
        orig = origs[i]
        constraints = parse_constraints(item['prompt'])
        if print_debug:
            print("Constraints:\n\t%s" % (constraints))
        vf = ConstraintValueFunction(constraints)
        for choice in completion['choices']:
            y = choice['text'].strip(' ')
            n_tokens = choice['n_tokens']
            logp = choice['logp']
            if length_normalize:
                lm_score = np.exp(logp / max(1, n_tokens))  # normalize
            else:
                lm_score = np.exp(logp)
            scored.append({
                'vf': vf.score(y),
                'lm': lm_score,
                'logp': logp,
                'n_tokens': n_tokens,
                'yt': y
            })
            total_sampled_tokens += n_tokens

        max_vf = max([x['vf'] for x in scored])
        for x in scored:
            x['vf_normalized'] = (x['vf'] / max_vf) if max_vf > 0 else x['vf']
            x['score'] = alpha * x['vf_normalized'] + beta * x['lm']

        best = list(sorted(scored, key=lambda x: -x['score']))[0]
        score, logp, n_tokens, y = best['score'], best['logp'], best['n_tokens'], best['yt']

        vf.advance(y)
        y = y.replace('<|extratoken_101|>', '')  # </proof> symbol
        print(y.replace('\\n', '\n'))

        print("=== Unsatisfied (%d/%d)\t Logp %.4E\tAvg Logp %.4E\tTotal sampled tokens %d" % (
            vf.num_unsatisfied(),
            len(vf.constraints),
            logp,
            logp/max(n_tokens, 1),
            total_sampled_tokens
        ))
        outputs.append({
            'metadata': orig['id'],
            'y': y,
            'logp': logp,
            'n_tokens': n_tokens,
            'total_sampled_tokens': total_sampled_tokens
        })
    return outputs


def load_core(ds, gpt3_ds, split, evalset_path):
    with open(evalset_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        core_evalset = [row for row in reader]
        core_evalset = [row for row in core_evalset if split == 'both' or row['split'] == split]
        pairs = [(ds[_['split']][int(_['ix_in_split'])], gpt3_ds[_['split']][int(_['ix_in_split'])]) for _ in core_evalset]
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--mode', choices=['full'], default='full')
    parser.add_argument('--core-only', action='store_true')

    parser.add_argument('--evalset-path', type=str, default='../../data/base/core_evalset.tsv')
    parser.add_argument('--datadir', type=str, default='../../data')
    parser.add_argument('--outdir', type=str, default='../../generated')

    parser.add_argument('--version', type=str, default='latest')
    parser.add_argument('--name', type=str, default='gptj')
    parser.add_argument('--refs', choices=['norefs', 'gtrefs', 'retrefs'], default='gtrefs')
    parser.add_argument('--length-normalize', action='store_true')
    parser.add_argument('--print-debug', action='store_true')
    parser.add_argument('--max-steps', type=int, default=25)
    parser.add_argument('--max-length', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--num-runs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--split', choices=['valid', 'train', 'test', 'both'], default='valid')

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--top-p', type=float, default=1.0)

    args = parser.parse_args()

    model, tokenizer = utils.load_model(args.ckpt)
    model.eval()

    with open(os.path.join(args.datadir, 'base', 'proofwiki__refs_ground_truth.json')) as f:
        ds = json.load(f)

    gpt3_ds = {}
    for split in ['train', 'valid', 'test']:
        suffix = '' if args.refs == 'norefs' else '_ref-pretrain'
        with open(os.path.join(args.datadir, 'gptj', f'gptjft_proofwiki_{args.refs}{suffix}.{split}.json')) as f:
            gpt3_ds[split] = [json.loads(line.strip('\n')) for line in f]

    if args.core_only:
        pairs = load_core(ds, gpt3_ds, args.split, args.evalset_path)
        thm_tag = 'core'
    else:
        pairs = list(
            zip(ds['valid'] + ds['test'], gpt3_ds['valid'] + gpt3_ds['test'])) if args.split == 'both' else list(
            zip(ds[args.split], gpt3_ds[args.split]))
        thm_tag = 'all'

    print("=== Generating with %d pairs" % len(pairs))
    decoding_tag = '%s__temp=%.2f_n=%d_alpha=%.2f' % (args.mode, args.temperature, args.n, args.alpha)
    os.makedirs(args.outdir, exist_ok=True)

    for run in tqdm(range(args.num_runs)):
        full_generations = []
        print("=== run %d/%d" % (run+1, args.num_runs))
        n_batches = (len(pairs) // args.batch_size)
        for i in tqdm(range(n_batches)):
            batch = pairs[i*args.batch_size:(i+1)*args.batch_size]
            if batch == []:
                continue
            if args.mode == 'full':
                outputs = generate_sample_and_rerank_constraints(
                    model, tokenizer, batch,
                    temperature=args.temperature,
                    n=args.n,
                    length_normalize=args.length_normalize,
                    print_debug=args.print_debug,
                    alpha=args.alpha,
                    max_length=args.max_length
                )
            full_generations.extend(outputs)

        run_ = run+1
        outfile = f'{args.outdir}/{args.name}_{args.refs}__{decoding_tag}__{thm_tag}__full_generations__run{run_}.json'
        json.dump({
            'full_generations': full_generations,
            'name': args.name,
            'ckpt': args.ckpt
        }, open(outfile, 'w'))
        print("=== Output file: %s" % outfile)

if __name__ == '__main__':
    main()

