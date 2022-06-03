import argparse
from tqdm import tqdm
import pickle
import json
import openai
from pathlib import Path
import csv

from npgen.evaluation_proofgen import parse_reference_titles
from collections import Counter
from copy import deepcopy
import transformers
import numpy as np
import os
openai.api_key = os.environ['OPENAI_API_KEY']

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')


# truncate prompt to 1024 tokens
def truncate_prompt(prompt):
    assert prompt[-8:] == ' <proof>'
    prompt = prompt[:-8] # get rid of ' <proof>', which is 3 tokens
    
    input_ids = tokenizer(prompt)['input_ids']
    
    # first, try to get rid of references
    while len(input_ids) > 1021 and prompt.endswith('</reference>'):
        prompt = prompt[:prompt.rfind('<reference>')-1] # -1 because of the trailing space
        input_ids = tokenizer(prompt)['input_ids']
    
    # if still overflows, truncate the theorem content
    if len(input_ids) > 1021:
        input_ids = input_ids[:1014] + input_ids[-7:] # The last 7 tokens are '</content> </theorem>'
        prompt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
        
    prompt += ' <proof>'
    return prompt

# truncate proof history to 900 tokens
def truncate_history(history):
    input_ids = tokenizer(history)['input_ids']
    
    while len(input_ids) > 900:
        history = ' ' + history[history.find('\\n')+2:] # get rid of the oldest history
        input_ids = tokenizer(history)['input_ids']
        
    return history

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


def parse_constraints(prompt, orig, vf_kind):
    if vf_kind == 'set':
        import re
        reg = r"<reference>([^<]*)</reference>"
        constraints = re.findall(reg, prompt, re.MULTILINE)
        constraints = [c.strip() for c in constraints]
    else:
        raise NotImplementedError()
    return constraints


def get_vf(constraints, vf_kind):
    if vf_kind == 'set':
        vf = ConstraintValueFunction(constraints)
    else:
        raise NotImplementedError()
    return vf


def generate_sample_and_rerank_constraints(
        ckpt, item, orig,
        temperature=1.0,
        n=10,
        length_normalize=False,
        print_debug=False,
        alpha=1.0,
        vf_kind='set'
):
    constraints = parse_constraints(item['prompt'], orig, vf_kind)
    if print_debug:
        print("Constraints:\n\t%s" % (constraints))
    vf = get_vf(constraints, vf_kind)
    prompt = truncate_prompt(item['prompt'])
    total_sampled_tokens = 0
    while True:
        try:
            completion = openai.Completion.create(
                model=ckpt,
                prompt=prompt,
                max_tokens=1000,
                temperature=temperature,
                n=n,
                logprobs=1,
                stop='</proof>',
            )
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            import time
            time.sleep(10)

    scored = []
    for choice in completion['choices']:
        y = choice['text'].strip(' ')
        n_tokens = len(choice['logprobs']['token_logprobs'])
        logp = sum(choice['logprobs']['token_logprobs'])
        if length_normalize:
            lm_score = logp / max(1, n_tokens)  # normalize
        else:
            lm_score = logp
        scored.append({
            'vf': vf.score(y),
            'lm': lm_score,
            'logp': logp,
            'n_tokens': n_tokens,
            'yt': y
        })
        total_sampled_tokens += n_tokens

    max_vf = max([x['vf'] for x in scored])
    max_lm = max(np.abs(np.array([x['lm'] for x in scored])))
    for x in scored:
        x['vf_normalized'] = (x['vf'] / max_vf) if max_vf > 0 else x['vf']
        x['lm_normalized'] = (x['lm'] / max_lm) if max_lm > 0 else x['lm']
        x['score'] = alpha * x['vf_normalized'] + (1.0-alpha) * x['lm_normalized']

    best = list(sorted(scored, key=lambda x: -x['score']))[0]
    score, logp, n_tokens, y = best['score'], best['logp'], best['n_tokens'], best['yt']

    vf.advance(y)

    print("=== Unsatisfied (%d/%d)\t Logp %.4E\tAvg Logp %.4E\tTotal sampled tokens %d" % (
        vf.num_unsatisfied(),
        len(vf.constraints),
        logp,
        logp/n_tokens,
        total_sampled_tokens
    ))
    return y, logp, n_tokens, total_sampled_tokens, scored


class Beam(object):
    def __init__(self, vf, prompt):
        self.vf = vf
        self.prompt = prompt
        self.proof_steps = []
        self.done = False
        self.total_logp = 0
        self.total_n_tokens = 0

    def get_history(self):
        if len(self.proof_steps) == 0:
            history = self.prompt
        else:
            history = self.prompt + ' ' + '\\n'.join(self.proof_steps) + '\\n'
        return truncate_history(history)

    def get_proof(self):
        return '\\n'.join(self.proof_steps)

    def score_candidate(self, choice):
        yt = choice['text'].strip(' ')
        n_tokens = len(choice['logprobs']['token_logprobs'])
        logp = sum(choice['logprobs']['token_logprobs'])

        lm_score = np.exp(logp)
        proof = self.get_proof() + '\\n' + yt
        out = {
            'vf': self.vf.score(proof),
            'lm': lm_score,
            'logp': logp,
            'n_tokens': n_tokens,
            'yt': yt
        }
        return out

    def advance(self, scored):
        self.proof_steps.append(scored['yt'])
        self.total_logp += scored['logp']
        self.total_n_tokens += scored['n_tokens']
        if '</proof>' in scored['yt'] or '{{qed}}' in scored['yt']:
            self.done = True


def expand_multitemp(beam_idx, beam_candidate, ckpt, history, multitemp_weights):
    expansions = []
    if multitemp_weights == 'broad':
        params = [(0.0, 1), (0.3, 3), (0.5, 3), (0.7, 3)]
    elif multitemp_weights == 'narrow':
        params = [(0.0, 1), (0.3, 4), (0.5, 4)]
    else:
        raise NotImplementedError()
    n_timeouts = 10
    for temp, n in params:
        while True:
            try:
                completion = openai.Completion.create(
                    model=ckpt,
                    prompt=history,
                    max_tokens=120,
                    temperature=temp,
                    n=n,
                    logprobs=1,
                    stop=['\\n', '</proof>'],
                )
                break
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")
                import time
                time.sleep(10)
                n_timeouts -= 1
                if n_timeouts == 0:
                    tqdm.write("Too many timeouts, skipping...")
                    break

        for choice in completion['choices']:
            expansions.append((beam_idx, beam_candidate.score_candidate(choice)))
    return expansions


def expand_singletemp(beam_idx, beam_candidate, ckpt, history, temp, n):
    expansions = []
    while True:
        try:
            completion = openai.Completion.create(
                model=ckpt,
                prompt=history,
                max_tokens=120,
                temperature=temp,
                n=n,
                logprobs=1,
                stop=['\\n', '</proof>'],
            )
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            import time
            time.sleep(10)

    for choice in completion['choices']:
        expansions.append((beam_idx, beam_candidate.score_candidate(choice)))
    return expansions


def select_topk(expansions, alpha, beam_size):
    # normalize globally
    max_vf = max([x['vf'] for _, x in expansions])
    max_lm = max(np.abs(np.array([x['lm'] for _, x in expansions])))
    for _, x in expansions:
        x['vf_normalized'] = (x['vf'] / max_vf) if max_vf > 0 else x['vf']
        x['lm_normalized'] = (x['lm'] / max_lm) if max_lm > 0 else x['lm']
        x['score'] = alpha * x['vf_normalized'] + (1.0 - alpha) * x['lm_normalized']

    # choose top-k
    topk = list(sorted(expansions, key=lambda x: -x[1]['score']))[:beam_size]
    return topk


def select_diversity(expansions, alpha, beam_size, diversity_weights='broad'):
    if diversity_weights == 'broad':
        params = [0.1, 0.5, 1.0]
    elif diversity_weights == 'narrow':
        params = [0.5, 0.75, 0.9]
    # normalize globally
    max_vf = max([x['vf'] for _, x in expansions])
    max_lm = max(np.abs(np.array([x['lm'] for _, x in expansions])))
    for _, x in expansions:
        x['vf_normalized'] = (x['vf'] / max_vf) if max_vf > 0 else x['vf']
        x['lm_normalized'] = (x['lm'] / max_lm) if max_lm > 0 else x['lm']
        x['score1'] = params[0] * x['vf_normalized'] + (1.0-params[0]) * x['lm_normalized']
        x['score2'] = params[1] * x['vf_normalized'] + (1.0-params[1]) * x['lm_normalized']
        x['score3'] = params[2] * x['vf_normalized'] + (1.0-params[2]) * x['lm_normalized']
        x['score'] = alpha * x['vf_normalized'] + (1.0-alpha) * x['lm_normalized']

    nc = beam_size // 3
    # choose top-k
    topk = (
        list(sorted(expansions, key=lambda x: -x[1]['score1']))[:nc] +
        list(sorted(expansions, key=lambda x: -x[1]['score2']))[:nc] +
        list(sorted(expansions, key=lambda x: -x[1]['score3']))[:nc]
    )
    return topk


def generate_stepwise_beam(
        ckpt, item, orig,
        temperature=1.0,
        n=10,
        max_steps=50,
        length_normalize=False,
        print_debug=False,
        alpha=1.0,
        beam_size=6,
        expansion='singletemp',
        selection='topk',
        diversity_weights='broad',
        multitemp_weights='broad',
        vf_kind='set'
):
    constraints = parse_constraints(item['prompt'], orig, vf_kind)
    if print_debug:
        print("Constraints:\n\t%s" % (constraints))
    prompt = truncate_prompt(item['prompt'])

    stop = False
    i = 0
    total_sampled_tokens = 0
    beam = [Beam(get_vf(constraints, vf_kind), prompt) for _ in range(beam_size)]
    completed = []
    while not stop:
        # expand each item in the beam
        expansions = []
        for beam_idx, beam_candidate in enumerate(beam):
            history = beam_candidate.get_history()
            if expansion == 'multitemp':
                expansions_ = expand_multitemp(beam_idx, beam_candidate, ckpt, history, multitemp_weights)
            else:
                expansions_ = expand_singletemp(beam_idx, beam_candidate, ckpt, history, temperature, n)
            expansions.extend(expansions_)

        for _, score_obj in expansions:
            total_sampled_tokens += score_obj['n_tokens']

        if selection == 'diversity':
            topk = select_diversity(expansions, alpha, beam_size, diversity_weights)
        else:
            topk = select_topk(expansions, alpha, beam_size)

        # advance the beam
        beam_ = []
        for beam_idx, score in topk:
            beam_obj = deepcopy(beam[beam_idx])
            beam_obj.advance(score)
            if beam_obj.done:
                completed.append(beam_obj)
            else:
                beam_.append(beam_obj)
        beam = beam_
        if len(completed) >= beam_size:
            stop = True
        if i > max_steps:
            stop = True
        i += 1

    # choose the best completed item
    scored = []
    if len(completed) == 0:
        completed = beam
    for beam_obj in completed:
        scored.append({
            'vf': beam_obj.vf.score(beam_obj.get_proof()),
            'lm': beam_obj.total_logp,
            'beam_obj': beam_obj
        })
    # normalize globally
    max_vf = max([x['vf'] for x in scored])
    max_lm = max([x['lm'] for x in scored])
    for x in scored:
        x['vf_normalized'] = (x['vf'] / max_vf) if max_vf > 0 else x['vf']
        x['lm_normalized'] = (x['lm'] / max_lm) if max_lm > 0 else x['lm']
        x['score'] = alpha*x['vf_normalized'] + (1-alpha)*x['lm_normalized']
    best = list(sorted(scored, key=lambda x: -x['score']))[0]

    proof = best['beam_obj'].get_proof()
    total_logp = best['beam_obj'].total_logp
    total_tokens = best['beam_obj'].total_n_tokens
    best['beam_obj'].vf.advance(proof)
    print(proof)
    print("=== Unsatisfied (%d/%d)\t Logp %.4E\tAvg Logp %.4E\tTotal sampled tokens %d" % (
        best['beam_obj'].vf.num_unsatisfied(),
        len(best['beam_obj'].vf.constraints),
        total_logp,
        total_logp/total_tokens,
        total_sampled_tokens
    ))
    return proof, total_logp, total_tokens, total_sampled_tokens


def load_core(ds, gpt3_ds, split, evalset_path):
    with open(evalset_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        core_evalset = [row for row in reader]
        core_evalset = [row for row in core_evalset if split == 'both' or row['split'] == split]
        pairs = [(ds[_['split']][int(_['ix_in_split'])], gpt3_ds[_['split']][int(_['ix_in_split'])]) for _ in core_evalset]
    return pairs


def load_thmids(theorem_ids_filepath, ds, gpt3_ds):
    theorem_ids = list(map(lambda x: int(x.split('\t')[1]), open(theorem_ids_filepath).readlines()))
    pairs = []
    for (orig, item) in list(zip(ds['valid'] + ds['test'], gpt3_ds['valid'] + gpt3_ds['test'])):
        if orig['id'][0] in theorem_ids:
            pairs.append((orig, item))
    return pairs


def get_name(args, run, extension='json'):
    if args.core_only:
        thm_tag = 'core'
    elif args.theorem_ids:
        thm_tag = Path(args.theorem_ids).stem
    else:
        thm_tag = 'all'
    vf_tag = '%s-vf' % args.vf_kind
    decoding_tag = '%s__temp=%.2f_n=%d_alpha=%.2f' % (args.mode, args.temperature, args.n, args.alpha)
    if args.selection != 'topk':
        decoding_tag += '__diversity'
        decoding_tag += ('-%s' % args.diversity_weights)
    if args.expansion != 'singletemp':
        decoding_tag += '__multitemp'
        decoding_tag += ('-%s' % args.multitemp_weights)
    name = '%s%s_%s_%s__%s__%s__full_generations__run%d.%s' % (
        args.name,
        args.ckpt,
        args.refs,
        decoding_tag,
        vf_tag,
        thm_tag,
        run,
        extension
    )
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='gpt3')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='../../data')
    parser.add_argument('--outdir', type=str, default='../../generated')
    parser.add_argument('--evalset-path', type=str, default='../../data/base/core_evalset.tsv')
    parser.add_argument('--split', choices=['valid', 'test', 'both'], default='valid')
    parser.add_argument('--refs', choices=['norefs', 'gtrefs', 'retrefs'], required=True)
    parser.add_argument('--mode', choices=['full', 'stepwise_beam'], required=True)
    parser.add_argument('--core-only', action='store_true')

    parser.add_argument('--selection', choices=['topk', 'diversity'], default='topk')
    parser.add_argument('--expansion', choices=['singletemp', 'multitemp'], default='singletemp')
    parser.add_argument('--diversity-weights', choices=['narrow', 'broad'], default='broad')
    parser.add_argument('--multitemp-weights', choices=['narrow', 'broad'], default='broad')
    parser.add_argument('--vf-kind', choices=['set', 'set-orig', 'order'], default='set')

    parser.add_argument('--theorem-ids', type=str, default=None)
    parser.add_argument('--length-normalize', action='store_true')
    parser.add_argument('--print-debug', action='store_true')
    parser.add_argument('--max-steps', type=int, default=25)
    parser.add_argument('--num-runs', type=int, default=1)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--save-all', action='store_true')

    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam-size', type=int, default=6)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--top-p', type=float, default=1.0)

    args = parser.parse_args()

    with open(os.path.join(args.datadir, 'base', 'proofwiki__refs_ground_truth.json')) as f:
        ds = json.load(f)

    gpt3_ds = {}
    for split in ['valid', 'test']:
        suffix = '' if args.refs == 'norefs' else '_ref-pretrain'
        with open(os.path.join(args.datadir, 'gpt3', f'gpt3ft_proofwiki_{args.refs}{suffix}.{split}.jsonl')) as f:
            gpt3_ds[split] = [json.loads(line.strip('\n')) for line in f]

    if args.core_only:
        pairs = load_core(ds, gpt3_ds, args.split, args.evalset_path)
    elif args.theorem_ids:
        pairs = load_thmids(args.theorem_ids, ds, gpt3_ds)
    else:
        pairs = list(zip(ds['valid'] + ds['test'], gpt3_ds['valid'] + gpt3_ds['test'])) if args.split == 'both' else list(zip(ds[args.split], gpt3_ds[args.split]))

    print("=== Generating with %d pairs" % len(pairs))

    for run in tqdm(range(args.num_runs)):
        full_generations = []
        print("=== run %d/%d" % (run+1, args.num_runs))
        for (orig, item) in tqdm(pairs):
            if args.mode == 'full':
                proof, logp, n_tokens, total_sampled_tokens, scored = generate_sample_and_rerank_constraints(
                    args.ckpt, item, orig,
                    temperature=args.temperature,
                    n=args.n,
                    length_normalize=args.length_normalize,
                    print_debug=args.print_debug,
                    alpha=args.alpha,
                    vf_kind=args.vf_kind
                )
            elif args.mode == 'stepwise_beam':
                proof, logp, n_tokens, total_sampled_tokens = generate_stepwise_beam(
                    args.ckpt, item, orig,
                    temperature=args.temperature,
                    n=args.n,
                    length_normalize=args.length_normalize,
                    max_steps=args.max_steps,
                    print_debug=args.print_debug,
                    alpha=args.alpha,
                    beam_size=args.beam_size,
                    expansion=args.expansion,
                    selection=args.selection,
                    diversity_weights=args.diversity_weights,
                    multitemp_weights=args.multitemp_weights,
                    vf_kind=args.vf_kind
                )
            generation = {
                'metadata': orig['id'],
                'logp': logp,
                'n_tokens': n_tokens,
                'text': proof,
                'orig': orig,
                'total_sampled_tokens': total_sampled_tokens,
                'all_cands': scored if args.save_all else []
            }
            full_generations.append(generation)

        os.makedirs(args.outdir, exist_ok=True)
        outjson = os.path.join(
            args.outdir, get_name(args, run=run+1, extension='json')
        )
        json.dump({
            'full_generations': full_generations,
            'name': args.name,
            'ckpt': args.ckpt
        }, open(outjson, 'w'), indent=4)
        print("=== Output file: %s" % outjson)


if __name__ == '__main__':
    main()
