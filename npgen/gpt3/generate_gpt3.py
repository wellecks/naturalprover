import argparse
from tqdm import tqdm
import json
import csv
import openai
import os
openai.api_key = os.environ['OPENAI_API_KEY']

import transformers
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

def generate_full_proof(ckpt, item):
    prompt = truncate_prompt(item['prompt'])
    
    while True:
        try:
            completion = openai.Completion.create(
                model=ckpt,
                prompt=prompt,
                max_tokens=1020,
                temperature=0.0,
                stop='</proof>',
            )
            break
        except openai.error.RateLimitError as e:
            tqdm.write(str(e))
            tqdm.write("Retrying in 10 min ...")
            import time
            time.sleep(600)
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying in 10 min...")
            import time
            time.sleep(600)
            
    proof = completion['choices'][0]['text']
    proof = proof.strip(' ')

    return proof

def generate_next_steps(ckpt, item, orig):
    prompt = truncate_prompt(item['prompt'])
    
    gold_steps = split_to_steps(orig['target'])
    
    proof_lines = []
    
    for i, gold_step in enumerate(gold_steps):
        history = '' if i == 0 else (' ' + '\\n'.join(gold_steps[:i]) + '\\n')
        history = truncate_history(history)
        
        while True:
            try:
                # greedy
                completion = openai.Completion.create(
                    model=ckpt,
                    prompt=prompt + history,
                    max_tokens=120,
                    temperature=0.0,
                    stop=['\\n', '</proof>'],
                )
                greedy = completion['choices'][0]['text'].strip(' ')
                '''
                # beam search
                completion = openai.Completion.create(
                    model=ckpt,
                    prompt=prompt + history,
                    max_tokens=120,
                    best_of=20,
                    stop=['\\n', '</proof>'],
                )
                beam = completion['choices'][0]['text'].strip(' ')
                '''
                # sampling
                completion = openai.Completion.create(
                    model=ckpt,
                    prompt=prompt + history,
                    max_tokens=120,
                    n=10,
                    stop=['\\n', '</proof>'],
                )
                samples = [completion['choices'][i]['text'].strip(' ') for i in range(10)]
                completion = openai.Completion.create(
                    model=ckpt,
                    prompt=prompt + history,
                    max_tokens=120,
                    n=10,
                    top_p=0.9,
                    stop=['\\n', '</proof>'],
                )
                samples_p9 = [completion['choices'][i]['text'].strip(' ') for i in range(10)]
                completion = openai.Completion.create(
                    model=ckpt,
                    prompt=prompt + history,
                    max_tokens=120,
                    n=10,
                    top_p=0.7,
                    stop=['\\n', '</proof>'],
                )
                samples_p7 = [completion['choices'][i]['text'].strip(' ') for i in range(10)]
                completion = openai.Completion.create(
                    model=ckpt,
                    prompt=prompt + history,
                    max_tokens=120,
                    n=10,
                    top_p=0.5,
                    stop=['\\n', '</proof>'],
                )
                samples_p5 = [completion['choices'][i]['text'].strip(' ') for i in range(10)]
                completion = openai.Completion.create(
                    model=ckpt,
                    prompt=prompt + history,
                    max_tokens=120,
                    n=10,
                    temperature=0.8,
                    stop=['\\n', '</proof>'],
                )
                samples_t8 = [completion['choices'][i]['text'].strip(' ') for i in range(10)]
                completion = openai.Completion.create(
                    model=ckpt,
                    prompt=prompt + history,
                    max_tokens=120,
                    n=10,
                    temperature=0.6,
                    stop=['\\n', '</proof>'],
                )
                samples_t6 = [completion['choices'][i]['text'].strip(' ') for i in range(10)]

                break
            except openai.error.RateLimitError as e:
                tqdm.write(str(e))
                tqdm.write("Retrying in 10 min ...")
                import time
                time.sleep(600)
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying in 10 min...")
                import time
                time.sleep(600)

        line_output = {
            'greedy': greedy,
            'true': gold_step,
            'samples': samples,
            'samples_p9': samples_p9,
            'samples_p7': samples_p7,
            'samples_p5': samples_p5,
            'samples_t8': samples_t8,
            'samples_t6': samples_t6,
        }
        proof_lines.append(line_output)

    output = { 'proof_lines': proof_lines }
    return output


def load_core(ds, gpt3_ds, split, evalset_path):
    with open(evalset_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        core_evalset = [row for row in reader]
        core_evalset = [row for row in core_evalset if split == 'both' or row['split'] == split]
        pairs = [(ds[_['split']][int(_['ix_in_split'])], gpt3_ds[_['split']][int(_['ix_in_split'])]) for _ in core_evalset]
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='gpt3')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='../../data')
    parser.add_argument('--outdir', type=str, default='../../generated')
    parser.add_argument('--evalset-path', type=str, default='../../data/base/core_evalset.tsv')
    parser.add_argument('--split', choices=['valid', 'test', 'both'], default='valid')
    parser.add_argument('--refs', choices=['norefs', 'gtrefs', 'retrefs'], required=True)
    parser.add_argument('--mode', choices=['fullgen', 'nextstep'], required=True)
    parser.add_argument('--core-only', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    args.output_path = os.path.join(
        args.outdir,
        f'generated_gpt3_{args.refs}_{args.mode}{".core" if args.core_only else ""}.{args.split}.json'
    )

    with open(os.path.join(args.datadir, 'base', 'proofwiki__refs_ground_truth.json')) as f:
        ds = json.load(f)

    gpt3_ds = {}
    for split in ['valid', 'test']:
        suffix = '' if args.refs == 'norefs' else '_ref-pretrain'
        with open(os.path.join(args.datadir, 'gpt3', f'gpt3ft_proofwiki_{args.refs}{suffix}.{split}.jsonl')) as f:
            gpt3_ds[split] = [json.loads(line.strip('\n')) for line in f]

    if args.core_only:
        pairs = load_core(ds, gpt3_ds, args.split, args.evalset_path)
    else:
        pairs = list(zip(ds['valid'] + ds['test'], gpt3_ds['valid'] + gpt3_ds['test'])) if args.split == 'both' else list(zip(ds[args.split], gpt3_ds[args.split]))

    if args.mode == 'fullgen':

        full_generations = []
        for (orig, item) in tqdm(pairs):

            proof = generate_full_proof(args.ckpt, item)
    
            generation = {
                'metadata': orig['id'],
                'text': proof,
                'orig': orig,
            }
            full_generations.append(generation)
    
        json.dump({
            'full_generations': full_generations,
            'name': args.name,
            'ckpt': args.ckpt
        }, open(args.output_path, 'w'), indent=4)

    elif args.mode == 'nextstep':

        nextstep_generations = []
        for (orig, item) in tqdm(pairs):

            output = generate_next_steps(args.ckpt, item, orig)
            nextstep_generations.append({'output': output, 'orig': orig})
    
        json.dump({
            'next_step_generations': nextstep_generations,
            'name': args.name,
            'ckpt': args.ckpt
        }, open(args.output_path, 'w'), indent=4)


if __name__ == '__main__':
    main()

