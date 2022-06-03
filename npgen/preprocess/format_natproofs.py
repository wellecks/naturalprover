from tqdm import tqdm
import json
import os
import pickle


def format_ground_truth(dataset, split):
    id2ref = {item['id']: item for item in dataset['theorems'] + dataset['definitions'] + dataset['others']}
    examples = split['examples']
    formatted = []
    line_sep = '\\n'
    for eid, (tid, pix) in tqdm(enumerate(examples), total=len(examples)):
        item = {}
        ex = id2ref[tid]
        proof = ex['proofs'][pix]

        item['id'] = (tid, pix)
        item['title'] = ex['title']
        item['text'] = line_sep.join(ex['contents'])
        item['target'] = line_sep.join(proof['contents'])
        item['ctxs'] = []

        for ref_id in sorted(set(proof['ref_ids'])):
            ref = id2ref[ref_id]
            ctx = {
                'title': ref['title'],
                'text': line_sep.join(ref['contents'])
            }
            item['ctxs'].append(ctx)

        formatted.append(item)
    return formatted


def _load_retrievals(retrieved_filepath, split_name):
    id2ranked_rids = pickle.load(open(retrieved_filepath, 'rb'))[split_name]['id2ranked_rids']
    return id2ranked_rids


def format_retrieved(dataset, split, split_name, retrieved_filepath, num_retrieved=100):
    id2ranked_rids = _load_retrievals(retrieved_filepath, split_name)
    id2ref = {item['id']: item for item in dataset['theorems'] + dataset['definitions'] + dataset['others']}
    examples = split['examples']
    formatted = []
    line_sep = '\\n'
    for eid, (tid, pix) in tqdm(enumerate(examples), total=len(examples)):
        item = {}
        ex = id2ref[tid]
        proof = ex['proofs'][pix]

        item['id'] = (tid, pix)
        item['title'] = ex['title']
        item['text'] = line_sep.join(ex['contents'])
        item['target'] = line_sep.join(proof['contents'])
        item['ctxs'] = []

        for ref_id in id2ranked_rids[eid][:num_retrieved]:
            ref = id2ref[ref_id]
            ctx = {
                'title': ref['title'],
                'text': line_sep.join(ref['contents']),
                'source': 'retrieved'
            }
            item['ctxs'].append(ctx)

        formatted.append(item)
    return formatted


def _filename(args):
    suffix = '' if args.suffix == '' else '__%s' % args.suffix
    outfile = os.path.join(args.output_dir, '%s__refs_%s%s.json' % (
        args.dataset_name,
        args.refs,
        suffix
    ))
    return outfile


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='')
    parser.add_argument(
        '--filepath',
        default='naturalproofs_proofwiki.json'
    )
    parser.add_argument(
        '--retrieved-filepath',
        default='eval__proofwiki.pkl'
    )
    parser.add_argument('--output-dir', default='/net/nfs2.corp/mosaic/home/seanw/datasets/naturalproofs_gen/raug')
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--refs', default='ground_truth', choices=['ground_truth', 'retrieved'])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    base = json.load(open(args.filepath, 'r'))
    ds = base['dataset']
    refs = ds['theorems'] + ds['definitions'] + ds['others']

    out = {}
    if args.refs == 'ground_truth':
        for split_name in ['train', 'valid', 'test']:
            split = base['splits'][split_name]
            out[split_name] = format_ground_truth(
                ds, split
            )
    if args.refs == 'retrieved':
        for split_name in ['train', 'valid', 'test']:
            split = base['splits'][split_name]
            out[split_name] = format_retrieved(
                ds, split, split_name, args.retrieved_filepath
            )

    outfile = _filename(args)
    print("Writing to %s" % outfile)
    with open(outfile, 'w') as f:
        json.dump(out, f)

    print("=== done.")
