"""Requires https://github.com/wellecks/naturalproofs"""
import pytorch_lightning as pl
import argparse
import os
import pickle
from pathlib import Path
import naturalproofs.encoder_decoder.model as mutils
import naturalproofs.encoder_decoder.utils as utils
from naturalproofs.encoder_decoder.analyze import analyze, analyze_rankings
import numpy as np
import torch
import transformers
from collections import defaultdict
from tqdm import tqdm


def predict(dl, model, rid2tok):
    model.eval()
    model.cuda()
    model.metrics_valid.reset()
    id2ranked_rids = {}
    tok2rid = {v: k for k, v in rid2tok.items()}
    with torch.no_grad():
        out = defaultdict(list)
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            x, y, metadata = batch
            x = x.cuda()
            y = y.cuda()

            model.validation_step((x, y), i)
            for j, (xj, yj) in enumerate(zip(x, y)):
                xj = xj.unsqueeze(0)
                ranked = utils.extract_rankings(
                    model, xj,
                    yhatj=torch.tensor([[model.hparams.bos]], dtype=torch.long, device=xj.device),
                    use_first=True,
                    use_generations=False
                )
                ranked_rids = [tok2rid[tok] for tok in ranked]
                id2ranked_rids[metadata[j]['eid']] = ranked_rids
                yj_ = utils.trim(model, yj.cpu().view(-1)).tolist()

                out['y'].append(yj_)
                out['ranked'].append(ranked)

    ms_tok = model.metrics_valid.report()
    return id2ranked_rids, out, ms_tok


def _name(args):
    name = 'eval'
    name += '__%s' % (args.dataset)
    name += '.pkl'
    return name


def cli_main():
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='proofwiki')
    parser.add_argument(
        '--checkpoint-path',
        default='/checkpoint/best.ckpt'
    )
    parser.add_argument(
        '--output-dir',
        default='/output'
    )
    parser.add_argument(
        '--datapath',
        default='/data/sequence_tokenized__bert-base-cased.pkl'
    )
    parser.add_argument('--token-limit', type=int, default=8192)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--dataloader-workers', type=int, default=0)
    parser.add_argument('--model-type', default='bert-base-cased')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    from naturalproofs.encoder_decoder.model_joint import ParallelSequenceRetriever
    model = ParallelSequenceRetriever.load_from_checkpoint(args.checkpoint_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)

    print("Loading data (%s)" % args.datapath)
    ds_raw = pickle.load(open(args.datapath, 'rb'))
    rid2tok = ds_raw['rid2tok']
    dls = utils.get_dataloaders(
        ds_raw['tokenized'],
        xpad=tokenizer.pad_token_id,
        ypad=rid2tok['<pad>'],
        token_limit=args.token_limit,
        buffer_size=args.buffer_size,
        workers=args.dataloader_workers,
        set_mode=1,
        order='ground-truth',
        include_metadata=True
    )

    split2output = {}
    for split in ['train', 'valid', 'test']:
        dl = dls[split]
        print("%d examples" % (len(dl.dataset.data)))

        id2ranked_rids, out, token_metrics = predict(dl=dl, model=model, rid2tok=rid2tok)
        rmetrics = analyze_rankings(out['y'], out['ranked'])
        print("\n== ranking metrics")
        for k, metric in rmetrics.items():
            print('\t%.5f\t%s' % (metric, k))

        output_file_contents = {
            'id2ranked_rids': id2ranked_rids,
            'token_metrics': token_metrics,
            'ranking-metrics': rmetrics,
            'args': vars(args),
        }
        split2output[split] = output_file_contents

    outfile = os.path.join(args.output_dir, _name(args))
    pickle.dump(split2output, open(outfile, 'wb'))

    print("\nWrote to %s" % outfile)
    print("== done.")


if __name__ == '__main__':
    cli_main()
