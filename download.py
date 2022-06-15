import gdown
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', default='./')
parser.add_argument('--data', action='store_true')
parser.add_argument('--model', action='store_true')
parser.add_argument('--other', action='store_true')

args = parser.parse_args()
os.makedirs(args.savedir, exist_ok=True)

if args.data:
    url = 'https://drive.google.com/uc?id=1CoTXOUgWjdVFEreHAT8YkEwoQuGDeL2g'
    os.makedirs(os.path.join(args.savedir, 'data'), exist_ok=True)
    out = os.path.join(args.savedir, 'data', 'naturalprover_data.tar.gz')
    gdown.download(url, out, quiet=False)
    gdown.extractall(out, os.path.join(args.savedir, './'))

if args.model:
    url = 'https://drive.google.com/uc?id=1LCsBiVQ_mjrfzPjOxCkWhPtogoSM1dNu'
    os.makedirs(os.path.join(args.savedir, 'model'), exist_ok=True)
    out = os.path.join(args.savedir, 'model', 'naturalprover_gpt2.tar.gz')
    gdown.download(url, out, quiet=False)
    gdown.extractall(out, os.path.join(args.savedir, './'))

if args.other:
    url = 'https://drive.google.com/uc?id=1uZFc2BD1SZTA8JkX7qXIi9w5sI17acA5'
    os.makedirs(os.path.join(args.savedir, 'other'), exist_ok=True)
    out = os.path.join(args.savedir, 'other', 'naturalprover_generations.tar.gz')
    gdown.download(url, out, quiet=False)
    gdown.extractall(out, os.path.join(args.savedir, './'))



