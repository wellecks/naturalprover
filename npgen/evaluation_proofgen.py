import torch
import os
import re
import editdistance
import numpy as np
from collections import defaultdict
from tqdm import tqdm, trange
from datasets import load_metric
import mwparserfromhell

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _format_ref(original_ref, style, mask_token, start_delimeter, end_delimeter):
    if style == 'mask':
        return ' ' + mask_token
    match = re.match(r'(\[\[(.+?)\|(.+?)\]\])', original_ref)
    if match is None:
        match = re.match(r'(\[\[(.+?)\]\])', original_ref)
        text = match.group(2)
        return '%s%s%s' % (start_delimeter, text, end_delimeter)
    else:
        title = match.group(2)
        surface_form = match.group(3)

    if style == 'title_surface':
        return '%s%s|%s%s' % (start_delimeter, title, surface_form, end_delimeter)

    if style == 'title':
        return '%s%s%s' % (start_delimeter, title, end_delimeter)

    if style == 'surface':
        return '%s%s%s' % (start_delimeter, surface_form, end_delimeter)


def format_refs(contents, ref_style, tokenizer=None, start_delimeter=None, end_delimeter=None):
    ref_info = []
    newstring = ''
    start = 0

    for m in re.finditer(r'(\[\[.+?\]\])', contents):
        end, newstart = m.span()
        newstring += contents[start:end]
        original = contents[end:newstart]
        if original.startswith('[[Category:'):  # category not considered a reference
            continue

        if tokenizer is None:
            mask_token = ''
        elif 't5' in tokenizer.name_or_path:
            global t5_mask_id
            mask_token = tokenizer.special_tokens_map_extended['additional_special_tokens'][t5_mask_id]
            t5_mask_id += 1
        else:
            mask_token = tokenizer.mask_token
        rep = _format_ref(original, ref_style, mask_token, start_delimeter, end_delimeter)
        info = {
            'original_start': end,
            'original_end': newstart,
            'original_ref': contents[end:newstart],
            'start': end,
            'end': end + len(rep),
            'ref': rep
        }
        ref_info.append(info)

        newstring += rep
        start = newstart
    newstring += contents[start:]
    return newstring, ref_info


def parse_reference_titles(text, tokenizer=None):
    _, ref_info = format_refs(text, 'title', tokenizer, '', '')
    ref_titles = [r['ref'] for r in ref_info]
    return ref_titles


class FullGenerationMetrics(object):
    def __init__(self, name, tokenizer, ref_names, redirects, title2ref):
        self.name = name
        self.tokenizer = tokenizer
        self.reset()
        self.meteor = load_metric('meteor')
        self.gleu = load_metric('google_bleu')
        self.ref_names = ref_names
        self.redirects = redirects
        self.redirects_ = {v: k for k, v in redirects.items()}
        self.title2ref = title2ref

    def get_ref(self, title):
        if title in self.title2ref:
            return self.title2ref[title]
        if title in self.redirects:
            return self.title2ref[self.redirects[title]]
        if title in self.redirects_:
            return self.title2ref[self.redirects_[title]]
        return None

    def reset(self):
        self._metrics = defaultdict(list)
        self._corpus_metrics = defaultdict(list)
        self._cache = defaultdict(list)

    def _parse_references(self, text):
        return parse_reference_titles(text, self.tokenizer)

    def _parse_text(self, text):
        # Only keep surface form for references
        text, _ = format_refs(text, 'surface', self.tokenizer, '', '')
        wikicode = mwparserfromhell.parse(text)
        text = ' '.join(map(str, wikicode.filter_text()))
        return text

    def _ref_f1_set(self, preds, actuals):
        if len(preds) == 0 and len(actuals) == 0:
            return 1.0, 1.0, 1.0
        elif len(preds) == 0:
            return 1.0, 0.0, 0.0
        elif len(actuals) == 0:
            return 0.0, 1.0, 0.0

        tp = 0.0
        fp = 0.0
        redirected_actuals = [self.redirects[ref] for ref in actuals if ref in self.redirects]
        seen = set()
        for pred in preds:
            redirected_pred = self.redirects[pred] if pred in self.redirects else pred
            if pred in seen or redirected_pred in seen:
                continue
            if (pred in actuals
                or redirected_pred in actuals
                or pred in redirected_actuals
                or redirected_pred in redirected_actuals
            ):
                tp += 1.0
            else:
                fp += 1.0
            seen.add(pred)
            seen.add(redirected_pred)

        prec = tp / len(preds)
        rec = tp / len(actuals)
        if prec + rec > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        return prec, rec, f1

    def _f1_set(self, preds, actuals):
        if len(preds) == 0 and len(actuals) == 0:
            return 1.0, 1.0, 1.0
        elif len(preds) == 0:
            return 1.0, 0.0, 0.0
        elif len(actuals) == 0:
            return 0.0, 1.0, 0.0

        tp = 0.0
        fp = 0.0
        for pred in preds:
            if (pred in actuals):
                tp += 1.0
            else:
                fp += 1.0

        prec = tp / len(preds)
        rec = tp / len(actuals)
        if prec + rec > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        return prec, rec, f1

    def update(self, pred_text, gt_text, logp=None, ntokens=None):
        pred_text_norm = self._parse_text(pred_text)
        gt_text_norm = self._parse_text(gt_text)
        pred_tokens = self.tokenizer.encode(pred_text_norm)
        gt_tokens = self.tokenizer.encode(gt_text_norm)

        # meteor and gleu
        meteor = self.meteor.compute(
            predictions=[pred_text_norm],
            references=[gt_text_norm],
        )['meteor']
        self._metrics['meteor'].append(meteor)
        gleu = self.gleu.compute(
            predictions=[self.tokenizer.convert_ids_to_tokens(pred_tokens)],
            references=[[self.tokenizer.convert_ids_to_tokens(gt_tokens)]],
        )['google_bleu']
        self._metrics['gleu'].append(gleu)

        # token f1
        prec, rec, f1 = self._f1_set(set(pred_tokens), set(gt_tokens))
        self._metrics['prec'].append(prec)
        self._metrics['rec'].append(rec)
        self._metrics['f1'].append(f1)

        # length
        self._metrics['steps'].append(len(pred_text.split('\\n')))
        self._metrics['len'].append(len(pred_tokens))
        self._metrics['steps_gt'].append(len(gt_text.split('\\n')))
        self._metrics['len_gt'].append(len(gt_tokens))

        # ---- Reference metrics
        pred_refs = self._parse_references(pred_text)
        gt_refs = self._parse_references(gt_text)

        # reference token f1
        pred_ref_tokens = []
        for ref in pred_refs:
            ref_ = self.get_ref(ref)
            if ref_ is not None:
                pred_ref_tokens.extend(
                    self.tokenizer.encode(self._parse_text(' \n '.join(ref_['contents'])))
                )
        gt_ref_tokens = []
        for ref in gt_refs:
            ref_ = self.get_ref(ref)
            if ref_ is not None:
                gt_ref_tokens.extend(
                    self.tokenizer.encode(self._parse_text(' \n '.join(ref_['contents'])))
                )
        kprec, krec, kf1 = self._f1_set(set(pred_ref_tokens), set(gt_ref_tokens))
        self._metrics['kprec'].append(kprec)
        self._metrics['krec'].append(krec)
        self._metrics['kf1'].append(kf1)

        # reference exact match
        em = pred_refs == gt_refs
        self._metrics['ref_em'].append(em)
        em_set = set(pred_refs) == set(gt_refs)
        self._metrics['ref_em_set'].append(em_set)

        # reference set precision/recall/F1
        prec, rec, f1 = self._ref_f1_set(preds=set(pred_refs), actuals=set(gt_refs))
        self._metrics['ref_precision'].append(prec)
        self._metrics['ref_recall'].append(rec)
        self._metrics['ref_f1'].append(f1)

        if logp is not None:
            self._metrics['logp'].append(logp)
        if ntokens is not None:
            self._metrics['ntokens'].append(ntokens)

        # hallucinated references
        halluc_refs_set = set([r for r in pred_refs if r not in self.ref_names])
        pred_refs_set = set(pred_refs)
        self._metrics['ref_halluc'].append(len(halluc_refs_set) / max(1, len(pred_refs_set)))
        # cache for corpus-level metrics
        self._cache['halluc_refs_set'].append(halluc_refs_set)
        self._cache['pred_refs_set'].append(pred_refs_set)
        self._cache['gt_refs'].append(gt_refs)
        self._cache['pred_refs'].append(pred_refs)
        self._cache['match_refs'].append([_ for _ in pred_refs if _ in gt_refs])

    def compute_corpus_metrics(self):
        # Reference hallucinations
        n_halluc = sum([len(s) for s in self._cache['halluc_refs_set']])
        n_refs = sum([len(s) for s in self._cache['pred_refs_set']])
        self._corpus_metrics['ref_halluc'] = float(n_halluc) / max(1.0, float(n_refs))
        self._corpus_metrics['ref_precision'] = sum([len(_) for _ in self._cache['match_refs']]) / sum([len(_) for _ in self._cache['pred_refs']])
        self._corpus_metrics['ref_recall'] = sum([len(_) for _ in self._cache['match_refs']]) / sum([len(_) for _ in self._cache['gt_refs']])
        self._corpus_metrics['ref_f1'] = 2 * self._corpus_metrics['ref_precision'] * self._corpus_metrics['ref_recall'] / (self._corpus_metrics['ref_precision'] + self._corpus_metrics['ref_recall'])

    def report(self):
        out = {}
        for k in self._metrics:
            if 'steps' in k or 'logp' in k:
                out[k] = np.mean(self._metrics[k])
            if 'logp' in k:
                out['avg_seq_logp'] = np.mean(self._metrics[k])
                out['ppl'] = 2**(-1*np.sum(self._metrics[k])/np.sum(self._metrics['ntokens']))
            elif 'len' in k:
                out[k] = np.round(np.mean(self._metrics[k]), 1)
            else:
                out[k] = np.mean(self._metrics[k])*100
        for k in self._corpus_metrics:
            out['corpus_' + k] = self._corpus_metrics[k]*100
        return out


class NextStepMetrics(FullGenerationMetrics):
    def __init__(self, name, tokenizer, ref_names, redirects, title2ref):
        super().__init__(name, tokenizer, ref_names, redirects, title2ref)

    # pick the best candidate along each metric
    def update_multiple_preds(self, pred_texts, gt_text):
        pred_texts_norm = [self._parse_text(pred_text) for pred_text in pred_texts]
        gt_text_norm = self._parse_text(gt_text)
        pred_tokenss = [self.tokenizer.encode(pred_text) for pred_text in pred_texts]
        gt_tokens = self.tokenizer.encode(gt_text)

        # exact match
        any_match = 0.0
        for pred_text in pred_texts:
            if pred_text == gt_text:
                any_match = 1.0
        self._metrics['best_match'].append(any_match)

        # edit distance
        min_dist = min([
            min(editdistance.eval(gt_tokens, pred_tokens)/max(len(gt_tokens), 1), 1)
            for pred_tokens in pred_tokenss
        ])
        self._metrics['best_edit'].append(min_dist)

        # meteor and gleu
        meteor = max([
            self.meteor.compute(
                predictions=[pred_text_norm],
                references=[gt_text_norm],
            )['meteor']
            for pred_text_norm in pred_texts_norm
        ])
        self._metrics['best_meteor'].append(meteor)
        gleu = max([
            self.gleu.compute(
                predictions=[self.tokenizer.convert_ids_to_tokens(pred_tokens)],
                references=[[self.tokenizer.convert_ids_to_tokens(gt_tokens)]],
            )['google_bleu']
            for pred_tokens in pred_tokenss
        ])
        self._metrics['best_gleu'].append(gleu)

        # token f1
        prec = max([
            self._f1_set(set(pred_tokens), set(gt_tokens))[0]
            for pred_tokens in pred_tokenss
        ])
        self._metrics['best_prec'].append(prec)
        rec = max([
            self._f1_set(set(pred_tokens), set(gt_tokens))[1]
            for pred_tokens in pred_tokenss
        ])
        self._metrics['best_rec'].append(rec)
        f1 = max([
            self._f1_set(set(pred_tokens), set(gt_tokens))[2]
            for pred_tokens in pred_tokenss
        ])
        self._metrics['best_f1'].append(f1)

        # ---- Reference metrics
        cache = defaultdict(list)
        for pred_text, pred_tokens in zip(pred_texts, pred_tokenss):
            pred_refs = self._parse_references(pred_text)
            gt_refs = self._parse_references(gt_text)

            # reference token f1
            pred_ref_tokens = []
            for ref in pred_refs:
                ref_ = self.get_ref(ref)
                if ref_ is not None:
                    pred_ref_tokens.extend(
                        self.tokenizer.encode(self._parse_text(' \n '.join(ref_['contents'])))
                    )
            gt_ref_tokens = []
            for ref in gt_refs:
                ref_ = self.get_ref(ref)
                if ref_ is not None:
                    gt_ref_tokens.extend(
                        self.tokenizer.encode(self._parse_text(' \n '.join(ref_['contents'])))
                    )
            kprec, krec, kf1 = self._f1_set(set(pred_ref_tokens), set(gt_ref_tokens))
            cache['best_kprec'].append(kprec)
            cache['best_krec'].append(krec)
            cache['best_kf1'].append(kf1)

            # reference exact match
            em = pred_refs == gt_refs
            cache['best_ref_em'].append(em)
            em_set = set(pred_refs) == set(gt_refs)
            cache['best_ref_em_set'].append(em_set)

            # reference set precision/recall/F1
            prec, rec, f1 = self._ref_f1_set(preds=set(pred_refs), actuals=set(gt_refs))
            cache['best_ref_precision'].append(prec)
            cache['best_ref_recall'].append(rec)
            cache['best_ref_f1'].append(f1)

            # hallucinated references
            halluc_refs_set = set([r for r in pred_refs if r not in self.ref_names])
            pred_refs_set = set(pred_refs)
            cache['best_ref_halluc'].append(len(halluc_refs_set) / max(1, len(pred_refs_set)))
            # cache for corpus-level metrics
            cache['best_halluc_refs_set'].append(halluc_refs_set)
            cache['best_pred_refs_set'].append(pred_refs_set)

        for k, vs in cache.items():
            if k in ['best_ref_halluc']:
                self._metrics[k].append(np.min(vs))

                best = None
                best_halluc_refs_set = None
                best_pred_refs_set = None
                for halluc_refs_set, pred_refs_set in zip(cache['best_halluc_refs_set'], cache['best_pred_refs_set']):
                    curr = len(halluc_refs_set) / max(1, len(pred_refs_set))
                    if best is None or best > curr:
                        best = curr
                        best_halluc_refs_set = halluc_refs_set
                        best_pred_refs_set = pred_refs_set
                if best_halluc_refs_set is not None and best_pred_refs_set is not None:
                    self._cache['best_halluc_refs_set'].append(best_halluc_refs_set)
                    self._cache['best_pred_refs_set'].append(best_pred_refs_set)
            elif k in ['best_halluc_refs_set', 'best_pred_refs_set']:
                pass
            else:
                self._metrics[k].append(np.max(vs))

    def compute_multiple_corpus_metrics(self):
        # Reference hallucinations
        n_halluc = sum([len(s) for s in self._cache['best_halluc_refs_set']])
        n_refs = sum([len(s) for s in self._cache['best_pred_refs_set']])
        self._corpus_metrics['best_ref_halluc'] = float(n_halluc) / max(1.0, float(n_refs))

    # pick the best candidate by aggregated metric
    def update_best_preds(self, pred_texts, gt_text):
        pred_texts_norm = [self._parse_text(pred_text) for pred_text in pred_texts]
        gt_text_norm = self._parse_text(gt_text)
        pred_tokenss = [self.tokenizer.encode(pred_text) for pred_text in pred_texts]
        gt_tokens = self.tokenizer.encode(gt_text)

        cache = defaultdict(list)

        # gleu
        gleu = [
            self.gleu.compute(
                predictions=[self.tokenizer.convert_ids_to_tokens(pred_tokens)],
                references=[[self.tokenizer.convert_ids_to_tokens(gt_tokens)]],
            )['google_bleu']
            for pred_tokens in pred_tokenss
        ]
        cache['best_gleu'] = gleu

        # token f1
        prec = [
            self._f1_set(set(pred_tokens), set(gt_tokens))[0]
            for pred_tokens in pred_tokenss
        ]
        cache['best_prec'] = prec
        rec = [
            self._f1_set(set(pred_tokens), set(gt_tokens))[1]
            for pred_tokens in pred_tokenss
        ]
        cache['best_rec'] = rec
        f1 = [
            self._f1_set(set(pred_tokens), set(gt_tokens))[2]
            for pred_tokens in pred_tokenss
        ]
        cache['best_f1'] = f1

        # ---- Reference metrics
        for pred_text, pred_tokens in zip(pred_texts, pred_tokenss):
            pred_refs = self._parse_references(pred_text)
            gt_refs = self._parse_references(gt_text)

            # reference token f1
            pred_ref_tokens = []
            for ref in pred_refs:
                ref_ = self.get_ref(ref)
                if ref_ is not None:
                    pred_ref_tokens.extend(
                        self.tokenizer.encode(self._parse_text(' \n '.join(ref_['contents'])))
                    )
            gt_ref_tokens = []
            for ref in gt_refs:
                ref_ = self.get_ref(ref)
                if ref_ is not None:
                    gt_ref_tokens.extend(
                        self.tokenizer.encode(self._parse_text(' \n '.join(ref_['contents'])))
                    )
            kprec, krec, kf1 = self._f1_set(set(pred_ref_tokens), set(gt_ref_tokens))
            cache['best_kprec'].append(kprec)
            cache['best_krec'].append(krec)
            cache['best_kf1'].append(kf1)

            # reference set precision/recall/F1
            prec, rec, f1 = self._ref_f1_set(preds=set(pred_refs), actuals=set(gt_refs))
            cache['best_ref_precision'].append(prec)
            cache['best_ref_recall'].append(rec)
            cache['best_ref_f1'].append(f1)

            # hallucinated references
            halluc_refs_set = set([r for r in pred_refs if r not in self.ref_names])
            pred_refs_set = set(pred_refs)
            cache['best_ref_halluc'].append(len(halluc_refs_set) / max(1, len(pred_refs_set)))
            # cache for corpus-level metrics
            cache['best_halluc_refs_set'].append(halluc_refs_set)
            cache['best_pred_refs_set'].append(pred_refs_set)

        best_sum = [0.0 for _ in pred_tokenss]
        for k, vs in cache.items():
            for i in range(len(best_sum)):
                if k in ['best_ref_halluc']:
                    best_sum[i] -= vs[i]
                elif k in ['best_halluc_refs_set', 'best_pred_refs_set']:
                    pass
                else:
                    best_sum[i] += vs[i]
        best = np.argmax(best_sum)
        for k, vs in cache.items():
            if k in ['best_halluc_refs_set', 'best_pred_refs_set']:
                self._cache[k].append(vs[best])
            else:
                self._metrics[k].append(vs[best])

    def compute_best_corpus_metrics(self):
        # Reference hallucinations
        n_halluc = sum([len(s) for s in self._cache['best_halluc_refs_set']])
        n_refs = sum([len(s) for s in self._cache['best_pred_refs_set']])
        self._corpus_metrics['best_ref_halluc'] = float(n_halluc) / max(1.0, float(n_refs))


def get_ref_names(ds_base, ds_generations):
    # We use this set to check for hallucinations. In some cases, there is a "redirect" version of the name
    # that differs from what we store in naturalproofs, so we additionally collect the references directly
    # from the proofs here.
    ref_names = set()
    refs = ds_base['dataset']['theorems'] + ds_base['dataset']['definitions'] + ds_base['dataset']['others']
    for ref in refs:
        ref_names.add(ref['label'])
    for ds in ds_generations.values():
        for example in ds:
            text = example['target']
            refs_ = parse_reference_titles(text)
            for ref in refs_:
                ref_names.add(ref)
    return ref_names


def full_generation_metrics(
        full_generations, ds_base, ds_generations, redirects, tokenizer,
        return_metrics_object=False
):
    refs = ds_base['dataset']['theorems'] + ds_base['dataset']['definitions'] + ds_base['dataset']['others']
    ref_names = get_ref_names(ds_base, ds_generations)
    id2ref = {x['id']: x for x in refs}
    title2ref = {x['title']: x for x in refs}
    metrics = FullGenerationMetrics('fullgen', tokenizer, ref_names, redirects, title2ref)
    for idx, gen in tqdm(enumerate(full_generations), total=len(full_generations)):
        pred_text = gen['text'] if 'text' in gen else gen['y']
        tid, pid = gen['metadata']
        gt_thm = id2ref[tid]
        gt_proof = gt_thm['proofs'][pid]
        gt_text = '\\n'.join(gt_proof['contents'])
        metrics.update(pred_text, gt_text, logp=gen.get('logp', None), ntokens=gen.get('n_tokens', None))
    metrics.compute_corpus_metrics()
    out = metrics.report()
    if return_metrics_object:
        return out, metrics
    return out


def next_step_metrics(generations, ds_base, ds_generations, redirects, tokenizer,
        return_metrics_object=False, strategy='samples'):
    refs = ds_base['dataset']['theorems'] + ds_base['dataset']['definitions'] + ds_base['dataset']['others']
    ref_names = get_ref_names(ds_base, ds_generations)
    id2ref = {x['id']: x for x in refs}
    title2ref = {x['title']: x for x in refs}
    metrics = NextStepMetrics('nextstep', tokenizer, ref_names, redirects, title2ref)
    for idx, gen in tqdm(enumerate(generations), total=len(generations)):
        lines = gen['output']['proof_lines']
        for line in lines:
            metrics.update(line['greedy'] if 'greedy' in line else line['beam'], line['true'])
            if strategy in line:
                metrics.update_best_preds(line[strategy], line['true'])
    metrics.compute_corpus_metrics()
    metrics.compute_best_corpus_metrics()
    out = metrics.report()
    if return_metrics_object:
        return out, metrics
    return out

