#!/usr/bin/env python
# coding: utf-8

import json
import os.path

from typing import Dict, Callable
from itertools import permutations
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertForMaskedLM
import torch
from torch import nn
from tqdm import tqdm
import copy
import pickle
from torch.nn.functional import pad
from collections import defaultdict
from pathlib import Path
torch.cuda.empty_cache()

BATCH_SIZE = 512  # BS number, don't remember what it was, isn't used in the experiments. Remove after refactor.


# Borrowed the starter code from https://github.com/writerai/fitbert
# Heavily modified, assume there are some strange changes ahead
class FitBert:
    def __init__(
            self,
            model_name="bert-large-uncased",
            reduced_precision=False,
            disable_gpu=False,
    ):
        # self.mask_token = mask_token
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        # self._score = pll_score_batched
        print("device:", self.device)

        self.bert:AutoModelForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_name)
        self.bert.to(self.device)
        # torch.compile(self.bert)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)
        self.mask_token = self.tokenizer.mask_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.input_embeddings = self.bert.get_input_embeddings()
        self.reduced_precision = reduced_precision
        self.float_dtype = torch.float32
        if self.reduced_precision:
            self.float_dtype = torch.float16
            self.input_embeddings.to(dtype=self.float_dtype)
        with torch.no_grad():
            self.mask_token_vector = self.input_embeddings(torch.LongTensor([self.tokenizer.mask_token_id]).to(self.device))[0]
            self.pad_token_vector = self.input_embeddings(torch.LongTensor([self.tokenizer.pad_token_id]).to(self.device))[0]


    @staticmethod
    def soft_top_k(x, k=10):
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        tk = torch.topk(x, k, sorted=False)
        # tk.values[tk.values <= 0] = -torch.inf
        # indices = tk.indices[tk.values > 0]
        # print("!=================")
        # # print(x)
        # print(tk.values)
        # print(tk.indices)
        # assert( torch.sum(tk.values).item() != 0 )
        # print(FitBert.softmax_(tk.values.detach().clone()))
        return torch.zeros_like(x).scatter_(-1, tk.indices, FitBert.softmax_(tk.values.detach().clone())).to(dtype=in_dtype)

    def get_vocab_output_dim(self):
        return self.bert.get_output_embeddings().out_features

    def __call__(self, data, is_split_into_words=False, use_softmax=True, *args, **kwds):
        if is_split_into_words:
            _tokens = self.tokenizer.convert_tokens_to_ids(data)
            _tokens = [self.tokenizer.cls_token_id] + _tokens + [self.tokenizer.sep_token_id]
            tokens = {'input_ids':torch.LongTensor([_tokens]).to(self.device)}
        else:
            tokens = self.tokenizer(data, add_special_tokens=True,padding=True, return_tensors='pt').to(self.device)
        
        # inp = torch.tensor(data, device=self.device)
        # if len(tokens.shape) == 1:
        #     inp = inp.unsqueeze(0)
        # print(tokens)
        # print(tokens.input_ids)
        # print(self.tokenizer.convert_ids_to_tokens(tokens.input_ids[0].tolist()))
        # print("=="*50)
        b = self.bert(**tokens)
        # print(b.logits.shape)
        if use_softmax:
            return self.softmax(b[0])[:, 1:-1, :]
        else:
            return b[0][:, 1:-1, :]
        # return self.softmax(self.bert_am(**tokens, **kwds)[0])[:, 1:, :]
#         return self.bert_am(torch.tensor(self._tokens(data, **kwds)), *args, **kwds)

    # def bert_with_subbed_tokens(self, data, tokens=None, **kwds):
    #     mask = tokens!=self.tokenizer.pad_token_id
    #     return self.softmax(self.bert(inputs_embeds=inp, attention_mask=mask, **kwds)[0])[:, 1:, :]

    def bert_am(self, data, *args, **kwds):
        return self.bert(data, *args, attention_mask=(data!=self.tokenizer.pad_token_id), **kwds)

    def tokenize(self, *args, **kwds):
        return self.tokenizer.tokenize(*args, **kwds)

    # def mask_tokenize(self, tokens, keep_original=False, pad=0):
    #     # tokens = self.tokenize(sent)
    #     if keep_original:
    #         return [self._tokens(tokens, pad=pad)] + self.mask_tokenize(tokens, keep_original=False, pad=pad)
    #     else:
    #         return (seq(tokens)
    #                 .enumerate()
    #                 .starmap(lambda i, x: self._tokens_to_masked_ids(tokens, i, pad=pad))
    #                 .list()
    #                 )

    def mask_tokenize(self, sent, keep_original=False, add_special_tokens=False, padding=False, return_full=False):
        tokens = self.tokenize(sent, add_special_tokens=add_special_tokens, padding=padding)
        # print(tokens)
        tlen = len(tokens)
        offset = 1 if add_special_tokens else 0
        token_mat = [tokens[:] for i in range(tlen - (2*offset))]
        for i in range(offset, tlen-offset):
            token_mat[i-offset][i] = self.tokenizer.mask_token
        if keep_original:
            token_mat = [tokens[:]] + token_mat

        if return_full:
            return token_mat, self.tokenizer(token_mat, add_special_tokens=(not add_special_tokens), is_split_into_words=True, return_tensors='pt')
        return token_mat

    def _tokens_to_masked_ids(self, tokens, mask_ind, pad=0):
        masked_tokens = tokens[:]
        masked_tokens[mask_ind] = self.mask_token
        masked_ids = self._tokens(masked_tokens, pad=pad)
        return masked_ids

    # def _tokens(self, tokens, pad=0, **kwds):
    #     tokens = tokens + [self.pad_token] * pad
    #     return self.tokenizer.convert_tokens_to_ids(tokens)

    @staticmethod
    def softmax(x:torch.Tensor):
        # Break into two functions to minimize the memory impact of calling .exp() on very large tensors.
        # print(torch.softmax(x.float(), dim=-1).shape)
        # print(torch.softmax(x.float(), dim=-1))
        # print(torch.softmax(x.float(), dim=-1).max(dim=-1))
        # raise AssertionError("Don't call this?")

        # in_dtype = x.dtype
        return torch.softmax(x.to(dtype=torch.float32), dim=-1).to(dtype=x.dtype)
        # return FitBert._inn_soft(x.to(dtype=torch.float32).exp()).to(dtype=x.dtype)

    @staticmethod
    def _inn_soft(xexp):
        # print(xexp.max(-1)[0])
        # print(xexp[0].shape)
        # print(xexp[0])
        # print("_inn_soft")
        # print(xexp.sum(-1).shape)
        # print(xexp.sum(-1))
        # print(xexp.max(-1)[0])
        # print((xexp / (xexp.sum(-1)).unsqueeze(-1))[0].shape)
        # print((xexp / (xexp.sum(-1)).unsqueeze(-1))[0])
        return xexp / (xexp.sum(-1)).unsqueeze(-1)

    @staticmethod
    def softmax_(x: torch.Tensor):
        # Break into two functions to minimize the memory impact of calling .exp() on very large tensors.
        # Further reduce memory impact by making it an in-place operation. Beware.
        # Rather, that's what we used to try to do. Now it's easier to accept the overhead,
        # otherwise we run into really rough floating point precision issues (lots of inf and NaN)
        # print("softmax_")
        return FitBert.softmax(x)
        # return FitBert._inn_soft((x.float() + 10e-6).exp_())

    @staticmethod
    def masked_softmax(x: torch.Tensor):
        # print("Masked_softmax")

        mask = (x > 0.0).to(dtype=x.dtype)*-torch.inf
        return torch.softmax(x * mask)
        # x = x.clone().detach()
        # return FitBert._inn_soft(x.float().exp() * (x > 0.0).float())

    def nonlin(self, nonlinearity: str) -> Callable:
        if nonlinearity in standard_nonlins:
            nl = standard_nonlins[nonlinearity]
        elif callable(nonlinearity):
            nl = nonlinearity
        else:
            raise Exception(f"Invalid nonlinearity. Must be callable or one of: {list(standard_nonlins.keys())}")
            nl = standard_nonlins[None]
        # print(f"For {nonlinearity=} chose {nl=}")
        return nl

    def augment(self, vecs, nonlinearity: str, pooling):
        nl=self.nonlin(nonlinearity=nonlinearity)
        if pooling:
            if pooling in ["mean", "avg"]:
                return nl(torch.mean(vecs, dim=0, keepdim=True))
            elif pooling == "max":
                # print(e.shape)
                # print(nl(e).shape)
                # print(torch.max(ent_vecs[nl(e)], dim=0, keepdim=True))
                # print(torch.max(ent_vecs[e], dim=0, keepdim=True))
                return nl(torch.max(vecs, dim=0, keepdim=True)[0])
            elif pooling == "sum":
                return nl(torch.sum(vecs, dim=0, keepdim=True))
            elif callable(pooling):
                return nl(pooling(vecs))
        else:
            return nl(vecs)

    def fuzzy_embed(self, vec):
        # print(vec.to(device=self.device, dtype=self.float_dtype)@self.input_embeddings.weight)
        # print("=================!")
        return vec.to(device=self.device, dtype=self.float_dtype)@self.input_embeddings.weight
    
# FitBert.nonlins = {None: lambda x:x,
#                "softmax": FitBert.softmax,
#                "relu": torch.relu,
#                "relmax": FitBert.masked_softmax,
#                "top10": FitBert.soft_top_k,
#                "top20": lambda x: FitBert.soft_top_k(x, 20),
#                "top50": lambda x: FitBert.soft_top_k(x, 50),
#                "top100": lambda x: FitBert.soft_top_k(x, 100)
#               }
    
standard_nonlins: Dict[str, Callable] = {
            None: lambda x:x,
            "softmax": lambda _x_sm: FitBert.softmax(_x_sm),
            "relu": torch.relu,
            "relmax": FitBert.masked_softmax,
            "top5": lambda x: FitBert.soft_top_k(x, 5),
            "top10": lambda x: FitBert.soft_top_k(x, 10),
            "top25": lambda x: FitBert.soft_top_k(x, 25),
            "top50": lambda x: FitBert.soft_top_k(x, 50),
            "top100": lambda x: FitBert.soft_top_k(x, 100)
            }

def read_punctuation(fb):
    punctuation = fb.tokenizer([".,-()[]{}_=+?!@#$%^&*\\/\"'`~;:|…）（•−"], add_special_tokens=False)['input_ids'][0]
    one_hot_punctuation = torch.ones(fb.bert.get_output_embeddings().out_features, dtype=torch.long)
    one_hot_punctuation[punctuation] = 0
    one_hot_punctuation[1232] = 0
    return one_hot_punctuation


class ScoringMethod(nn.Module):
    def __init__(self, label):
        super(ScoringMethod, self).__init__()
        self.label = label


class PllScoringMethod(ScoringMethod):
    def __init__(self, label):
        super(PllScoringMethod, self).__init__(label)

    def forward(self, probs, origids, return_all=False, **kwargs):
        mask = origids >= 0
        origids[~mask] = 0
        slen = len(probs) - 1
        dia = torch.diag(probs[1:].gather(-1, origids.unsqueeze(0).repeat(slen, 1).unsqueeze(-1)).squeeze(-1), diagonal=0)[mask]
        dia_list = dia.tolist()
        prob = torch.mean(torch.log(dia), dim=-1).detach().item()
        if return_all:
            return prob, dia_list
        return prob


class ComparativeScoringMethod(ScoringMethod):
    def __init__(self, label):
        super(ComparativeScoringMethod, self).__init__(label)

    def forward(self, probs, return_all=False, **kwargs):
        slen = len(probs) - 1
        dia = self.calc(probs[0, :slen], probs[torch.arange(1, slen + 1), torch.arange(slen)])
        dia_list = dia.tolist()
        prob = torch.mean(torch.log(dia), dim=-1).detach().item()
        if return_all:
            return prob, dia_list
        return prob

    def calc(self, p: torch.tensor, q: torch.tensor):
        raise NotImplementedError


class JSD(ComparativeScoringMethod):
    def __init__(self):
        super(JSD, self).__init__("jsd")
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def calc(self, p: torch.tensor, q: torch.tensor):
        m = torch.log((0.5 * (p + q)))
        return 1 - (0.5 * (torch.sum(self.kl(m, p.log()), dim=-1) + torch.sum(self.kl(m, q.log()), dim=-1)))


class PLL(PllScoringMethod):
    def __init__(self):
        super(PLL, self).__init__("pll")


class CSD(ComparativeScoringMethod):
    def __init__(self):
        super(CSD, self).__init__("csd")
        self.csd = torch.nn.CosineSimilarity(dim=1)

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.csd(p, q)


class ESD(ComparativeScoringMethod):
    def __init__(self):
        super(ESD, self).__init__("esd")
        self.pwd = torch.nn.PairwiseDistance()
        self.sqrt = torch.sqrt(torch.tensor(2, requires_grad=False))

    def norm(self, dist):
        return (torch.relu(self.sqrt - dist) + 0.000001) / self.sqrt

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.norm(self.pwd(p, q))


class MSD(ComparativeScoringMethod):
    def __init__(self):
        super(MSD, self).__init__("msd")
        self.mse = torch.nn.MSELoss(reduction="none")

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.mse(p, q).mean(axis=-1)


class HSD(ComparativeScoringMethod):
    def __init__(self):
        super(HSD, self).__init__("hsd")
        self.sqrt = torch.sqrt(torch.tensor(2, requires_grad=False))

    def calc(self, p: torch.Tensor, q: torch.Tensor):
        p = p.clone()
        q = q.clone()
        return 1 - torch.sqrt_(torch.sum(torch.pow(torch.sqrt_(p) - torch.sqrt_(q), 2), dim=-1)) / self.sqrt


KNOWN_METHODS = [CSD(), ESD(), JSD(), MSD(), HSD(), PLL()]
KNOWN_METHODS = {m.label: m for m in KNOWN_METHODS}


def prompt(rel, xy=True, ensure_period=True):
    if xy:
        prompt = rel_info[rel]['prompt_xy']
    else:
        prompt = rel_info[rel]['prompt_yx']
    if ensure_period and prompt[-1] != '.':
        return prompt + "."
    else:
        return prompt


def candidates(prompt:str, choices, return_ments=False):
    for a, b in permutations(choices, 2):
        if return_ments:
            yield prompt.replace("?x", a, 1).replace("?y", b, 1), (a, b)
        else:
            yield prompt.replace("?x", a, 1).replace("?y", b, 1)


# scores equivalently to the old method, even with padding.
# Can be used to batch across examples.
def pll_score_batched(self, sents: list, return_all=False):
    self.bert.eval()
    key_to_sent = {}
    with torch.no_grad():
        data = {}
        for sent in sents:
            tkns = self.tokenizer.tokenize(sent)
            data[len(data)] = {
                'tokens': tkns,
                'len': len(tkns)
            }
        scores = {"pll": {}}
        all_plls = {"pll": {}}

        sents_sorted = list(sorted(data.keys(), key=lambda k: data[k]['len']))

        inds = []
        lens = []

        methods = [PLL()]

        for sent in sents_sorted:
            n_tokens = data[sent]['len']
            if sum(lens) <= BATCH_SIZE:
                inds.append(sent)
                lens.append(n_tokens)
            else:
                # There is at least one sentence.
                # If the count is zero, then its size is larger than the batch size.
                # Send it anyway.
                flag = (len(inds) == 0)
                if flag:
                    inds.append(sent)
                    lens.append(n_tokens)
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)
                inds = [sent]
                lens = [n_tokens]
            if sent == sents_sorted[-1]:
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)

        for d in data:
            data[d].clear()
        data.clear()
        # del all_probs
        if self.device == "cuda":
            torch.cuda.empty_cache()
        for method in scores:
            assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return unsort_flatten(scores)["pll"], unsort_flatten(all_plls)["pll"]
        return unsort_flatten(scores)["pll"]


def unsort_flatten(mapping):
    # print(mapping.keys())
    return {f: list(mapping[f][k] for k in range(len(mapping[f]))) for f in mapping}


def cos_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[CSD()], sents=sents, return_all=return_all)


def euc_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[ESD()], sents=sents, return_all=return_all)


def jsd_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[JSD()], sents=sents, return_all=return_all)


def msd_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[MSD()], sents=sents, return_all=return_all)


def hel_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[HSD()], sents=sents, return_all=return_all)


def all_score_batched(self, sents: list,  return_all=True):
    return score_batched(self, methods=list(KNOWN_METHODS.values()), sents=sents, return_all=return_all)


# # Purely for reference.
# def mask_tokenize(self, sent, keep_original=False, add_special_tokens=False, padding=False, return_full=False):
#     print("S:", type(sent), sent)
#     tokens = self.tokenize(sent, add_special_tokens=add_special_tokens, padding=padding)
#     # print(tokens)
#     tlen = len(tokens)
#     offset = 1 if add_special_tokens else 0
#     token_mat = [tokens[:] for i in range(tlen - (2*offset))]
#     for i in range(offset, tlen-offset):
#         token_mat[i-offset][i] = self.tokenizer.mask_token
#     if keep_original:
#         token_mat = [tokens[:]] + token_mat

#     if return_full:
#         return token_mat, self.tokenizer(token_mat, add_special_tokens=(not add_special_tokens), is_split_into_words=True, return_tensors='pt')
#     return token_mat


# For this one:
# Take a sentence, tokenize it, return all needed information like number of tokens.
def _inner_tokenize_sentence(self, sent, keep_original):
    _, tkns = self.mask_tokenize(sent, keep_original=keep_original, add_special_tokens=True, return_full=True)
    # print(tkns)
    # print(f"TKNS:{len(tkns.input_ids[0]) - 2}")
    return tkns, len(tkns.input_ids[0]) - 2


def score_batched(self, methods, sents: list, return_all=True):
    # Enforce evaluation mode
    self.bert.eval()
    with torch.no_grad():
        data = {}
        for sent in sents:
            # Tokenize every sentence
            # print("S2:", sent)
            tkns, n_tkns = _inner_tokenize_sentence(self, sent, keep_original=True)
            data[len(data)] = {
                'tokens': tkns,
                'len': n_tkns
            }
        # print("Boo")

        scores = {m.label: {} for m in methods}
        all_plls = {m.label: {} for m in methods}

        sents_sorted = list(sorted(data.keys(), key=lambda k: data[k]['len']))

        inds = []
        lens = []

        for sent in sents_sorted:
            n_tokens = data[sent]['len']
            if sum(lens) <= BATCH_SIZE:
                inds.append(sent)
                lens.append(n_tokens)
            else:
                # There is at least one sentence.
                # If the count is zero, then its size is larger than the batch size.
                # Send it anyway.
                flag = (len(inds) == 0)
                if flag:
                    inds.append(sent)
                    lens.append(n_tokens)
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)
                inds = [sent]
                lens = [n_tokens]
            if sent == sents_sorted[-1]:
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)

        for d in data:
            data[d].clear()
        data.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        for method in scores:
            assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return unsort_flatten(scores), unsort_flatten(all_plls)
        return unsort_flatten(scores)


def bert_am(self, data, *args, **kwds):
    return self.bert(data, *args, attention_mask=(data!=self.tokenizer.pad_token_id), **kwds)

def _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all):
    longest = max(lens)

    bert_forward = torch.concat([pad(data[d]['tokens'].input_ids, (0, longest - l), 'constant', self.tokenizer.pad_token_id ) for d,l in zip(inds, lens)], dim=0).to(self.device)
    token_type_ids = torch.concat([pad(data[d]['tokens'].token_type_ids, (0, longest - l), 'constant', 0) for d, l in zip(inds, lens)], dim=0).to(self.device)
    _probs = self.softmax(bert_am(self, bert_forward, token_type_ids=token_type_ids)[0])[:, 1:, :]
    
    del bert_forward

    use_pll = any(["pll" in method.label for method in methods])
    print(["pll" in method.label for method in methods])
    print(use_pll)

    for ind, slen in zip(inds, lens):
        origids = data[ind]['tokens'].input_ids[0][1:-1].to(self.device) if use_pll else None
        for method in methods:
            prob, alls = method(_probs[:slen + 1], origids=origids, return_all=True)
            if return_all:
                assert ind not in all_plls[method.label]
                all_plls[method.label][ind] = alls
            assert ind not in scores[method.label]
            scores[method.label][ind] = prob
            del alls, prob
        _probs = _probs[slen + 1:]
    del _probs


def extend_bert(fb: FitBert, num_blanks:int, tokens_per_blank:int, roberta:bool=False) -> FitBert:
    add_tokens = ['?x', '?y']
    if roberta:
        add_tokens.extend([' ?x', ' ?y'])
        add_tokens.append(f' {fb.mask_token}')
    for e in range(num_blanks):
        for t in range(tokens_per_blank):
            add_tokens.append(f"[ENT_{e}_{t}]")
            if roberta:
                add_tokens.append(f" [ENT_{e}_{t}]")
        add_tokens.append(f"[ENT_{e}_x]")  # Quick hack to help with preprocessing
        if roberta:
            add_tokens.append(f" [ENT_{e}_x]")
    add_tokens.append('[ENT_BEG]')
    add_tokens.append('[ENT_END]')
    if roberta:
        add_tokens.append(' [ENT_BEG]')
        add_tokens.append(' [ENT_END]')
    fb.tokenizer.add_tokens(add_tokens, special_tokens=True)  # Add the tokens to the tokenizer.
    fb.bert.resize_token_embeddings(len(fb.tokenizer))  # Add the tokens to the embedding matrix, initialize with defaults. DO NOT TRAIN.
    fb.token_width = tokens_per_blank
    fb.entity_tokens = fb.tokenizer("".join(add_tokens), add_special_tokens=False)['input_ids']
    fb.input_embeddings = fb.bert.get_input_embeddings().to(dtype=fb.float_dtype)
    return fb


class Document:
    def __init__(self, doc, num, width=1, num_passes=1, mlm: FitBert=None, use_blanks=True, use_ent=True):
        self.doc = doc
        self.num = num
        self.mlm = mlm
        self.overlaps = {}
        self.mentions, self.mention_types, self.m_to_e  = self.read_mentions()
        self.entities = self.read_entities(self.doc['vertexSet'])
        self.relations = set([a[1] for a in self.answers(detailed=False)])
        self.num_passes = num_passes
        self.use_ent = use_ent
        self.width = width
        self.use_blanks = use_blanks
        self._masked_doc = None

    @property
    def masked_doc(self):
        if not self._masked_doc and (self.mlm and self.use_blanks):
            self._masked_doc = self.apply_blank_width(self.mlm, self.width)
        return self._masked_doc
        
    def apply_blank_width(self, mlm, width):
        if mlm:
            self.mlm = mlm
            self.blank_width = width
            if self.blank_width > 0:
                return self.mask_entities()
            else:
                return self.tokenize_entities()

    def contextualize_doc(self):
        self.unmasked_doc = self.contextualize(mask=False)
        return self.unmasked_doc

    def __getitem__(self, item):
        return self.doc[item]
    
    def __contains__(self, item):
        return item in self.doc

    def sentences(self):
        for sent in self['sents']:
            yield " ".join(sent)

    def text(self):
        return " ".join(self.sentences())

    def read_mentions(self):

        # Accumulate all mentions with their position (s, b, e) = (w, t)
        # avoid duplicates
        # Sort by key, ascending
        # return ordered list of mentions (w), mapping from index to type (t).

        mentions = dict()
        m_to_e = dict()
        for i, v in enumerate(self['vertexSet']):
            for m in v:
                s = m['sent_id']
                b, e = m['pos']
                w = self['sents'][s][b:e]
                t = m['type']
                if (s, b, e) not in mentions:
                    mentions[(s, b, e)] = (w, t, i)

        ments = list()
        types = dict()
        for i, (_, v) in enumerate(sorted(mentions.items())):
            w, t, e = v
            ments.append(w)
            m_to_e[i] = e
            types[i] = t
        return ments, types, m_to_e

    @staticmethod
    def read_entities(vertSet):
        ents = {}
        for i, ent in enumerate(vertSet):
            ents[i] = list(set(e['name'] for e in ent))
        return ents

    def tokenize_entities(self):
        # Step 1: Copy
        sents = copy.deepcopy(self['sents'])
        e_beg = '[ENT_BEG]'
        e_end = '[ENT_END]'

        seen_positions = []
        positions = []
        for i, v in enumerate(self['vertexSet']):
            for m in v:
                s = m['sent_id']
                b, e = m['pos']
                if (s, b, e) not in seen_positions:
                    seen_positions.append((s, b, e))
                    positions.append((s, b, e, i))
                else:
                    print(f"Duplicate at {(s, b, e)}")
        positions = list(sorted(positions, reverse=True))

        for s, b, e, _ in positions:
            sents[s][b:e] = [e_beg] + sents[s][b:e] + [e_end]
        sents = sum(sents, [])
        if "roberta" in self.mlm.model_name.lower():
            alt_sents = self.fix_roberta_punctuation(" ".join(sents).replace(" [ENT_BEG]", e_beg).replace(" [ENT_END]", e_end))
            tkns = self.mlm.tokenizer.tokenize(alt_sents, add_special_tokens=False, is_split_into_words=False)
        else:
            tkns = self.mlm.tokenizer.tokenize(sents, add_special_tokens=False, is_split_into_words=True)
        print(tkns, flush=True)
        #     pass
        # assert False
        e = 0
        mentions = []  # Handled.
        mention_mask = []  # Handled.
        mapp = {}
        e = -1
        m = len(positions)
        m_types = [None]*m
        ment_lens = [0]*m
        # m -= 1
        for i, w in enumerate(reversed(tkns)):
            if w == e_end:
                en = i
                e = positions.pop(0)[-1]
                if e not in mapp:
                    mapp[e] = []
                m -= 1
                mapp[e].append(m)
            elif w == e_beg:
                en = 0
                e = -1
            else:
                if e > -1:
                    ment_lens[m] += 1
                mention_mask.append(e > -1)
                mentions.append(m if mention_mask[-1] else -1)
        mentions = list(reversed(mentions))
        mention_mask = list(reversed(mention_mask))


        e_count = len(self['vertexSet'])

        tokens = [t for t in tkns if t.upper() not in [e_beg, e_end] ]
        # print(tokens)
        # print(mentions)
        # assert False
        
        return {
            "length": len(tokens),
            "tokens": tokens,
            "ments": mentions,
            "ment_mask":mention_mask,
            "ment_types":m_types,
            "ment_lens":ment_lens,
            "ents": mapp,
            "m_count": len(m_types),
            "e_count": e_count,
        }

    def fix_roberta_punctuation(self, text:str):
        if "roberta" in self.mlm.model_name.lower():
            for c in '.,;:!?)':
                text = text.replace(f' {c}', c)
            text = text.replace(' - ',  '-')
            text = text.replace(" 's ", "'s ")
        return text


    # Next step: How do we get entity types read from here?
    def mask_entities(self):
        # Step 1: Copy
        sents = copy.deepcopy(self['sents'])
        
        # Step 2: Replace all mention tokens with a placeholder
        # Note that these tokens are tokenized differently from BERT's tokens.
        for i, v in enumerate(self['vertexSet']):
            ent = f'[ENT_{i}_x]'
            for m in v:
                sid = m['sent_id']

                # Check for overlap here.
                w = sents[sid][m['pos'][0]]
                if (w[0:5].upper() == '[ENT_'):
                    self.overlaps[int(w.split("_")[1])] = i
                for r in range(*m['pos']):
                    sents[sid][r] = ent
        e_count = i + 1
        
        # Step 3: Replace all placeholders with single tokens
        mn = 0
        entities = []
        mentions = []
        mention_mask = []
        mapp = {}
        
        tokens = []
        e = -1

        if "roberta" in self.mlm.model_name.lower():
            alt_sents = self.fix_roberta_punctuation(" ".join(" ".join(s) for s in sents))
            tkns = self.mlm.tokenizer.tokenize(alt_sents, add_special_tokens=False, is_split_into_words=False)
        else:
            tkns = self.mlm.tokenizer.tokenize(" ".join(" ".join(s) for s in sents), add_special_tokens=False, is_split_into_words=False)
        print(tkns, flush=True)
        # assert False
        for w in tkns:
            if ('[ENT_' in w[0:6].upper()):
                _e = int(w.split("_")[1])
                if e != _e:
                    e = _e
                    for i in range(self.blank_width):
                        if self.use_ent:
                            tokens.append(w.replace('x', str(i)))
                        else:
                            tokens.append(self.mlm.tokenizer.mask_token)
                        mentions.append(mn)
                        mention_mask.append(True)
                    if e not in mapp:
                        mapp[e] = []
                    mapp[e].append(mn)
                    mn += 1
            else:
                tokens.append(w)
                mentions.append(-1)
                mention_mask.append(False)
                e = -1
        m_types = [None]*mn
        ment_lens = [self.blank_width]*mn
        
        for i, v in enumerate(self['vertexSet']):
            tl = [m['type'] for m in sorted(v, key=lambda x: (x['sent_id'], x['pos'][0]))]
            if i in mapp:
                for m, t in zip(mapp[i], tl):
                    m_types[m] = t
            else:
                # print(f"Missing entity {i} for doc {self.num}")
                # if i in self.overlaps:
                #     print(f"It overlaps with {self.overlaps[i]}")
                # else:
                #     print("It doesn't overlap with anything")
                
                # found = False
                # for ans in self.answers(detailed=False):
                #     # print(ans)
                #     if (ans[0] == i) or (ans[2] == 1):
                #         found=True
                # if found:
                #     print("It WAS an answer entity.")
                pass

        print(tokens)
        # assert False
        
        return {
            "length": len(tokens),
            "tokens": tokens,
            "ments": mentions,
            "ment_mask":mention_mask,
            "ment_types":m_types,
            "ment_lens":ment_lens,
            "ents": mapp,
            "m_count": mn,
            "e_count": e_count,
        }

    def answers(self, detailed=True):
        ans = []
        for an in self['labels']:
            
            if detailed:
                ents = self.entities
                hs = ents[an['h']]
                ts = ents[an['t']]
                r = an['r']
                trips = []
                for h in hs:
                    for t in ts:
                        trips.append((h, r, t))
                ans.append(trips)
            else:
                ans.append((an['h'], an['r'], an['t']))
        return ans

    def answer_prompts(self):
        ents = self.entities()
        if 'labels' in self:
            ans = []
            for an in self['labels']:
                _ans = []
                pmpt = prompt(an['r'])
                for h in ents[an['h']]:
                    for t in ents[an['t']]:
                        _ans.append(pmpt.replace("?x", h, 1).replace("?y", t, 1))
                ans.append(_ans)
            return ans

    def candidate_maps(self, rel:str=None, filt=True):
        if rel:
            rels = [rel]
        else:
            rels = rel_info
        for rel in rels:
            pmpt = prompt(rel)
            prompts = {}
            dom = rel_info[rel]['domain']
            ran = rel_info[rel]['range']
            for a, b in permutations(self.mentions, 2):
                ta, tb = self.mention_types[a], self.mention_types[b]
                if not filt or (ta in dom and tb in ran):
                    prompts[pmpt.replace("?x", a, 1).replace("?y", b, 1)] = ((a, ta), (b, tb))
        return prompts

    def entity_vecs(self, nonlinearity=lambda x:x, pooling=None, passes=0):
        ent_vecs = {}
        ment_inds = {}
        if self.blank_width > 0:
            for e, inds in self.masked_doc['ents'].items():
                ent_vecs[e] = self.mlm.augment(self.ment_vecs[passes][inds], nonlinearity, pooling)
            minus_one = torch.LongTensor([-1]*self.blank_width).cpu()
            for i, s in enumerate(self.masked_doc['ment_lens']):
                ment_inds[i] = minus_one.clone()
        else:
            # Each ent_vecs[e] needs to be the correct corresponding set of vectors.
            # Lengths are available:
            ment_vecs = {}
            _s = 0
            for i, s in enumerate(self.masked_doc['ment_lens']):
                ment_vecs[i] = self.mlm.augment(self.ment_vecs[passes][_s:_s + s], nonlinearity, None)  # We can't pool here.
                ment_inds[i] = self.ment_inds[_s:_s + s]
                if passes > 0:
                    # ment_inds[i] = [-1]*len(ment_inds[i])
                    ment_inds[i] = torch.ones_like(ment_inds[i]) * -1
                _s += s
            for e, inds in self.masked_doc['ents'].items():
                ent_vecs[e] = [ment_vecs[m] for m in inds]
        return ent_vecs, ment_inds
        

def read_document(task_name: str = 'docred', dset: str = 'dev', *, num_blanks=1, num_passes=1, mlm = None, path: str = 'data', doc=-1, verbose=False, use_ent=True):
    if (task_name == 'docred' or task_name == "docred") and dset == 'train':
        dset = 'train_annotated'
    with open(f"{path}/{task_name}/{dset}.json") as datafile:
        jfile = json.load(datafile)
        if doc >= 0:
            yield Document(jfile[doc], doc, width=num_blanks, mlm=mlm, num_passes=num_passes, use_ent=use_ent)
        else:
            for i, doc in enumerate(jfile):
                yield Document(doc, i, width=num_blanks, mlm=mlm, num_passes=num_passes, use_ent=use_ent)


def repair_masking(masked_doc):
    masked = ""
    for token in masked_doc:
        if token[:2] == "##":
            masked += token[2:]
        else:
            if masked:
                masked += " "
            masked += token
    return masked


def mask_vectors(self, sent, keep_original=False, add_special_tokens=False, pad_to=0):
    # tokens = self.tokenize(sent, add_special_tokens=add_special_tokens, padding=padding)
    # print(tokens)
    sent.squeeze_(0)
    # print(sent.shape)
    tlen = len(sent)
    offset = 1 if add_special_tokens else 0
    if pad_to > 0:
        pad_length = pad_to - tlen
        if pad_length > 0:
            sent = torch.cat( (sent, torch.clone(self.pad_token_vector).repeat((pad_length, 1)).to(sent.device)), dim=-2)
    else:
        pad_to = tlen
    token_mat = [torch.clone(sent) for i in range(tlen - (2*offset))]
    for i in range(offset, tlen-offset):
        # print("ti B:", token_mat[i-offset][i])
        token_mat[i-offset][i] = self.mask_token_vector
        # print("ti A:", token_mat[i-offset][i])
    if keep_original:
        token_mat = [torch.clone(sent)] + token_mat
    
    mask = torch.ones((len(token_mat), pad_to), dtype=torch.int)
    mask[:, tlen:pad_to] = 0
    # print(pad_to, tlen, len(token_mat), mask.shape)
    assert mask.shape[-1] == pad_to
    return torch.stack(token_mat), mask


def score_vectors(self, methods, probs, return_all=True):
    # Enforce evaluation mode
    self.bert.eval()
    with torch.no_grad():
        use_pll = "pll" in [method.label for method in methods]

        assert not use_pll, "PLL cannot be used for raw vectors. Please choose a different metric."

        all_plls = {}
        scores = {}
        
        for method in methods:
            prob, alls = method(probs, return_all=True)
            if return_all:
                all_plls[method.label] = alls
            scores[method.label] = prob

        if self.device == "cuda":
            torch.cuda.empty_cache()
        # for method in scores:
        #     assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return scores, all_plls
        return scores


def run_exp(resdir, fb:FitBert = None, task_name='docred', dset='dev', doc=0, num_blanks=2, num_passes=1, use_ent=False, top_k=0, skip=[], model='bert-large-cased', start_at=0):
    roberta = "roberta" in model.lower()
    with torch.no_grad():  # Super enforce no gradients whatsoever.
        torch.cuda.empty_cache()
        if fb is None:
            fb = extend_bert(FitBert(model_name=model), 50, num_blanks, roberta=roberta)
            # nls = {None: lambda x:x, "softmax": FitBert.softmax, "relu":torch.relu}
            global one_hot_punctuation
            one_hot_punctuation = read_punctuation(fb)
        # qx, qy = fb.tokenizer(["?x?y"], add_special_tokens=False)['input_ids'][0]
        # Take a document
        # Find all the entities
        processed_docs = []
        tot = 1000 if dset == 'dev' else 3053
        d: Document = ...  # Silly way to get some IDEs to give better completions.
        for d in read_document(task_name=task_name, dset=dset, doc=doc, num_blanks=num_blanks, mlm=fb, path='data', use_ent=use_ent):
            docfile = f"{resdir}/{task_name}_{model}_{dset}_{d.num}_{num_blanks}b_{num_passes}p.pickle"
            # docfile = f"{resdir}/{(task_name + '_') if task_name != 'docred' else ''}{(model + '_') if model != 'bert-large-cased' else ''}{dset}_{d.num}{'_' + str(num_blanks) + 'blanks' if num_blanks != 2 else ''}{'' if use_ent else '_MASK'}{'' if num_passes == 1 else '_'+str(num_passes)}.pickle"
            # print(docfile)
            # if os.path.exists(docfile) or
            if d.num in skip or d.num < start_at:
                # print(f"Pass {p_doc.num}")
                # print(f"Document {d.num} skipped.", flush=True)
                continue
            print(f"Document {d.num} started.", flush=True)
            md = d.masked_doc
            if len(md['tokens']) <= 510:
                mask = [False] + md['ment_mask'] + [False]
                # print("mt", len(md['tokens']))
                # ents = md['ents']
                _tokens = fb.tokenizer.convert_tokens_to_ids(md['tokens'])
                _tokens = [fb.tokenizer.cls_token_id] + _tokens + [fb.tokenizer.sep_token_id]
                _tokens = torch.LongTensor([_tokens])
                V = fb.get_vocab_output_dim()  # [tokens, vocab(29028)]

                d.ment_inds = _tokens.squeeze(0)[mask]
                # d.ment_inds_masked = d.ment_inds.clone()
                # d.ment_inds_masked[d.ment_inds >= min(fb.entity_tokens)] = -1
                out: Dict[int, torch.Tensor] = dict()
                # Initial pass: Just one-hot vectors as inputs.
                # print("t", _tokens.shape)
                input_vecs = torch.nn.functional.one_hot(_tokens, V).to(fb.float_dtype)
                out[0] = input_vecs.cpu()
                # if num_passes == 0:
                #     # Then take the input tokens and convert them to one-hot vectors.
                # else:
                #     input_vecs = fb.input_embeddings(_tokens.to(fb.device)).cpu()
                #     out[0] = fb.bert(inputs_embeds=input_vecs.to(fb.device)).logits.cpu()
                #     d.ment_inds = torch.LongTensor([-1]*sum(mask))
                # del _tokens
                np = num_passes
                cp = 1
                while np > 0:
                    # Second and further passes: Run through the MLM.
                    # Step 1: Take original input vecs and sub in the entities.
                    if cp > 1:
                        entities = out[cp - 1].squeeze(0)[mask]
                        if top_k > 0:
                            entities = fb.fuzzy_embed(fb.soft_top_k(entities, k=top_k))
                        elif top_k == 0:
                            entities = fb.fuzzy_embed(fb.softmax_(entities.clone()))
                        else:
                            entities = fb.fuzzy_embed(entities)
                        input_embeds.squeeze(0)[mask] = entities.squeeze(0)
                    else:
                        # First pass: Just use normal input embeddings.
                        input_embeds = fb.input_embeddings(_tokens.to(fb.device))

                    # print(f"{cp}: {input_embeds.dtype}!!")
                    # else it's the same vectors already.
                    # Then we do a forward pass, gather the new output logits.
                    with torch.autocast("cuda", dtype=fb.float_dtype):
                        out[cp] = fb.bert(inputs_embeds=input_embeds).logits.detach().float().cpu()
                    # If bert is working with lower precision, increase it here.
                    # if reduced_precision:
                    #     # for cp in out:
                    #     out[cp].float()
                    # print(f"{cp}: {out[cp].dtype}!!")
                    print(cp, out[cp].shape)
                    cp += 1
                    np -= 1
                del _tokens
                
                d.ment_vecs = dict()
                for p in out:
                    if fb.token_width > 0:
                        d.ment_vecs[p] = out[p].squeeze(0)[mask].view(-1, fb.token_width, V)
                    else:
                        d.ment_vecs[p] = out[p].squeeze(0)[mask].view(-1, V)
                    # print(d.ment_vecs[p].shape)
                del out
                torch.cuda.empty_cache()
                print(f"Document {d.num} preprocessed.", flush=True)
                yield d, docfile
                # exit(0)
            else:
                print("skipped", d.num)


def replace_embeddings(x, y, prompt):
    ix = prompt['ix']
    iy = prompt['iy']
    embs = prompt['vecs'].clone()
    # tkns = prompt['input_ids']
    return torch.cat([embs[:,:ix], x, embs[:,ix+1:iy], y, embs[:,iy+1:]], dim=1)


def replace_ids(x, y, prompt):
    ix = prompt['ix']
    iy = prompt['iy']
    embs = prompt['input_ids']
    return torch.cat([embs[:ix], x, embs[ix+1:iy], y, embs[iy+1:]])


def output_to_fuzzy_embeddings(fb: FitBert, v: torch.Tensor):
    # print(v.to(device=fb.device, dtype=fb.float_dtype)@fb.input_embeddings.weight)
    # print("=================!")
    # return vec.to(device=self.device, dtype=self.float_dtype)@self.input_embeddings.weight
    return (v.to(device=fb.device, dtype=fb.float_dtype)@fb.input_embeddings.weight).cpu()


def meminfo():
    f, t = torch.cuda.mem_get_info()
    f = f / (1024 ** 3)
    t = t / (1024 ** 3)
    return f"{f:.2f}g/{t:.2f}g"


def run_many_experiments_orig(task_name, dset, rel_info, nonlins, poolers, scorers, resdir, num_blanks, num_passes=1, max_batch=2000, use_ent=False, skip=[], stopfile='', model='bert-large-cased', start_at=0, reduced_precision=False, big_batch=False):
    # import time
    roberta = "roberta" in model.lower()
    if num_blanks == 0:
        poolers = [None]
    with torch.no_grad():
        torch.cuda.empty_cache()
        fb = extend_bert(FitBert(model_name=model, reduced_precision=reduced_precision), 50, num_blanks, roberta=roberta)
        model = model.split('/')[-1]
        # nls = {None: lambda x:x, "softmax": FitBert.softmax, "relu":torch.relu}
        # global one_hot_punctuation
        # one_hot_punctuation = read_punctuation(fb)
        qx, qy = fb.tokenizer(["?x?y"], add_special_tokens=False)['input_ids'][0]
        if roberta:
            sqx, sqy = fb.tokenizer([" ?x ?y"], add_special_tokens=False)['input_ids'][0]
        prompt_data = {}
        if task_name == "docred" or task_name == "docshred":
            prompts = ['P17', 'P27', 'P131', 'P150', 'P161', 'P175', 'P527', 'P569', 'P570', 'P577']
        elif task_name == "biored":
            prompts = ['Association', 'Bind', 'Negative_Correlation', 'Positive_Correlation']
        else:
            prompts = list(sorted(rel_info.keys()))
        # prompts = ['P17']
        for prompt in prompts:
            pi = dict()
            tkns = fb.tokenizer(rel_info[prompt]['prompt_xy'], return_tensors='pt')['input_ids']    
            if roberta:
                tkns[tkns == sqx] = qx
                tkns[tkns == sqy] = qy
            pi['input_ids'] = tkns[0].cpu()
            pi['ix'] = torch.where(tkns[0] == qx)[0].item()
            pi['iy'] = torch.where(tkns[0] == qy)[0].item()
            # print("Cosine score:", score_batched(fb, [scorer], [prompt])[0]['csd'][0])
            pi['vecs'] = fb.input_embeddings(tkns.to(device=fb.device)).cpu()
            prompt_data[prompt] = pi

        otime = 0
        ntime = 0
        for p_doc, docfile in run_exp(resdir, fb=fb, task_name=task_name, dset=dset, doc=-1, num_blanks=num_blanks, num_passes=num_passes, use_ent=use_ent, skip=skip, model=model, start_at=start_at, top_k=0):
            all_scores = dict()
            for nps in range(0, num_passes):
                print(f"{nps=}")
                if num_blanks > 0 and nps == 0:
                    continue
                if os.path.exists(docfile.replace(f'_{num_passes}p', f'_{nps}p')):
                    print(f"Document {p_doc.num} at {nps} passes skipped.", flush=True)
                    continue
                all_scores = dict()
                for nonlin in nonlins:
                    print(f"{nonlin=}")
                    # print(f"NL: {nonlin}")
                    all_scores[nonlin] = {}
                    # nl = FitBert.nonlins[nonlin]
                    for pooler in poolers:
                        print(f"{pooler=}")
                        # print(f"PL: {pooler}")
                        all_scores[nonlin][pooler] = {}
                        evs, ev_tkns = p_doc.entity_vecs(nonlinearity=nonlin, pooling=pooler, passes=nps)
                        fuzzy_embeds = {e:[output_to_fuzzy_embeddings(fb, v1.unsqueeze(0)) for v1 in evs[e]] for e in evs}
                        e_to_m_map = p_doc.masked_doc['ents']
                        # print(f"A: {torch.cuda.mem_get_info()}")
                        # times = []
                        for prompt_id in prompts: # rel_info:
                            # print(f"{prompt_id=}")
                            if prompt_id.split('|')[0] not in p_doc.relations:
                                continue
                            # nnow = time.time()
                            # print(f"PI: {prompt_id}")
                            ans = [(a[0], a[2]) for a in p_doc.answers(detailed=False) if a[1] == prompt_id.split('|')[0]]
                            # BioRED explicitly states that all relations are non-directional.
                            # This is honestly false, but we marked the ones that aren't clearly non-directional to avoid issues.
                            # For example, "Conversion" is a one-way process between chemicals.
                            # "Bind" is questionable in this regard. It feels one-directional in some circumstances, but I'm not an expert...
                            # The remaining relations are obviously symmetric:
                            # Association, Positive/Negative Correlation, Comparison, Co-Treament, and Drug Interaction.
                            # The only DocRED relation marked symmetric is "sister city".
                            # "spouse" and "sibling" should also be marked as such, though, so that might get updated.
                            # (Those relations aren't examined in these experiments)
                            if rel_info[prompt_id]["symmetric"] == "true":
                                ans.extend([(a[1], a[0]) for a in ans if (a[1], a[0]) not in ans])
                            # print(f"{ans=}")
                            # No sense in setting up a bunch of examples if none are correct.
                            # Maybe a retrieval system (RAG?) can make this selection in the wild?
                            if len(ans) == 0:
                                print(f"{prompt_id=} has noans")
                                continue
                            print(f"Document {p_doc.num} for {nonlin} {pooler} {prompt_id} at {nps} passes.", flush=True)

                            if big_batch:
                                from time import time
                                # stime = time()
                                # scores_old = rme_inner(fb, ans, nps, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id)
                                # otime += time() - stime
                                # stime = time()
                                scores = rme_inner2(fb, ans, nps, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id, max_batch)
                                # ntime += time() - stime
                                # # print(f"{'Original' if dtime < dtime2 else 'New'}: {dtime:.2f} vs {dtime2:.2f}")
                                # with open(f'res/final/{task_name}_{model}_{dset}_{p_doc.num}_{num_blanks}b_{nps}p.pickle', 'rb') as other_scores_pickle:
                                #     scores_c = pickle.load(other_scores_pickle)
                                # # scores_a = sorted(scores_old['pll'], key=lambda x: (x[-1], *x[0:3]))
                                # # scores_b = sorted(scores['pll'], key=lambda x: (x[-1], *x[0:3]))
                                # # scores_c = sorted(scores_c[nonlin][pooler][prompt_id]['pll'], key=lambda x: (x[-1], *x[0:3]))

                                # scores_a = sorted(scores_old['pll'], key=lambda x: tuple(x[0:2]))
                                # scores_b = sorted(scores['pll'], key=lambda x: tuple(x[0:2]))
                                # scores_c = sorted(scores_c[nonlin][pooler][prompt_id]['pll'], key=lambda x: tuple(x[0:2]))

                                # # # print([(a, b, c, d) for a, b, c, d in scores_a])
                                # # # print([(a, b, c, d) for a, b, c, d in scores_b])
                                # def printem(t):
                                #     return f"({t[0]},{t[1]},{t[3]:.4f})"

                                # for i, (a, b, c) in enumerate(zip(scores_a, scores_b, scores_c)):
                                # #     # assert a[0] == b[0], f"{i}: {a} <-> {b}"
                                # #     # assert a[1] == b[1], f"{i}: {a} <-> {b}"
                                # #     # assert a[2] == b[2], f"{i}: {a} <-> {b}"
                                # #     # assert a[2] == c[2], f"{i}: {a} <-> {c}"
                                #     print(f"{i: 4}: {printem(a)}, {printem(b)}, {printem(c)}")
                                # print(f"{'Original' if otime < ntime else 'New'}: {otime:.2f} vs {ntime:.2f}")
                                # exit(0)

                                
                            else:
                                                    # fb: FitBert, ans, nps, nonlins, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id, max_batch
                                scores = rme_inner(fb, ans, nps, None, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id, max_batch)
                                # print([(a, b, c, d) for a, b, c, d in scores['pll']])
                                
                            
                            all_scores[nonlin][pooler][prompt_id] = scores
                            # return
                #exit(0)
                with open(docfile.replace(f'_{num_passes}p', f'_{nps}p'), 'wb') as resfile:
                    pickle.dump(all_scores, resfile)
                if os.path.getsize(stopfile) > 0:
                    break
            #     scores_a = sorted(all_scores['top10'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     scores_c = sorted(all_scores['top20'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     scores_d = sorted(all_scores['softmax'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     # scores_f = sorted(all_scores['top20'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     scores_f = sorted(all_scores[None][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     scores_h = sorted(all_scores['top50'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     # scores_i = sorted(all_scores['top20'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     # with open('res/test/docred_bert-large-cased_dev_0_0b_0p.pickle', 'rb') as f:
            #     #     scores_two = pickle.load(f)
            #     with open('res/test/docred_bert-large-cased_dev_0_0b_0p.pickle', 'rb') as f:
            #         scores_old = pickle.load(f)
            #     scores_b = sorted(scores_old['top10'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     scores_i = sorted(scores_old['top50'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     scores_e = sorted(scores_old['softmax'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
            #     scores_g = sorted(scores_old[None][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))

            #     for z, (a, b, c, d, e, f, g, h, i) in enumerate(zip(scores_a, scores_b, scores_c, scores_d, scores_e, scores_f, scores_g, scores_h, scores_i)):
            #         if z % 10 == 0:
            #             print( "-------------------------------------------------------------------------------------")
            #             print(f"new@10 | old@10 | new@20 |X| new@SM | old@SM |X|  new@N |  OLD@N |X| new@50 | old@50 ")
            #         print(f"{a[3]:.3f} | {b[3]:.3f} | {c[3]:.3f} |X| {d[3]:.3f} | {e[3]:.3f} |X| {f[3]:.3f} | {g[3]:.3f} |X| {h[3]:.3f} | {i[3]:.3f}")
            # exit(0)

            # if p_doc.num == 9:
            #     print(f"{'Original' if otime < ntime else 'New'}: {otime:.2f} vs {ntime:.2f}")
            #     exit(0)
                    
        # if stopfile:
        #     with open('stopper.txt', 'w') as stopfile:
        #         pass


def run_many_experiments(task_name, dset, rel_info, nonlins, poolers, scorers, resdir, num_blanks, num_passes=1, max_batch=2000, use_ent=False, skip=[], stopfile='', model='bert-large-cased', start_at=0, reduced_precision=False, big_batch=False):
    # import time
    roberta = "roberta" in model.lower()
    if num_blanks == 0:
        poolers = [None]
    with torch.no_grad():
        torch.cuda.empty_cache()
        fb = extend_bert(FitBert(model_name=model, reduced_precision=reduced_precision), 50, num_blanks, roberta=roberta)
        model = model.split('/')[-1]
        # nls = {None: lambda x:x, "softmax": FitBert.softmax, "relu":torch.relu}
        # global one_hot_punctuation
        # one_hot_punctuation = read_punctuation(fb)
        qx, qy = fb.tokenizer(["?x?y"], add_special_tokens=False)['input_ids'][0]
        if roberta:
            sqx, sqy = fb.tokenizer([" ?x ?y"], add_special_tokens=False)['input_ids'][0]
        prompt_data = {}
        if task_name == "docred" or task_name == "docshred":
            prompts = ['P17', 'P27', 'P131', 'P150', 'P161', 'P175', 'P527', 'P569', 'P570', 'P577']
        elif task_name == "biored":
            prompts = ['Association', 'Bind', 'Negative_Correlation', 'Positive_Correlation']
        else:
            prompts = list(sorted(rel_info.keys()))
        # prompts = ['P17']
        for prompt in prompts:
            pi = dict()
            tkns = fb.tokenizer(rel_info[prompt]['prompt_xy'], return_tensors='pt')['input_ids']
            if roberta:
                tkns[tkns == sqx] = qx
                tkns[tkns == sqy] = qy
            pi['input_ids'] = tkns[0].cpu()
            pi['ix'] = torch.where(tkns[0] == qx)[0].item()
            pi['iy'] = torch.where(tkns[0] == qy)[0].item()
            # print("Cosine score:", score_batched(fb, [scorer], [prompt])[0]['csd'][0])
            pi['vecs'] = fb.input_embeddings(tkns.to(device=fb.device)).cpu()
            prompt_data[prompt] = pi
        
        otime = 0
        ntime = 0
        pooler = None
        for p_doc, docfile in run_exp(resdir, fb=fb, task_name=task_name, dset=dset, doc=-1, num_blanks=num_blanks, num_passes=num_passes, use_ent=use_ent, skip=skip, model=model, start_at=start_at, top_k=0):
            for nps in range(0, num_passes):
                scores = None
                print(f"{nps=}")
                if num_blanks > 0 and nps == 0:
                    continue
                cur_docfile = docfile.replace(f'_{num_passes}p', f'_{nps}p')
                # cur_docfile = cur_docfile.replace('test', 'final')
                if os.path.exists(cur_docfile):
                    continue
                    print(f"Document {p_doc.num} at {nps} passes found, loading scores.", flush=True)
                    with open(cur_docfile, 'rb') as picklefile:
                        scores = pickle.load(picklefile)
                    if scores:
                        nls = [nl for nl in nonlins if nl not in scores]
                        # nls = nonlins
                        if len(nls) == 0:
                            print("Already finished, skipping.")
                            continue
                        else:
                            print(f"Resuming for NLs: {nls}")
                    else:
                        print("Empty scores.")
                else:
                    if nps == 0:
                        nls = [None]  # The only valid nonlinearity for the first pass is no nonlinearity.
                    else:
                        nls = nonlins
                # for nonlin in nonlins:
                #     print(f"{nonlin=}")
                #     # print(f"NL: {nonlin}")
                #     all_scores[nonlin] = {}
                #     nl = FitBert.nonlins[nonlin]
                # for pooler in poolers:
                #     print(f"{pooler=}")
                #     # print(f"PL: {pooler}")

                # all_scores[nonlin][pooler] = {}
                evs, ev_tkns = p_doc.entity_vecs(nonlinearity=None, pooling=None, passes=nps)
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # Problem is here:
                fuzzy_embeds = {nl: {e:[output_to_fuzzy_embeddings(fb, fb.nonlin(nl)(v1).unsqueeze(0)) for v1 in evs[e]] for e in evs } for nl in nls}
                # exit(0)
                e_to_m_map = p_doc.masked_doc['ents']
                # print(f"A: {torch.cuda.mem_get_info()}")
                # times = []
                # Check to see if we've run this experiment.


                # print("RESETTING SCORES")
                for prompt_id in prompts: # rel_info:
                    # print(f"{prompt_id=}")
                    if prompt_id.split('|')[0] not in p_doc.relations:
                        continue
                    # nnow = time.time()
                    # print(f"PI: {prompt_id}")
                    ans = [(a[0], a[2]) for a in p_doc.answers(detailed=False) if a[1] == prompt_id.split('|')[0]]
                    # BioRED explicitly states that all relations are non-directional.
                    # This is honestly false, but we marked the ones that aren't clearly non-directional to avoid issues.
                    # For example, "Conversion" is a one-way process between chemicals.
                    # "Bind" is questionable in this regard. It feels one-directional in some circumstances, but I'm not an expert...
                    # The remaining relations are obviously symmetric:
                    # Association, Positive/Negative Correlation, Comparison, Co-Treament, and Drug Interaction.
                    # The only DocRED relation marked symmetric is "sister city".
                    # "spouse" and "sibling" should also be marked as such, though, so that might get updated.
                    # (Those relations aren't examined in these experiments)
                    if rel_info[prompt_id]["symmetric"] == "true":
                        ans.extend([(a[1], a[0]) for a in ans if (a[1], a[0]) not in ans])
                    # print(f"{ans=}")
                    # No sense in setting up a bunch of examples if none are correct.
                    # Maybe a retrieval system (RAG?) can make this selection in the wild?
                    if len(ans) == 0:
                        print(f"{prompt_id=} has noans")
                        continue
                    print(f"Document {p_doc.num} for {prompt_id} at {nps} passes.", flush=True)

                    if big_batch:
                        from time import time
                        # stime = time()
                        # scores_old = rme_inner(fb, ans, nps, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id)
                        # otime += time() - stime
                        # stime = time()
                        scores = rme_inner2(fb, scores, ans, nps, nls, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id, max_batch)
                        # ntime += time() - stime
                        # # print(f"{'Original' if dtime < dtime2 else 'New'}: {dtime:.2f} vs {dtime2:.2f}")
                        # with open(f'res/final/{task_name}_{model}_{dset}_{p_doc.num}_{num_blanks}b_{nps}p.pickle', 'rb') as other_scores_pickle:
                        #     scores_c = pickle.load(other_scores_pickle)
                        # # scores_a = sorted(scores_old['pll'], key=lambda x: (x[-1], *x[0:3]))
                        # # scores_b = sorted(scores['pll'], key=lambda x: (x[-1], *x[0:3]))
                        # # scores_c = sorted(scores_c[nonlin][pooler][prompt_id]['pll'], key=lambda x: (x[-1], *x[0:3]))

                        # scores_a = sorted(scores_old['pll'], key=lambda x: tuple(x[0:2]))
                        # scores_b = sorted(scores['pll'], key=lambda x: tuple(x[0:2]))
                        # scores_c = sorted(scores_c[nonlin][pooler][prompt_id]['pll'], key=lambda x: tuple(x[0:2]))

                        # # # print([(a, b, c, d) for a, b, c, d in scores_a])
                        # # # print([(a, b, c, d) for a, b, c, d in scores_b])
                        # def printem(t):
                        #     return f"({t[0]},{t[1]},{t[3]:.4f})"

                        # for i, (a, b, c) in enumerate(zip(scores_a, scores_b, scores_c)):
                        # #     # assert a[0] == b[0], f"{i}: {a} <-> {b}"
                        # #     # assert a[1] == b[1], f"{i}: {a} <-> {b}"
                        # #     # assert a[2] == b[2], f"{i}: {a} <-> {b}"
                        # #     # assert a[2] == c[2], f"{i}: {a} <-> {c}"
                        #     print(f"{i: 4}: {printem(a)}, {printem(b)}, {printem(c)}")
                        # print(f"{'Original' if otime < ntime else 'New'}: {otime:.2f} vs {ntime:.2f}")
                        # exit(0)

                        
                    else:
                                        #  fb: ans, nps, nonlins, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id, max_batch
                        scores = rme_inner(fb, ans, nps, nls, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id, max_batch)
                        # print([(a, b, c, d) for a, b, c, d in scores['pll']])
                # print(scores)
                # print(scores.keys())
                with open(docfile.replace(f'_{num_passes}p', f'_{nps}p'), 'wb') as resfile:
                    print(f"Saving to: {docfile.replace(f'_{num_passes}p', f'_{nps}p')}")
                    pickle.dump(scores, resfile)
                # if os.path.getsize(stopfile) > 0:
                #     break
                # scores_a = sorted(scores['top10'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # scores_c = sorted(scores['top20'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # scores_d = sorted(scores['softmax'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # # scores_f = sorted(all_scores['top20'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # scores_f = sorted(scores[None][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # scores_h = sorted(scores['top50'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # # scores_i = sorted(all_scores['top20'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # # with open('res/test/docred_bert-large-cased_dev_0_0b_0p.pickle', 'rb') as f:
                # #     scores_two = pickle.load(f)
                # with open('res/test/docred_bert-large-cased_dev_0_0b_0p.pickle', 'rb') as f:
                #     scores_old = pickle.load(f)
                # scores_b = sorted(scores_old['top10'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # scores_i = sorted(scores_old['top50'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # scores_e = sorted(scores_old['softmax'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
                # scores_g = sorted(scores_old[None][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))

                # for z, (a, b, c, d, e, f, g, h, i) in enumerate(zip(scores_a, scores_b, scores_c, scores_d, scores_e, scores_f, scores_g, scores_h, scores_i)):
                #     if z % 10 == 0:
                #         print( "-------------------------------------------------------------------------------------")
                #         print(f"new@10 | old@10 | new@20 |X| new@SM | old@SM |X|  new@N |  OLD@N |X| new@50 | old@50 ")
                #     print(f"{a[3]:.3f} | {b[3]:.3f} | {c[3]:.3f} |X| {d[3]:.3f} | {e[3]:.3f} |X| {f[3]:.3f} | {g[3]:.3f} |X| {h[3]:.3f} | {i[3]:.3f}")
            # exit(0)
            # if p_doc.num == 9:
            #     print(f"{'Original' if otime < ntime else 'New'}: {otime:.2f} vs {ntime:.2f}")
            #     exit(0)
                    
        # if stopfile:
        #     with open('stopper.txt', 'w') as stopfile:
        #         pass

def rme_inner(fb: FitBert, ans, nps, nonlins, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id, max_batch) -> dict:
    scores = {}
    for sc in scorers:
        scores[sc.label] = []
    torch.cuda.empty_cache()
    # res = defaultdict(lambda:-float('inf'))
    all_replaced_m = {}
    all_labels_m = {}
    for e1 in fuzzy_embeds:
        for e2 in fuzzy_embeds:
            if e1 != e2:
                _seen = set()
                for v1, m1 in zip(fuzzy_embeds[e1], e_to_m_map[e1]):
                    for v2, m2 in zip(fuzzy_embeds[e2], e_to_m_map[e2]):
                        if nps == 0:
                            vals = tuple(ev_tkns[m1].tolist() + [None] + ev_tkns[m2].tolist())
                            if vals in _seen:
                                continue
                            else:
                                _seen.add(vals)
                        rep_vecs = replace_embeddings(v1, v2, prompt_data[prompt_id])
                        mv, ms = mask_vectors(fb, rep_vecs, keep_original=True, add_special_tokens=True)
                        size = mv.shape[0]
                        if size not in all_replaced_m:
                            all_replaced_m[size] = []
                            all_labels_m[size] = []
                        all_replaced_m[size].append(mv)
                        all_labels_m[size].append((e1, e2, m1, m2))
    for size in all_replaced_m:
        print(f"{size=}")
        _max_batch_resized = max_batch - (max_batch % size)
        _sentences_per_batch = _max_batch_resized // size
        all_labels = all_labels_m[size]
        print(f"{len(all_labels)} candidate statements.", flush=True)
        bert_forward = torch.cat(all_replaced_m[size], dim=0).cpu()
        for v in all_replaced_m[size]:
            del v
        # all_replaced_m[size] = None
        torch.cuda.empty_cache()
        # print(bert_forward.shape)
        # fwd_pieces = []
        print(f"Scoring bert forward ({size}, {len(bert_forward)})", flush=True)
        while len(bert_forward) > 0:
            # print(f"BF: {len(bert_forward)} ({min(_max_batch_resized, len(bert_forward))//size}/{len(all_labels)})", flush=True)
            # print(bert_forward[:_max_batch_resized].to(device=fb.device, dtype=fb.float_dtype)[0].shape)
            # print(bert_forward[:_max_batch_resized].to(device=fb.device, dtype=fb.float_dtype)[0])
            # print(fb.bert(inputs_embeds=bert_forward[:_max_batch_resized].to(device=fb.device, dtype=fb.float_dtype)).logits[0].shape)
            # print(fb.bert(inputs_embeds=bert_forward[:_max_batch_resized].to(device=fb.device, dtype=fb.float_dtype)).logits[0])
            # sm = fb.softmax_(fb.bert(inputs_embeds=bert_forward[:_max_batch_resized].to(device=fb.device, dtype=fb.float_dtype)).logits)[0]
            # print(sm.shape)
            # print(sm)
            # exit(0)
            with torch.autocast("cuda", dtype=fb.float_dtype):
                sm_bert = fb.softmax_(fb.bert(inputs_embeds=bert_forward[:_max_batch_resized].to(device=fb.device, dtype=fb.float_dtype)).logits.detach())[:, 1:, :]
            bert_forward = bert_forward[_max_batch_resized:]
            torch.cuda.empty_cache()
            # sm_bert = fb.softmax_(fb.bert(inputs_embeds=bert_forward.to(fb.device)).logits)[:, 1:, :]
            # print(f"Document {p_doc.num} Past.", flush=True)
            # This will cause issues.
            # sm_bert = torch.cat(fwd_pieces) if len(fwd_pieces) > 1 else fwd_pieces[0]
            # print(len(sm_bert.view(-1, score_len, sm_bert.shape[1], sm_bert.shape[2])), len(all_labels))
            # print(all_labels)
            # print(f"SM: {sm_bert.shape}")
            # print(len(sm_bert.view(-1, score_len, sm_bert.shape[1], sm_bert.shape[2])), len(all_labels))
            for (e1, e2, m1, m2), s in zip(all_labels[:_sentences_per_batch], sm_bert.view(-1, size, sm_bert.shape[1], sm_bert.shape[2])):
                # print("S:", s.shape)
                for scorer in scorers:
                    origids = None
                    if scorer.label == "pll":
                        # Then make the index from its parts, same as the other thing.
                        origids = replace_ids(ev_tkns[m1], ev_tkns[m2], prompt_data[prompt_id])[1:-1].to(fb.device)
                    scores[scorer.label].append((e1, e2, (e1, e2) in ans, scorer(s, origids=origids)))
            all_labels = all_labels[_sentences_per_batch:]
            # print(f"Document {p_doc.num} Tick.", flush=True)
            # for scorer in scorers:
            #     print(scorer.label, len(all_scores[nonlin][pooler][prompt_id][scorer.label]))
            del s
            del sm_bert
            torch.cuda.empty_cache()
        del bert_forward
    return scores

def rme_inner2(fb: FitBert, scores:dict, ans, nps, nonlins, scorers, fuzzy_embeds, e_to_m_map, ev_tkns, prompt_data, prompt_id, max_batch) -> dict:
    if not scores:
        scores = {}
    for nl in nonlins:
        if nl not in scores:
            scores[nl] = {None:{prompt_id: {s.label:[] for s in scorers}}}
        else:
            if prompt_id not in scores[nl][None]:
                scores[nl][None][prompt_id] = {s.label:[] for s in scorers}
            else:
                for s in scorers:
                    if s.label not in scores[nl][None][prompt_id]:
                        scores[nl][None][prompt_id][s.label] = []
    # scores = {nl: {None:{prompt_id: {s.label:[] for s in scorers}}} for nl in nonlins}
    # for sc in scorers:
    #     scores[sc.label] = []
    torch.cuda.empty_cache()
    # res = defaultdict(lambda:-float('inf'))
    all_replaced_m = {}
    all_masks_m = {}
    all_labels_m = {}
    all_sizes_m = {}
    all_nls_m = {}
    maxlen = len(max(ev_tkns.values(), key=len)) * 2 + len(prompt_data[prompt_id]['input_ids']) - 2

    # print(f"start: {maxlen=}")
    for nl in nonlins:
        for e1 in fuzzy_embeds[nl]:
            for e2 in fuzzy_embeds[nl]:
                if e1 != e2:
                    _seen = set()
                    for v1, m1 in zip(fuzzy_embeds[nl][e1], e_to_m_map[e1]):
                        for v2, m2 in zip(fuzzy_embeds[nl][e2], e_to_m_map[e2]):
                            if nps == 0:
                                vals = tuple(ev_tkns[m1].tolist() + [None] + ev_tkns[m2].tolist())
                                if vals in _seen:
                                    continue
                                else:
                                    _seen.add(vals)
                            
                            rep_vecs = replace_embeddings(v1.float(), v2.float(), prompt_data[prompt_id])
                            mv, ms = mask_vectors(fb, rep_vecs, keep_original=True, add_special_tokens=True, pad_to=maxlen)
                            # print(mv.shape, ms.shape)

                            size = maxlen
                            # print(fb.bert(inputs_embeds=mv.to(fb.device), attention_mask=ms.to(fb.device)).logits[:, 1:, :])
                            if size not in all_replaced_m:
                                all_replaced_m[size] = []
                                all_masks_m[size] = []
                                all_labels_m[size] = []
                                all_sizes_m[size] = []
                                all_nls_m[size] = []
                            all_replaced_m[size].append(mv)
                            all_masks_m[size].append(ms)
                            all_sizes_m[size].append(len(mv))
                            all_labels_m[size].append((e1, e2, m1, m2))
                            all_nls_m[size].append(nl)
    # print("end")
    for size in all_replaced_m:
        print(f"{size=}")
        # _max_batch_resized = max_batch - (max_batch % size)
        # TODO: cumsum this to find out how many sentences per batch we actually use.
        # cumsum
        # where > max_batch
        # slice to where - 2
        # _sentences_per_batch = _max_batch_resized // size

        all_labels = all_labels_m[size]
        all_sizes = all_sizes_m[size]
        all_nls = all_nls_m[size]
        print(f"{len(all_labels)} candidate statements.", flush=True)

        # print([v.shape for v in all_replaced_m[size]])

        bert_forward = torch.cat(all_replaced_m[size], dim=0).cpu()
        for v in all_replaced_m[size]:
            del v
        bert_forward_mask = torch.cat(all_masks_m[size], dim=0).cpu()
        # print(sum(all_sizes))
        # print(bert_forward.shape)
        # print(bert_forward_mask.shape)
        for v in all_masks_m[size]:
            del v
        # all_replaced_m[size] = None
        torch.cuda.empty_cache()
        # print(bert_forward.shape)
        # fwd_pieces = []
        print(f"Scoring bert forward ({size}, {len(bert_forward)})", flush=True)
        stride_length = 65536 // size
        if "roberta" in fb.model_name.lower():
            stride_length /= 2
        strides = batchify(all_sizes, stride_length)
        # print(strides)
        # tst = []
        # for st in strides:
        #     tst.append(sum(all_sizes[:st]))
        #     all_sizes = all_sizes[st:]
        # print(tst)
        # print(sum(tst))
        # exit(0)
        # print()
        while len(bert_forward) > 0:

            stride, *strides = strides
            sizes = all_sizes[:stride]
            nls = all_nls[:stride]
            all_sizes = all_sizes[stride:]
            all_nls = all_nls[stride:]
            this_batch = sum(sizes)
            lg = max(sizes) + 1

            print(f"BF: {len(all_labels)} statements left, ({this_batch}/{len(bert_forward)} sequences)", flush=True)
            with torch.autocast("cuda", dtype=fb.float_dtype):
                sm_bert = fb.softmax_(fb.bert(inputs_embeds=bert_forward[:this_batch, :lg].to(fb.device), attention_mask=bert_forward_mask[:this_batch, :lg].to(fb.device)).logits)[:, 1:, :]
            bert_forward = bert_forward[this_batch:]
            bert_forward_mask = bert_forward_mask[this_batch:]
            torch.cuda.empty_cache()
            labels = all_labels[:stride]

            for (e1, e2, m1, m2), size, nl in zip(labels, sizes, nls):
                # s = sm_bert[:size]
                # print("S:", s.shape)
                for scorer in scorers:
                    origids = None
                    if scorer.label == "pll":
                        # Then make the index from its parts, same as the other thing.
                        origids = replace_ids(ev_tkns[m1], ev_tkns[m2], prompt_data[prompt_id])[1:-1].to(fb.device)
                    scores[nl][None][prompt_id][scorer.label].append((e1, e2, (e1, e2) in ans, scorer(sm_bert[:size], origids=origids)))
                sm_bert = sm_bert[size:]
            # print(sm_bert.shape)
            all_labels = all_labels[stride:]
            # all_sizes = all_sizes[_sentences_per_batch:]
            # print(f"Document {p_doc.num} Tick.", flush=True)
            # for scorer in scorers:
            #     print(scorer.label, len(all_scores[nonlin][pooler][prompt_id][scorer.label]))
            # del s
            del sm_bert
            torch.cuda.empty_cache()
        del bert_forward
    return scores
    
def batchify(values, max_per_batch):
    tot = 0
    subsets = []
    num = 0
    for v in values:
        if v + tot > max_per_batch:
            subsets.append(num)
            tot = v
            num = 1
        else:
            tot += v
            num += 1
    subsets.append(num)
    return subsets


if __name__ == '__main__':
    # dset = sys.argv[1]
    # vals = [1, 5, 2, 6, 3, 8, 5, 2, 5, 3, 6, 8, 3, 1, 2, 7 , 6, 2, 4, 5, 4, 7]
    # bats = batchify(vals, 10)
    # print(vals)
    # print(bats)
    # for bat in bats:
    #     vs = vals[:bat]
    #     vals = vals[bat:]
    #     print(sum(vs), vs)
    # exit(0)



    import sys
    num_blanks = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_passes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    data_path = sys.argv[3] if len(sys.argv) > 3 else "data"
    task_name = sys.argv[4] if len(sys.argv) > 4 else "docred"
    data_set = sys.argv[5] if len(sys.argv) > 5 else "dev"
    resdir = sys.argv[6] if len(sys.argv) > 6 else "res"
    max_batch = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
    use_ent = False  # Old parameter, unsuccessful test.
    model = sys.argv[8] if len(sys.argv) > 8 else "bert-large-cased"
    start_at = int(sys.argv[9]) if len(sys.argv) > 9 else 0
    stopfile = sys.argv[10] if len(sys.argv) > 10 else "stopper.txt"
    # SENTS_PER_BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    # BATCH_SIZE = 512
    # # If you have memory issues with some documents, you can add them here to skip. 332 is particularly difficult.
    # TOO_BIG = []  # [332]
    # SENTS_PER_BATCH = 32
    # width = 2
    # resdir = 'res/semantics2024'
    # doc=2
    # num_blanks=2
    # skip = [73, 234]
    # num_passes=3
    # nonlins = ["relu", "softmax", None]
    # poolers = ["mean", "max", None]
    # scorers = [CSD(), ESD(), JSD(), MSD(), HSD()]

    models = {
        "bert": "bert-large-cased",
        "roberta": "roberta-large", # (cased)
        "distilbert": "distilbert-base-cased",  # Doesn't work for some reason? Ranking looks suspicious and follows no trends.
        "biobert": "dmis-lab/biobert-large-cased-v1.1",
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
        "medbert": "Charangan/MedBERT"  # It should work! We just didn't get around to testing it in time...
    }

    if model.lower() in models:
        model = models[model.lower()]

    with open(f'{data_path}/{task_name}/rel_info_full.json', 'r') as rel_info_file:
        rel_info = json.load(rel_info_file)

    assert task_name in ["docred", "docshred", "biored"]
    
    skip = []
    if task_name == 'docred' or task_name == "docshred":
        skip=[723]
        if num_blanks >= 4:
            skip += [121]
    elif task_name == 'biored':
        if num_blanks > 0:
            # skip += [38, 78, 87, 96, 136, 150, 151, 163]
            pass
        if num_blanks > 1:
            pass
            # skip += [20, 32]

    # def run_many_experiments(data, dset, rel_info, nonlins, poolers, scorers, resdir, num_blanks, num_passes=1, max_batch=2000, use_ent=True, skip=[], stopfile='', model='bert-large-cased'):

    os.makedirs(resdir, exist_ok=True)

    torch.set_printoptions(edgeitems=4)

    reduced_precision = False

    if reduced_precision:
        torch.set_float32_matmul_precision("high")

    # try:
    #     os.remove('res/test/docred_bert-large-cased_dev_0_0b_0p.pickle', )
    # except:
    #     pass

    import sys
    print(sys.version)
    # print(sys.version_info)

    # for x, y in FitBert.nonlins.items():
    #     print(x, callable(y), type(y))
    # exit(0)

    # _nps = 2
    # if os.path.exists(f'res/test3/docred_bert-large-cased_dev_0_0b_{_nps}p.pickle'):
    #     print("Loading from files.")
    #     with open(f'res/test3/docred_bert-large-cased_dev_0_0b_{_nps}p.pickle', 'rb') as f:
    #         scores = pickle.load(f)

    #     # print(scores.keys())
    #     # print(scores[None][None].keys())
    #     scores_a = sorted(scores['top10'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     scores_c = sorted(scores['top20'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     scores_d = sorted(scores['softmax'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     # scores_f = sorted(all_scores['top20'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     scores_f = sorted(scores[None][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     scores_h = sorted(scores['top50'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     # scores_i = sorted(all_scores['top20'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     # with open('res/test/docred_bert-large-cased_dev_0_0b_0p.pickle', 'rb') as f:
    #     #     scores_two = pickle.load(f)
    #     with open(f'res/test2/docred_bert-large-cased_dev_0_0b_{_nps}p.pickle', 'rb') as f:
    #         scores_old = pickle.load(f)
    #     scores_b = sorted(scores_old['top10'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     scores_i = sorted(scores_old['top50'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     scores_e = sorted(scores_old['softmax'][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))
    #     scores_g = sorted(scores_old[None][None]['P17']['pll'], key=lambda x: tuple(x[0:2]))

    #     for z, (a, b, c, d, e, f, g, h, i) in enumerate(zip(scores_a, scores_b, scores_c, scores_d, scores_e, scores_f, scores_g, scores_h, scores_i)):
    #         if z % 10 == 0:
    #             print( "-------------------------------------------------------------------------------------")
    #             print(f"new@10 | old@10 | new@20 |X| new@SM | old@SM |X|  new@N |  OLD@N |X| new@50 | old@50 ")
    #         print(f"{a[3]:.3f} | {b[3]:.3f} | {c[3]:.3f} |X| {d[3]:.3f} | {e[3]:.3f} |X| {f[3]:.3f} | {g[3]:.3f} |X| {h[3]:.3f} | {i[3]:.3f}")
    #     exit(0)

    fb = extend_bert(FitBert(model_name="bert-large-cased", reduced_precision=False), 50, 0, roberta=False)



    for _nb in range(num_blanks + 1):
        run_many_experiments(task_name,
                            data_set,
                            rel_info,
                            nonlins=["top10"],#[None, "softmax", "top5", "top10", "top25", "top50", "top100"],
                            poolers=[None],
                            scorers=[PLL()],
                            resdir=resdir,
                            num_blanks=_nb,
                            num_passes=num_passes,
                            max_batch=max_batch,
                            use_ent=use_ent,
                            skip=skip,
                            stopfile=stopfile,
                            model=model,
                            start_at=start_at,
                            reduced_precision=reduced_precision,
                            big_batch=True,
                        )
        
    

    # Note to self: Why in the heck are top50 and top100 identical, but softmax and top10 not?!
    # Answer: Because some of the methods were not being seen as callable methods, and were defaulting to the identity function.
