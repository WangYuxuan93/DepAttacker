#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time     : 2020/11/19 4:33 PM
# @Author   : jeffrey
# @File     : mlm_generator.py
# @Software : PyCharm
import argparse
import json
from collections import OrderedDict
import os, sys
import torch
from adversary.lm.bert import Bert

def load_conll(f):
    data = []
    sents = f.read().strip().split("\n\n")
    for sent in sents:
        data.append([line.strip().split("\t") for line in sent.strip().split("\n")])
    return data

class MLM_Generator(object):

    def __init__(self, bert_path, device=None, temperature=1.0, top_k=100, top_p=None, n_mlm_cands=50):
        print ("Loading MLM generator from: {}".format(bert_path))
        self.cand_mlm_model = Bert(bert_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p)
        self.cand_mlm_model.model.eval()
        self.n_mlm_cands = n_mlm_cands

    def _get_mlm_cands(self, tokens, idx, n=50):
        original_word = tokens[idx]
        tmps = tokens.copy()
        tmps[idx] = self.cand_mlm_model.MASK_TOKEN
        masked_text = ' '.join(tmps)

        candidates = self.cand_mlm_model.predict(masked_text, target_word=original_word, n=n)

        return [candidate[0] for candidate in candidates]

    def generate(self, tokens, n=50):
        cands_list = []
        for i in range(len(tokens)):
            cands = self._get_mlm_cands(tokens, i, n=n)
            cands_list.append({"orig":tokens[i], "cands":cands})
        return cands_list