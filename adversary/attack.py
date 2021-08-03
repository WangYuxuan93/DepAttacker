import os
import sys
import gc
import json
import pickle
import string

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

try:
    from allennlp.modules.elmo import batch_to_ids
except:
    print ("can not import batch_to_ids!")

import time
import argparse
import math
import numpy as np
import torch
import random
#from torch.optim.adamw import AdamW
from torch.optim import SGD, Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from neuronlp2.nn.utils import total_grad_norm
from neuronlp2.io import get_logger, conllx_data, ud_data, conllx_stacked_data #, iterate_data
from neuronlp2.io import ud_stacked_data
from neuronlp2.models.biaffine_parser import BiaffineParser
from neuronlp2.models.stack_pointer_parser import StackPointerParser
from neuronlp2.models.ensemble_parser import EnsembleParser
from neuronlp2.optim import ExponentialScheduler, StepScheduler, AttentionScheduler
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.nn.utils import freeze_embedding
from neuronlp2.io import common
from transformers import AutoTokenizer, AutoModelWithLMHead
from neuronlp2.io.common import PAD, ROOT, END
from neuronlp2.io.batcher import multi_language_iterate_data, iterate_data, iterate_data_dp
from neuronlp2.io import multi_ud_data
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from adversary.attackers.blackbox import BlackBoxAttacker
from adversary.scripts.mlm_generator import load_conll,MLM_Generator
from collections import OrderedDict

def convert_tokens_to_ids(tokenizer, tokens):

    all_wordpiece_list = []
    all_first_index_list = []
    convert_map = {"-LRB-":"(", "-RRB-":")", "-LCB-":"{", "-RCB-":"}", PAD:tokenizer.pad_token,
                 ROOT: tokenizer.cls_token, END:tokenizer.sep_token}
    for toks in tokens:
        """
        toks = [toks_[0], toks_[1]]
        for i in range(2,len(toks_)):
            t = toks_[i]
            # LCB, LRB, `` have left blank
            if t in [PAD, ROOT, END, "-RCB-","-RRB-","--","''"] or t in string.punctuation:
                toks.append(t)
            else:
                toks.append(" "+t)
        """
        wordpiece_list = []
        first_index_list = []
        for i, token in enumerate(toks):
            if token in convert_map:
                token = convert_map[token]
            if not (i == 1 or token in string.punctuation or token in ["--","''",
                tokenizer.pad_token,tokenizer.cls_token, tokenizer.sep_token]):
                token = " "+token
            wordpiece = tokenizer.tokenize(token)
            # add 1 for cls_token <s>
            first_index_list.append(len(wordpiece_list)+1)
            wordpiece_list += wordpiece
            #print (wordpiece)
        #print ("wordpiece_list:\n", wordpiece_list)
        #print (first_index_list)
        bpe_ids = tokenizer.convert_tokens_to_ids(wordpiece_list)
        #print ("bpe_ids:\n", bpe_ids)
        bpe_ids = tokenizer.build_inputs_with_special_tokens(bpe_ids)
        #print (bpe_ids)
        all_wordpiece_list.append(bpe_ids)
        all_first_index_list.append(first_index_list)

    all_wordpiece_max_len = max([len(w) for w in all_wordpiece_list])
    all_wordpiece = np.stack(
          [np.pad(a, (0, all_wordpiece_max_len - len(a)), 'constant', constant_values=tokenizer.pad_token_id) for a in all_wordpiece_list])
    all_first_index_max_len = max([len(i) for i in all_first_index_list])
    all_first_index = np.stack(
          [np.pad(a, (0, all_first_index_max_len - len(a)), 'constant', constant_values=0) for a in all_first_index_list])

    # (batch, max_bpe_len)
    input_ids = torch.from_numpy(all_wordpiece)
    # (batch, seq_len)
    first_indices = torch.from_numpy(all_first_index)

    return input_ids, first_indices

def mlm_gen(input,out_put):
    bert_path="/users2/yxwang/work/data/models/bert-large-uncased"
    temp=1.0
    top_k=100
    n_mlm_cands=50
    with open(input, 'r') as f:
        data = load_conll(f)
    device = torch.device('cuda', 0) #if args.cuda else torch.device('cpu')
    generator = MLM_Generator(bert_path, device=device, temperature=temp,
                            top_k=top_k, top_p=None, n_mlm_cands=n_mlm_cands)

    all_cands = OrderedDict()
    print ("total sent:", len(data))
    for i, sent in enumerate(data):
        if i % 100 == 0:
            print(i,"... ",end="")
            sys.stdout.flush()
        tokens = [line[1] for line in sent]
        cands_list = generator.generate(tokens, args.n_mlm_cands)
        all_cands[i] = cands_list
    json.dump(all_cands, open(out_put, 'w'), indent=4)

def eval(alg, data, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, 
        device, beam=1, batch_size=256, write_to_tmp=True, prev_best_lcorr=0, prev_best_ucorr=0,
        pred_filename=None, tokenizer=None, multi_lan_iter=False, ensemble=False):
    network.eval()
    accum_ucorr = 0.0
    accum_lcorr = 0.0
    accum_total = 0
    accum_ucomlpete = 0.0
    accum_lcomplete = 0.0
    accum_ucorr_nopunc = 0.0
    accum_lcorr_nopunc = 0.0
    accum_total_nopunc = 0
    accum_ucomlpete_nopunc = 0.0
    accum_lcomplete_nopunc = 0.0
    accum_root_corr = 0.0
    accum_total_root = 0.0
    accum_total_inst = 0.0
    accum_recomp_freq = 0.0

    accum_ucorr_err = 0.0
    accum_lcorr_err = 0.0
    accum_total_err = 0
    accum_ucorr_err_nopunc = 0.0
    accum_lcorr_err_nopunc = 0.0
    accum_total_err_nopunc = 0

    all_words = []
    all_postags = []
    all_heads_pred = []
    all_rels_pred = []
    all_lengths = []
    all_src_words = []
    all_heads_by_layer = []

    if hasattr(network, 'use_elmo'):
        use_elmo = network.use_elmo
    else:
        use_elmo = network.module.use_elmo

    if multi_lan_iter:
        iterate = multi_language_iterate_data
    else:
        iterate = iterate_data_dp
        lan_id = None

    if ensemble:
        tokenizers = tokenizer
        tokenizer = tokenizers[0]
        n = len(data) - 1
        data_ = data
        data = data_[0]
        sub_batchers = []
        for d in data_[1:]:
            sub_batchers.append(iter(iterate(d, batch_size)))
        assert len(sub_batchers) == len(tokenizers)-1

    for data in iterate(data, batch_size):
        if multi_lan_iter:
            lan_id, data = data
            lan_id = torch.LongTensor([lan_id]).to(device) 
        words = data['WORD'].to(device)
        chars = data['CHAR'].to(device)
        postags = data['POS'].to(device)
        heads = data['HEAD'].numpy()
        rels = data['TYPE'].numpy()
        lengths = data['LENGTH'].numpy()
        srcs = data['SRC']
        if words.size()[0] == 1 and len(srcs) > 1:
            srcs = [srcs]
        if use_elmo:
            input_elmo = batch_to_ids(srcs)
            input_elmo = input_elmo.to(device)
        else:
            input_elmo = None
        if tokenizer:
            bpes, first_idx = convert_tokens_to_ids(tokenizer, srcs)
            bpes = bpes.to(device)
            first_idx = first_idx.to(device)
        else:
            bpes = first_idx = None
        if ensemble:
            words = [words]
            chars = [chars]
            postags = [postags]
            bpes = [bpes]
            first_idx = [first_idx]
            for batcher, sub_tokenizer in zip(sub_batchers, tokenizers[1:]):
                sub_data = next(batcher, None)
                if sub_tokenizer:
                    sub_bpes, sub_first_idx = convert_tokens_to_ids(sub_tokenizer, srcs)
                    sub_bpes = sub_bpes.to(device)
                    sub_first_idx = sub_first_idx.to(device)
                else:
                    sub_bpes = sub_first_idx = None
                bpes.append(sub_bpes)
                first_idx.append(sub_first_idx)
                lens = sub_data['LENGTH'].numpy()
                assert (lens == lengths).all()
                words.append(sub_data['WORD'].to(device))
                chars.append(sub_data['CHAR'].to(device))
                postags.append(sub_data['POS'].to(device))
        if alg == 'graph':
            pres = data['PRETRAINED'].to(device)
            masks = data['MASK'].to(device)
            #err_types = data['ERR_TYPE']
            err_types = None
            heads_pred, rels_pred = network.decode(words, pres, chars, postags, mask=masks, 
                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id, 
                leading_symbolic=common.NUM_SYMBOLIC_TAGS)
        else:
            pres = None
            err_types = None
            masks = data['MASK_ENC'].to(device)

            heads_pred, rels_pred = network.decode(words, pres, chars, postags, mask=masks, 
                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id, 
                beam=beam, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
        
        if ensemble:
            words = words[0]
            postags = postags[0]

        words = words.cpu().numpy()
        postags = postags.cpu().numpy()

        if write_to_tmp:
            pred_writer.write(words, postags, heads_pred, rels_pred, lengths, symbolic_root=True, src_words=data['SRC'])
        else:
            all_words.append(words)
            all_postags.append(postags)
            all_heads_pred.append(heads_pred)
            all_rels_pred.append(rels_pred)
            all_lengths.append(lengths)
            all_src_words.append(data['SRC'])

        #gold_writer.write(words, postags, heads, rels, lengths, symbolic_root=True)
        #print ("heads_pred:\n", heads_pred)
        #print ("rels_pred:\n", rels_pred)
        #print ("heads:\n", heads)
        #print ("err_types:\n", err_types)
        stats, stats_nopunc, err_stats, err_nopunc_stats, stats_root, num_inst = parser.eval(
                                    words, postags, heads_pred, rels_pred, heads, rels,
                                    word_alphabet, pos_alphabet, lengths, punct_set=punct_set, 
                                    symbolic_root=True, err_types=err_types)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        ucorr_err, lcorr_err, total_err = err_stats
        ucorr_err_nopunc, lcorr_err_nopunc, total_err_nopunc = err_nopunc_stats
        corr_root, total_root = stats_root

        accum_ucorr += ucorr
        accum_lcorr += lcorr
        accum_total += total
        accum_ucomlpete += ucm
        accum_lcomplete += lcm

        accum_ucorr_nopunc += ucorr_nopunc
        accum_lcorr_nopunc += lcorr_nopunc
        accum_total_nopunc += total_nopunc
        accum_ucomlpete_nopunc += ucm_nopunc
        accum_lcomplete_nopunc += lcm_nopunc

        accum_ucorr_err += ucorr_err
        accum_lcorr_err += lcorr_err
        accum_total_err += total_err
        accum_ucorr_err_nopunc += ucorr_err_nopunc
        accum_lcorr_err_nopunc += lcorr_err_nopunc
        accum_total_err_nopunc += total_err_nopunc

        accum_root_corr += corr_root
        accum_total_root += total_root

        accum_total_inst += num_inst

    print('W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr, accum_lcorr, accum_total, accum_ucorr * 100 / accum_total, accum_lcorr * 100 / accum_total,
        accum_ucomlpete * 100 / accum_total_inst, accum_lcomplete * 100 / accum_total_inst))
    print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr_nopunc, accum_lcorr_nopunc, accum_total_nopunc, accum_ucorr_nopunc * 100 / accum_total_nopunc,
        accum_lcorr_nopunc * 100 / accum_total_nopunc,
        accum_ucomlpete_nopunc * 100 / accum_total_inst, accum_lcomplete_nopunc * 100 / accum_total_inst))
    print('Root: corr: %d, total: %d, acc: %.2f%%' %(accum_root_corr, accum_total_root, accum_root_corr * 100 / accum_total_root))
    if accum_total_err == 0:
        accum_total_err = 1
    if accum_total_err_nopunc == 0:
        accum_total_err_nopunc = 1
    #print('Error Token: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
    #    accum_ucorr_err, accum_lcorr_err, accum_total_err, accum_ucorr_err * 100 / accum_total_err, accum_lcorr_err * 100 / accum_total_err))
    #print('Error Token Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
    #    accum_ucorr_err_nopunc, accum_lcorr_err_nopunc, accum_total_err_nopunc, 
    #    accum_ucorr_err_nopunc * 100 / accum_total_err_nopunc, accum_lcorr_err_nopunc * 100 / accum_total_err_nopunc))

    if not write_to_tmp:
        if prev_best_lcorr < accum_lcorr_nopunc or (prev_best_lcorr == accum_lcorr_nopunc and prev_best_ucorr < accum_ucorr_nopunc):
            print ('### Writing New Best Dev Prediction File ... ###')
            pred_writer.start(pred_filename)
            for i in range(len(all_words)):
                pred_writer.write(all_words[i], all_postags[i], all_heads_pred[i], all_rels_pred[i], 
                                all_lengths[i], symbolic_root=True, src_words=all_src_words[i])
            pred_writer.close()

    return (accum_ucorr, accum_lcorr, accum_ucomlpete, accum_lcomplete, accum_total), \
           (accum_ucorr_nopunc, accum_lcorr_nopunc, accum_ucomlpete_nopunc, accum_lcomplete_nopunc, accum_total_nopunc), \
           (accum_root_corr, accum_total_root, accum_total_inst)


def attack(attacker, alg, data, network, pred_writer, adv_gold_writer, punct_set, word_alphabet, pos_alphabet, 
        device, beam=1, batch_size=256, write_to_tmp=True, prev_best_lcorr=0, prev_best_ucorr=0,
        pred_filename=None, tokenizer=None, multi_lan_iter=False, debug=1, pretrained_alphabet=None,
        use_pad=False, cand_cache_path=None, normalize_digits=False, ensemble=False):
    network.eval()
    accum_ucorr = 0.0
    accum_lcorr = 0.0
    accum_total = 0
    accum_ucomlpete = 0.0
    accum_lcomplete = 0.0
    accum_ucorr_nopunc = 0.0
    accum_lcorr_nopunc = 0.0
    accum_total_nopunc = 0
    accum_ucomlpete_nopunc = 0.0
    accum_lcomplete_nopunc = 0.0
    accum_root_corr = 0.0
    accum_total_root = 0.0
    accum_total_inst = 0.0
    accum_recomp_freq = 0.0

    accum_ucorr_err = 0.0
    accum_lcorr_err = 0.0
    accum_total_err = 0
    accum_ucorr_err_nopunc = 0.0
    accum_lcorr_err_nopunc = 0.0
    accum_total_err_nopunc = 0

    accum_total_edit = 0
    accum_total_change_score = 0.0
    accum_total_score = 0.0
    accum_total_perp_diff = 0.0
    accum_success_attack = 0
    accum_total_sent = 0.0
    accum_total_head_change = 0.0
    accum_total_rel_change = 0.0

    all_words = []
    all_postags = []
    all_heads_pred = []
    all_rels_pred = []
    all_lengths = []
    all_src_words = []
    all_heads_by_layer = []

    if multi_lan_iter:
        iterate = multi_language_iterate_data
    else:
        iterate = iterate_data_dp
        lan_id = None

    if ensemble:
        tokenizers = tokenizer
        tokenizer = tokenizers[0]
        word_alphabets = word_alphabet.copy()
        word_alphabet = word_alphabets[0]
        data_ = data
        data = data_[0]
        sub_batchers = []
        for d in data_[1:]:
            sub_batchers.append(iter(iterate(d, batch_size)))
        assert len(sub_batchers) == len(tokenizers)-1

    use_elmo = network.use_elmo

    if cand_cache_path is not None and attacker.cached_cands is None:
        save_cache = True
        if os.path.exists(cand_cache_path):
            print ("Find existing cache file in %s" % cand_cache_path)
            exit()
        all_cand_cache = []
    else:
        save_cache = False

    for data in iterate(data, batch_size):
        if multi_lan_iter:
            lan_id, data = data
            lan_id = torch.LongTensor([lan_id]).to(device)
        words = data['WORD']
        pres = data['PRETRAINED'].to(device) if alg == 'graph' else None
        chars = data['CHAR'].to(device)
        postags = data['POS'].to(device)
        heads = data['HEAD'].numpy()
        rels = data['TYPE'].numpy()
        lengths = data['LENGTH'].numpy()
        
        if ensemble:
            num_models = len(network.networks)
            words = [words]
            chars = [chars]
            postags = [postags]
            for batcher in sub_batchers:
                sub_data = next(batcher, None)
                lens = sub_data['LENGTH'].numpy()
                assert (lens == lengths).all()
                words.append(sub_data['WORD'].to(device))
                chars.append(sub_data['CHAR'].to(device))
                postags.append(sub_data['POS'].to(device))
            adv_words = [t.clone() for t in words]
        else:
            adv_words = words.clone()
        #if ensemble:
        #    adv_words = num_models * [adv_words]
        adv_pres = pres.clone() if alg == 'graph' else None
        adv_src = []
        for i in range(len(lengths)):
            accum_total_sent += 1
            length = lengths[i]
            adv_tokens = data['SRC'][i].copy()
            if ensemble:
                adv_postags = [pos_alphabet.get_instance(w) for w in postags[0][i][:length]]
            else:
                adv_postags = [pos_alphabet.get_instance(w) for w in postags[i][:length]]
            adv_heads = heads[i][:length]
            adv_rels = rels[i][:length]
            adv_rels[0] = 0
            if debug == 3: # Jeffrey
                print ("\n###############################")
                print ("Attacking sent-{}".format(int(accum_total_sent)-1))
                print ("tokens:\n", adv_tokens)
            if debug == 1: print ("original sent:", adv_tokens)
            # TODO：改变sent_id,用于合并中断的攻击语料
            # ************** jeffrey **************
            offset = 0
            sent_id = int(accum_total_sent)-1+offset
            # ********* Jeffrey ***************
            result, cand_cache = attacker.attack(adv_tokens, adv_postags, adv_heads, adv_rels, sent_id=sent_id, debug=debug,
                                                cache=save_cache)
            if save_cache:
                all_cand_cache.append({'sent_id':int(accum_total_sent)-1, 'tokens': cand_cache})
            if result is None:
                adv_src.append(adv_tokens[:length])
                continue
            adv_tokens, adv_infos = result
            num_edit, total_score, total_change_score, total_perp_diff, total_head_change, total_rel_change = adv_infos
            if total_change_score <= 0:
                adv_src.append(data['SRC'][i])
                continue
            accum_success_attack += 1
            accum_total_edit += num_edit
            accum_total_score += total_score
            accum_total_change_score += total_change_score
            accum_total_perp_diff += total_perp_diff
            accum_total_head_change += total_head_change
            accum_total_rel_change += total_rel_change
            if debug == 1: print ("adv sent:", adv_tokens)
            adv_src.append(adv_tokens[:length])
            pre_list = []
            if ensemble:
                for w in adv_tokens:
                    w_ = DIGIT_RE.sub("0", w) if normalize_digits else w
                    pid = pretrained_alphabet.get_index(w_)
                    if pid == 0:
                        pid = pretrained_alphabet.get_index(w_.lower())
                    pre_list.append(pid)
                for j in range(num_models):
                    word_list = []
                    for w in adv_tokens:
                        w_ = DIGIT_RE.sub("0", w) if normalize_digits else w
                        word_list.append(word_alphabets[j].get_index(w_))
                    adv_words[j][i][:length] = torch.from_numpy(np.array(word_list))
            else:
                word_list = []
                for w in adv_tokens:
                    w_ = DIGIT_RE.sub("0", w) if normalize_digits else w
                    word_list.append(word_alphabet.get_index(w_))
                    pid = pretrained_alphabet.get_index(w_)
                    if pid == 0:
                        pid = pretrained_alphabet.get_index(w_.lower())
                    pre_list.append(pid)
                adv_words[i][:length] = torch.from_numpy(np.array(word_list))
            if alg == 'graph':
                adv_pres[i][:length] = torch.from_numpy(np.array(pre_list))
        if ensemble:
            adv_words = [a.to(device) for a in adv_words]
        else:
            adv_words = adv_words.to(device)
        if alg == 'graph':
            adv_pres = adv_pres.to(device)
        #print ("orig_words:\n{}\nadv_words:\n{}".format(words, adv_words))

        if use_elmo:
            input_elmo = batch_to_ids(adv_src)
            input_elmo = input_elmo.to(device)
        else:
            input_elmo = None
        if ensemble:
            bpes, first_idx = [], []
            for tokenizer in tokenizers:
                if tokenizer:
                    sub_bpes, sub_first_idx = convert_tokens_to_ids(tokenizer, adv_src)
                    sub_bpes = sub_bpes.to(device)
                    sub_first_idx = sub_first_idx.to(device)
                else:
                    sub_bpes = sub_first_idx = None
                bpes.append(sub_bpes)
                first_idx.append(sub_first_idx)
        else:
            if tokenizer:
                bpes, first_idx = convert_tokens_to_ids(tokenizer, adv_src)
                bpes = bpes.to(device)
                first_idx = first_idx.to(device)
            else:
                bpes = first_idx = None

        if alg == 'graph':
            masks = data['MASK'].to(device)
            heads_pred, rels_pred = network.decode(adv_words, adv_pres, chars, postags, mask=masks, 
                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id, leading_symbolic=common.NUM_SYMBOLIC_TAGS)
        else:
            masks = data['MASK_ENC'].to(device)
            heads_pred, rels_pred = network.decode(adv_words, adv_pres, chars, postags, mask=masks, 
                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id, beam=beam, leading_symbolic=common.NUM_SYMBOLIC_TAGS)

        if ensemble:
            adv_words = adv_words[0]
            postags = postags[0]
            words = words[0]
        adv_words = adv_words.cpu().numpy()
        postags = postags.cpu().numpy()

        if write_to_tmp:
            pred_writer.write(words, postags, heads_pred, rels_pred, lengths, symbolic_root=True, src_words=data['SRC'] ,adv_words=adv_src)
        else:
            all_words.append(adv_words)
            all_postags.append(postags)
            all_heads_pred.append(heads_pred)
            all_rels_pred.append(rels_pred)
            all_lengths.append(lengths)
            all_src_words.append(adv_src)

        adv_gold_writer.write(words, postags, heads, rels, lengths, symbolic_root=True, src_words=data['SRC'] ,adv_words=adv_src)
        #print ("heads_pred:\n", heads_pred)
        #print ("rels_pred:\n", rels_pred)
        #print ("heads:\n", heads)
        #print ("err_types:\n", err_types)
        stats, stats_nopunc, err_stats, err_nopunc_stats, stats_root, num_inst = parser.eval(
                                    words, postags, heads_pred, rels_pred, heads, rels,
                                    word_alphabet, pos_alphabet, lengths, punct_set=punct_set, 
                                    symbolic_root=True, err_types=None)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        ucorr_err, lcorr_err, total_err = err_stats
        ucorr_err_nopunc, lcorr_err_nopunc, total_err_nopunc = err_nopunc_stats
        corr_root, total_root = stats_root

        accum_ucorr += ucorr
        accum_lcorr += lcorr
        accum_total += total
        accum_ucomlpete += ucm
        accum_lcomplete += lcm

        accum_ucorr_nopunc += ucorr_nopunc
        accum_lcorr_nopunc += lcorr_nopunc
        accum_total_nopunc += total_nopunc
        accum_ucomlpete_nopunc += ucm_nopunc
        accum_lcomplete_nopunc += lcm_nopunc

        accum_ucorr_err += ucorr_err
        accum_lcorr_err += lcorr_err
        accum_total_err += total_err
        accum_ucorr_err_nopunc += ucorr_err_nopunc
        accum_lcorr_err_nopunc += lcorr_err_nopunc
        accum_total_err_nopunc += total_err_nopunc

        accum_root_corr += corr_root
        accum_total_root += total_root

        accum_total_inst += num_inst

    print('W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr, accum_lcorr, accum_total, accum_ucorr * 100 / accum_total, accum_lcorr * 100 / accum_total,
        accum_ucomlpete * 100 / accum_total_inst, accum_lcomplete * 100 / accum_total_inst))
    print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr_nopunc, accum_lcorr_nopunc, accum_total_nopunc, accum_ucorr_nopunc * 100 / accum_total_nopunc,
        accum_lcorr_nopunc * 100 / accum_total_nopunc,
        accum_ucomlpete_nopunc * 100 / accum_total_inst, accum_lcomplete_nopunc * 100 / accum_total_inst))
    print('Root: corr: %d, total: %d, acc: %.2f%%' %(accum_root_corr, accum_total_root, accum_root_corr * 100 / accum_total_root))
    if accum_total_err == 0:
        accum_total_err = 1
    if accum_total_err_nopunc == 0:
        accum_total_err_nopunc = 1
    #print('Error Token: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
    #    accum_ucorr_err, accum_lcorr_err, accum_total_err, accum_ucorr_err * 100 / accum_total_err, accum_lcorr_err * 100 / accum_total_err))
    #print('Error Token Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
    #    accum_ucorr_err_nopunc, accum_lcorr_err_nopunc, accum_total_err_nopunc, 
    #    accum_ucorr_err_nopunc * 100 / accum_total_err_nopunc, accum_lcorr_err_nopunc * 100 / accum_total_err_nopunc))
    if accum_total_edit == 0:
        accum_total_edit = 1
    print('Attack: success/total examples = %d/%d (%.2f%%), AVG perts.: %.2f\nTotal head change: %d, rel change: %d, change score: %.2f\nAverage score: %.2f, change score: %.2f, perp diff: %.2f, change-edit ratio: %.2f' % (
        accum_success_attack, accum_total_sent, accum_success_attack * 100 / float(accum_total_sent), accum_total_edit/accum_total_sent,
        accum_total_head_change, accum_total_rel_change, accum_total_change_score,
        accum_total_score/accum_total_sent, 
        accum_total_change_score/accum_total_sent, accum_total_perp_diff/accum_total_sent, 
        accum_total_change_score/accum_total_edit))

    if not write_to_tmp:
        if prev_best_lcorr < accum_lcorr_nopunc or (prev_best_lcorr == accum_lcorr_nopunc and prev_best_ucorr < accum_ucorr_nopunc):
            print ('### Writing New Best Dev Prediction File ... ###')
            pred_writer.start(pred_filename)
            for i in range(len(all_words)):
                pred_writer.write(all_words[i], all_postags[i], all_heads_pred[i], all_rels_pred[i], 
                                all_lengths[i], symbolic_root=True, src_words=all_src_words[i])
            pred_writer.close()

    if save_cache:
        print ('Saving candidate cache file to %s' % cand_cache_path)
        with open(cand_cache_path, 'w') as cache_f:
            json.dump(all_cand_cache, cache_f, indent=4)

    return (accum_ucorr, accum_lcorr, accum_ucomlpete, accum_lcomplete, accum_total), \
           (accum_ucorr_nopunc, accum_lcorr_nopunc, accum_ucomlpete_nopunc, accum_lcomplete_nopunc, accum_total_nopunc), \
           (accum_root_corr, accum_total_root, accum_total_inst)


def alphabet_equal(a1, a2):
    if a1.size() != a2.size():
        return False
    if a1.items() == a2.items():
        return True
    else:
        return False

def run(args):

    logger = get_logger("Parsing")
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    data_format = args.format

    if data_format == 'conllx':
        data_reader = conllx_data
        test_path = args.test
    elif data_format == 'ud':
        data_reader = ud_data
        test_path = args.test.split(':')
    else:
        print ("### Unrecognized data formate: %s ###" % data_format)
        exit()

    # # *****************************************
    # add_adv = args.add_adv
    # sents = []
    # alpha_adv = 100.0
    # seed=5617
    # np.random.seed(seed)
    #
    # # **************** jeffrey: generate new adv sentence ********************
    # if add_adv == "true":
    #     logger.info("Generating %.2f%% training samples" % alpha_adv)
    #     with open(test_path, "r", encoding="utf-8") as f:
    #         sents = f.read().strip().split("\n\n")
    #         np.random.shuffle(sents)
    #     test_path = os.path.join(args.new_path, "PTB_train_auto.conll_" + str(seed)+ "_" + str(alpha_adv))
    #     with open(test_path, "w", encoding="utf-8") as f:
    #         for ind in range(round(len(sents) * alpha_adv / 100)):
    #             f.write(sents[ind] + "\n\n")

    # 执行脚本,产生新的对应句子顺序的候选词
    # if add_adv == "true":
    #     mlm_gen(test_path, "/users7/zllei/NeuroNLP2/adversary/pkt-train-1.0/mlm_cand.json")
    # *******************************************
    logger.info("mlm candidates is ok!")


    print(args)
    punctuation = args.punctuation
    pretrained_lm = args.pretrained_lm
    lm_path = args.lm_path
    

    if args.ensemble:
        model_paths = args.model_path.split(':')
        n = len(model_paths)
        word_alphabets, char_alphabets, pos_alphabets, rel_alphabets, pretrained_alphabets = n*[None],n*[None],n*[None],n*[None],n*[None]
        num_words, num_chars, num_pos, num_rels, num_pretrained = n*[None],n*[None],n*[None],n*[None],n*[None]
        # load alphabet from different paths
        for i, model_path in enumerate(model_paths):
            logger.info("Creating Alphabets-%d"% i)
            alphabet_path = os.path.join(model_path, 'alphabets')
            assert os.path.exists(alphabet_path)
            word_alphabets[i], char_alphabets[i], pos_alphabets[i], rel_alphabets[i] = data_reader.create_alphabets(alphabet_path, None, 
                                            normalize_digits=args.normalize_digits, pos_idx=args.pos_idx, log_name="Create Alphabets-%d"%i)
            pretrained_alphabets[i] = utils.create_alphabet_from_embedding(alphabet_path)
            if not alphabet_equal(rel_alphabets[0], rel_alphabets[i]):
                logger.info("Label alphabet mismatch: ({}) vs. ({})".format(model_paths[0], model_paths[i]))
                exit()

            num_words[i] = word_alphabets[i].size()
            num_chars[i] = char_alphabets[i].size()
            num_pos[i] = pos_alphabets[i].size()
            num_rels[i] = rel_alphabets[i].size()
            num_pretrained[i] = pretrained_alphabets[i].size()

            logger.info("Word Alphabet Size: %d" % num_words[i])
            logger.info("Pretrained Alphabet Size: %d" % num_pretrained[i])
            logger.info("Character Alphabet Size: %d" % num_chars[i])
            logger.info("POS Alphabet Size: %d" % num_pos[i])
            logger.info("Rel Alphabet Size: %d" % num_rels[i])
        
        model_path = model_paths[0]
        hyps = [json.load(open(os.path.join(path, 'config.json'), 'r')) for path in model_paths]
        model_type = hyps[0]['model']
        assert model_type in ['Biaffine', 'StackPointer']
    else:
        model_path = args.model_path
        model_name = os.path.join(model_path, 'model.pt')
    
        logger.info("Creating Alphabets")
        alphabet_path = os.path.join(model_path, 'alphabets')
        assert os.path.exists(alphabet_path)
        word_alphabet, char_alphabet, pos_alphabet, rel_alphabet = data_reader.create_alphabets(alphabet_path, None, 
                                        normalize_digits=args.normalize_digits, pos_idx=args.pos_idx)
        pretrained_alphabet = utils.create_alphabet_from_embedding(alphabet_path)

        num_words = word_alphabet.size()
        num_chars = char_alphabet.size()
        num_pos = pos_alphabet.size()
        num_rels = rel_alphabet.size()
        num_pretrained = pretrained_alphabet.size()

        logger.info("Word Alphabet Size: %d" % num_words)
        logger.info("Pretrained Alphabet Size: %d" % num_pretrained)
        logger.info("Character Alphabet Size: %d" % num_chars)
        logger.info("POS Alphabet Size: %d" % num_pos)
        logger.info("Rel Alphabet Size: %d" % num_rels)

        hyps = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
        model_type = hyps['model']
        assert model_type in ['Biaffine', 'StackPointer']

    result_path = os.path.join(model_path, 'tmp')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    logger.info("loading network...")
    
    num_lans = 1
    if data_format == 'ud' and not args.mix_datasets:
        lans_train = args.lan_train.split(':')
        lans_dev = args.lan_dev.split(':')
        lans_test = args.lan_test.split(':')
        #languages = set(lans_train + lans_dev + lans_test)
        language_alphabet = utils.creat_language_alphabet(alphabet_path)
        num_lans = language_alphabet.size()
        data_reader = multi_ud_data

    logger.info("##### Parser Type: {} #####".format(model_type))
    alg = 'transition' if model_type == 'StackPointer' else 'graph'
    if args.ensemble:
        network = EnsembleParser(hyps, num_pretrained, num_words, num_chars, num_pos, num_rels, 
                                   device=device, pretrained_lm=args.pretrained_lm, lm_path=args.lm_path,
                                   model_type=model_type,
                                   use_pretrained_static=args.use_pretrained_static, 
                                   use_random_static=args.use_random_static,
                                   use_elmo=args.use_elmo, elmo_path=args.elmo_path,
                                   num_lans=num_lans, model_paths=model_paths, merge_by=args.merge_by)
        tokenizers = []
        for pretrained_lm, lm_path in zip(network.pretrained_lms, network.lm_paths):
            if pretrained_lm == 'none':
                tokenizer = None 
            else:
                tokenizer = AutoTokenizer.from_pretrained(lm_path)
            tokenizers.append(tokenizer)
        tokenizer = tokenizers
    else:
        if model_type == 'Biaffine':
            network = BiaffineParser(hyps, num_pretrained, num_words, num_chars, num_pos, num_rels,
                                   device=device, pretrained_lm=args.pretrained_lm, lm_path=args.lm_path,
                                   use_pretrained_static=args.use_pretrained_static, 
                                   use_random_static=args.use_random_static,
                                   use_elmo=args.use_elmo, elmo_path=args.elmo_path,
                                   num_lans=num_lans)
        elif model_type == 'StackPointer':
            network = StackPointerParser(hyps, num_pretrained, num_words, num_chars, num_pos, num_rels,
                                   device=device, pretrained_lm=args.pretrained_lm, lm_path=args.lm_path,
                                   use_pretrained_static=args.use_pretrained_static, 
                                   use_random_static=args.use_random_static,
                                   use_elmo=args.use_elmo, elmo_path=args.elmo_path,
                                   num_lans=num_lans)
        else:
            raise RuntimeError('Unknown model type: %s' % model_type)

        network = network.to(device)
        network.load_state_dict(torch.load(model_name, map_location=device),strict=False)

        if pretrained_lm in ['none']:
            tokenizer = None 
        else:
            tokenizer = AutoTokenizer.from_pretrained(lm_path)

    if args.cand.endswith('.json'):
        cands = json.load(open(args.cand, 'r'))
        candidates = {int(i):dic for (i,dic) in cands.items()}
    else:
        candidates = pickle.load(open(args.cand, 'rb'))
    vocab = json.load(open(args.vocab, 'r'))
    synonyms = json.load(open(args.syn, 'r'))
    num_gpu = torch.cuda.device_count()
    if num_gpu >= 2:
        lm_device = torch.device('cuda', 1)
    else:
        lm_device = device
    logger.info("parser device:{}, lm device:{}".format(device, lm_device))
    if args.adv_lm_path is not None:
        adv_tokenizer = AutoTokenizer.from_pretrained(args.adv_lm_path)
        adv_lm = AutoModelWithLMHead.from_pretrained(args.adv_lm_path)
        adv_lm = adv_lm.to(lm_device)
        adv_lms = (adv_tokenizer,adv_lm)
    else:
        adv_lms = None
    filters = args.filters.split(':')
    generators = args.generators.split(':')
    if args.ensemble:
        alphabets = word_alphabets, char_alphabets, pos_alphabets, rel_alphabets[0], pretrained_alphabets
    else:
        alphabets = word_alphabet, char_alphabet, pos_alphabet, rel_alphabet, pretrained_alphabet
    if args.mode == 'black':
        attacker = BlackBoxAttacker(network, candidates, vocab, synonyms, filters=filters, generators=generators,
                        max_mod_percent=args.max_mod_percent, tagger=args.tagger, ensemble=args.ensemble,
                        punct_set=punct_set, beam=args.beam, normalize_digits=args.normalize_digits,
                        cached_path=args.cached_path, train_vocab=args.train_vocab, knn_path=args.knn_path, 
                        max_knn_candidates=args.max_knn_candidates, sent_encoder_path=args.sent_encoder_path,
                        min_word_cos_sim=args.min_word_cos_sim, min_sent_cos_sim=args.min_sent_cos_sim, 
                        cand_mlm=args.cand_mlm, dynamic_mlm_cand=args.dynamic_mlm_cand, temperature=args.temp, 
                        top_k=args.top_k, top_p=args.top_p, 
                        n_mlm_cands=args.n_mlm_cands, mlm_cand_file=args.mlm_cand_file,
                        adv_lms=adv_lms, rel_ratio=args.adv_rel_ratio, fluency_ratio=args.adv_fluency_ratio,
                        ppl_inc_thres=args.ppl_inc_thres,
                        alphabets=alphabets, tokenizer=tokenizer, device=device, lm_device=lm_device,
                        batch_size=args.adv_batch_size, random_backoff=args.random_backoff, wordpiece_backoff=args.wordpiece_backoff)

    #tokens = ["_ROOT", "The", "Dow", "fell", "22.6", "%", "on", "black", "Monday"]#, "."]
    #tags = ["_ROOT_POS", "DT", "NNP", "VBD", "CD", ".", "IN", "NNP", "NNP"]#, "."]
    #heads = [0, 2, 3, 0, 5, 3, 3, 8, 6]#, 3]
    #rels = [0, 3, 4, 5, 6, 7, 8, 9, 10]#, 11]
    #attacker.attack(tokens, tags, heads, rels, True)
    #exit()


    logger.info("Reading Data")
    if args.ensemble:
        n = len(word_alphabets)
        data_tests = [None] * n
        for i in range(n):
            if alg == 'graph':
                if data_format == 'ud' and not args.mix_datasets:
                    data_tests[i] = data_reader.read_data(test_path, word_alphabets[i], char_alphabets[i], pos_alphabets[i], 
                                                    rel_alphabets[i], normalize_digits=args.normalize_digits, 
                                                    symbolic_root=True, pre_alphabet=pretrained_alphabets[i], 
                                                    pos_idx=args.pos_idx, lans=lans_test, 
                                                    lan_alphabet=language_alphabet)
                else:
                    data_tests[i] = data_reader.read_data(test_path, word_alphabets[i], char_alphabets[i], pos_alphabets[i], 
                                                  rel_alphabets[i], normalize_digits=args.normalize_digits, 
                                                  symbolic_root=True, pre_alphabet=pretrained_alphabets[i], 
                                                  pos_idx=args.pos_idx)
            elif alg == 'transition':
                prior_order = hyps[0]['input']['prior_order']
                if data_format == "conllx":
                    data_tests[i] = conllx_stacked_data.read_data(test_path, word_alphabets[i], char_alphabets[i], pos_alphabets[i], rel_alphabets[i],
                                                        normalize_digits=args.normalize_digits,
                                                        pos_idx=args.pos_idx, 
                                                        prior_order=prior_order)
                else:
                    data_tests[i] = ud_stacked_data.read_data(test_path, word_alphabets[i], char_alphabets[i], pos_alphabets[i], rel_alphabets[i],
                                                        normalize_digits=args.normalize_digits, symbolic_root=True,
                                                        pre_alphabet=pretrained_alphabets[i], pos_idx=args.pos_idx, 
                                                        prior_order=prior_order)
        word_alphabet, char_alphabet, pos_alphabet, rel_alphabet = word_alphabets[0], char_alphabets[0], pos_alphabets[0], rel_alphabets[0]
        pretrained_alphabet = pretrained_alphabets[0]
        data_test = data_tests
    else:
        if alg == 'graph':
            if data_format == 'ud' and not args.mix_datasets:
                data_test = data_reader.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, 
                                                rel_alphabet, normalize_digits=args.normalize_digits, 
                                                symbolic_root=True, pre_alphabet=pretrained_alphabet, 
                                                pos_idx=args.pos_idx, lans=lans_test, 
                                                lan_alphabet=language_alphabet)
            else:
                data_test = data_reader.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, 
                                              rel_alphabet, normalize_digits=args.normalize_digits, 
                                              symbolic_root=True, pre_alphabet=pretrained_alphabet, 
                                              pos_idx=args.pos_idx)
        elif alg == 'transition':
            prior_order = hyps['input']['prior_order']
            if data_format == "conllx":
                data_test = conllx_stacked_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet,
                                                    normalize_digits=args.normalize_digits,
                                                    pos_idx=args.pos_idx, 
                                                    prior_order=prior_order)
            else:
                data_test = ud_stacked_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet,
                                                    normalize_digits=args.normalize_digits, symbolic_root=True,
                                                    pre_alphabet=pretrained_alphabet, pos_idx=args.pos_idx, 
                                                    prior_order=prior_order)


    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    adv_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    adv_gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    if args.output_filename:
        pred_filename = args.output_filename
    else:
        pred_filename = os.path.join(result_path, 'pred.conll')
    pred_writer.start(pred_filename)
    if args.adv_filename:
        adv_filename = args.adv_filename
    else:
        adv_filename = os.path.join(result_path, 'adv.conll')
    if args.adv_gold_filename:
        adv_gold_filename = args.adv_gold_filename
    else:
        adv_gold_filename = os.path.join(result_path, 'adv_gold.conll')
    adv_writer.start(adv_filename)
    adv_gold_writer.start(adv_gold_filename)
    #gold_filename = os.path.join(result_path, 'gold.txt')
    #gold_writer.start(gold_filename)

    if alg == 'graph' and data_format == 'ud' and not args.mix_datasets:
        multi_lan_iter = True
    else:
        multi_lan_iter = False
    with torch.no_grad():
        print('Parsing Original Data...')
        start_time = time.time()
        eval(alg, data_test, network, pred_writer, gold_writer, punct_set, word_alphabet, 
            pos_alphabet, device, args.beam, batch_size=args.batch_size, tokenizer=tokenizer, 
            multi_lan_iter=multi_lan_iter, ensemble=args.ensemble)
        print('Time: %.2fs' % (time.time() - start_time))
    print ('\n------------------\n')
    with torch.no_grad():
        print('Attacking...')
        logger.info("use pad in input to attacker: {}".format(args.use_pad))
        start_time = time.time()
        if args.ensemble:
            word_alphabet = word_alphabets
            #data_test = data_test[0]
        # debug = 1: show orig/adv tokens / debug = 2: show log inside attacker
        # 通过debug=0来减少日志输出  jeffrey
        attack(attacker, alg, data_test, network, adv_writer, adv_gold_writer, punct_set, word_alphabet, 
            pos_alphabet, device, args.beam, batch_size=args.batch_size, tokenizer=tokenizer, 
            multi_lan_iter=multi_lan_iter, debug=3, pretrained_alphabet=pretrained_alphabet,
            use_pad=args.use_pad, cand_cache_path=args.cand_cache_path, ensemble=args.ensemble)
        print('Time: %.2fs' % (time.time() - start_time))
        

    pred_writer.close()
    #gold_writer.close()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['black', 'random', 'gray', 'gray_single'], required=True, help='processing mode')
    args_parser.add_argument('--seed', type=int, default=666, help='Random seed for torch and numpy (-1 for random)')
    args_parser.add_argument('--config', type=str, help='config file')
    args_parser.add_argument('--vocab', type=str, help='vocab file for attacker')
    args_parser.add_argument('--cand', type=str, help='candidate file for attacker')
    args_parser.add_argument('--syn', type=str, help='synonym file for attacker')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    args_parser.add_argument('--eval_batch_size', type=int, default=256, help='Number of sentences in each batch while evaluating')
    args_parser.add_argument('--noscreen', action='store_true', default=True, help='do not print middle log')
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--pos_idx', type=int, default=4, choices=[3, 4], help='Index in Conll file line for Part-of-speech tags')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--use_pretrained_static', action='store_true', help='Whether to use pretrained static word embedding.')
    args_parser.add_argument('--use_random_static', action='store_true', help='Whether to use extra randomly initialized trainable word embedding.')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words')
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters')
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--use_elmo', action='store_true', default=False, help='Use elmo as input?')
    args_parser.add_argument('--elmo_path', default=None, help='path for pretrained elmo')
    args_parser.add_argument('--pretrained_lm', default='none', choices=['none', 'elmo', 'bert', 'bart', 'roberta', 'xlm-r', 'electra', 'tc_bert', 'tc_bart', 'tc_roberta', 'tc_electra'], help='Pre-trained language model')
    args_parser.add_argument('--lm_path', help='path for pretrained language model')
    args_parser.add_argument('--normalize_digits', default=False, action='store_true', help='normalize digits to 0 ?')
    args_parser.add_argument('--mix_datasets', default=False, action='store_true', help='Mix dataset from different languages ? (should be False for CPGLSTM)')
    args_parser.add_argument('--format', type=str, choices=['conllx', 'ud'], default='conllx', help='data format')
    args_parser.add_argument('--lan_train', type=str, default='en', help='lc for training files (split with \':\')')
    args_parser.add_argument('--lan_dev', type=str, default='en', help='lc for dev files (split with \':\')')
    args_parser.add_argument('--lan_test', type=str, default='en', help='lc for test files (split with \':\')')
    args_parser.add_argument('--train', help='path for training file.')
    args_parser.add_argument('--dev', help='path for dev file.')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--adv_lm_path', help='path for pretrained language model (gpt2) for adv filtering')
    args_parser.add_argument('--cand_mlm', help='path for mlm candidate generating')
    args_parser.add_argument('--mlm_cand_file', help='path for mlm candidate preprocessed json file')
    args_parser.add_argument('--dynamic_mlm_cand', action='store_true', default=False, help='Whether generate MLM candidates dynamically')
    args_parser.add_argument('--temp', type=float, default=1.0, help='Temperature for mlm candidate generating')
    args_parser.add_argument('--n_mlm_cands', type=int, default=50, help='Select candidate number for mlm candidate generating')
    args_parser.add_argument('--top_k', type=int, default=100, help='Top candidate number for filtering mlm candidate generating')
    args_parser.add_argument('--top_p', type=float, default=None, help='Top proportion for filtering mlm candidate generating')
    args_parser.add_argument('--output_filename', type=str, help='output filename for parse')
    args_parser.add_argument('--adv_filename', type=str, help='output adversarial filename')
    args_parser.add_argument('--adv_gold_filename', type=str, help='output adversarial text with gold heads & rels')
    args_parser.add_argument('--adv_rel_ratio', type=float, default=0.5, help='Relation importance in adversarial attack')
    args_parser.add_argument('--adv_fluency_ratio', type=float, default=0.2, help='Fluency importance in adversarial attack')
    args_parser.add_argument('--ppl_inc_thres', type=float, default=20.0, help='Perplexity difference threshold in adversarial attack')
    args_parser.add_argument('--max_mod_percent', type=float, default=0.05, help='Maximum modification percentage of words')
    args_parser.add_argument('--adv_batch_size', type=int, default=16, help='Number of sentences in adv lm each batch')
    args_parser.add_argument('--random_backoff', action='store_true', default=False, help='randomly substitute if no change')
    args_parser.add_argument('--wordpiece_backoff', action='store_true', default=False, help='choose longest wordpiece substitute if no change')
    args_parser.add_argument('--knn_path', type=str, help='knn embedding path for adversarial attack')
    args_parser.add_argument('--max_knn_candidates', type=int, default=50, help='max knn candidate number')
    args_parser.add_argument('--min_word_cos_sim', type=float, default=0.9, help='Min word cos similarity')
    args_parser.add_argument('--min_sent_cos_sim', type=float, default=0.9, help='Min sent cos similarity')
    args_parser.add_argument('--sent_encoder_path', type=str, help='universal sentence encoder path for sent cos sim')
    args_parser.add_argument('--train_vocab', type=str, help='Training set vocab file (json) for train filter')
    args_parser.add_argument('--filters', type=str, default='word_sim:sent_sim:lm', help='filters for word substitution')
    args_parser.add_argument('--generators', type=str, default='synonym:sememe:embedding', help='generators for word substitution')
    args_parser.add_argument('--tagger', choices=['stanza', 'nltk', 'spacy', 'stanford'], default='stanza', help='POS tagger for POS checking in KNN embedding candidates')
    args_parser.add_argument('--use_pad', action='store_true', default=False, help='use PAD in input to attacker')
    args_parser.add_argument('--cached_path', type=str, default=None, help='input cached file for preprocessed candidate cache file')
    args_parser.add_argument('--cand_cache_path', type=str, default=None, help='output filename for candidate cache file')
    args_parser.add_argument('--ensemble', action='store_true', default=False, help='ensemble multiple parsers for predicting')
    args_parser.add_argument('--merge_by', type=str, choices=['logits', 'probs'], default='logits', help='ensemble policy')
    args = args_parser.parse_args()
    run(args)
