import pickle
import argparse
import json
from collections import OrderedDict
#import numpy as np
import random

def load_conll(f):
	data = []
	sents = f.read().strip().split("\n\n")
	for sent in sents:
		data.append([line.strip().split("\t") for line in sent.strip().split("\n")])
	return data

def check_root(data):
	n_multi, n_no = 0, 0
	n_lab = 0
	for sent in data:
		num_root = 0
		for line in sent:
			if line[6] == '0':
				if line[7] != 'root':
					n_lab += 1
				num_root += 1
		if num_root == 0:
			n_no += 1
		elif num_root > 1:
			n_multi += 1
	print ("Multi root={}, No root={}, Wrong label={}".format(n_multi, n_no, n_lab))

def eval_sent(gold, pred, rm_punc=True):
	punc = ['.', '``', "''", ':', ',']
	assert len(gold) == len(pred)
	l_cor, u_cor = 0, 0
	tot = 0
	for glines, plines in zip(gold,pred):
		if rm_punc and glines[3] in punc:
			continue
		tot += 1
		if glines[6] == plines[6]:
			u_cor += 1
			if glines[7] == plines[7]:
				l_cor += 1
	return u_cor, l_cor, tot

def per_num(sent):
	n = 0
	for line in sent:
		if line[9] != '_':
			n += 1
	return n

def search(gold, orig, adv, rm_punc=True):
	output = []
	n_tot = 0
	n_unlab_succ, n_lab_succ = 0, 0
	n_orig_ucor, n_orig_lcor = 0, 0
	n_adv_ucor, n_adv_lcor = 0, 0
	print (len(adv), len(orig))
	for gsent, osent, asent in zip(gold, orig, adv):
		u1, l1, k1 = eval_sent(gsent, osent, rm_punc)
		u2, l2, k2 = eval_sent(gsent, asent, rm_punc)
		if u1 > u2:
			n_unlab_succ += 1
		if u1 > u2 or l1 > l2:
			n_lab_succ += 1
		n_orig_ucor += u1
		n_orig_lcor += l1
		n_adv_ucor += u2
		n_adv_lcor += l2
		n_tot += k1
	unlab_succ = float(n_unlab_succ) / len(gold)
	lab_succ = float(n_lab_succ) / len(gold)
	orig_uas = float(n_orig_ucor) / n_tot
	orig_las = float(n_orig_lcor) / n_tot
	adv_uas = float(n_adv_ucor) / n_tot
	adv_las = float(n_adv_lcor) / n_tot
	print ("total sents:%d, tokens:%d, unlabel succ: %d, label succ: %d"%(len(gold), n_tot, n_unlab_succ, n_lab_succ))
	print ("unlabel succ={:.2f}%, label succ rate={:.2f}%".format(unlab_succ*100, lab_succ*100))
	print ("Original UAS={:.2f}%, LAS={:.2f}%".format(orig_uas*100, orig_las*100))
	print ("Adversarial UAS={:.2f}%, LAS={:.2f}%".format(adv_uas*100, adv_las*100))
	return output

def output(data, fo):
	for gsent, osent, asent, dif, orig_dif in data:
		fo.write('err inc=%d, orig diff=%d\n'%(dif, orig_dif))
		sent_str = []
		for l in asent:
			sent_str.append(l[1])
			if l[9] != '_':
				sent_str.append(l[9])
		fo.write('adv sent: %s\n' % ' '.join(sent_str))
		for gline, oline, aline in zip(gsent, osent, asent):
			str = '\t'.join(gline)
			str += '\n'+'\t'.join(oline)
			str += '\n'+'\t'.join(aline)
			fo.write(str+'\n')
		fo.write('\n')

parser = argparse.ArgumentParser()
parser.add_argument("--orig", type=str, help="input orig conll file")
parser.add_argument("--adv", type=str, help="input adv conll file")
parser.add_argument("--gold", type=str, help="input gold conll file")
parser.add_argument("--p", action='store_true', default=False, help="do not count punc")
args = parser.parse_args()

with open(args.orig, 'r') as fi:
	origs = load_conll(fi)
print ("Check root for %s" % args.orig)
check_root(origs)

with open(args.adv, 'r') as fi:
	advs = load_conll(fi)
print ("Check root for %s" % args.adv)
check_root(advs)

with open(args.gold, 'r') as fi:
	golds = load_conll(fi)

out = search(golds, origs, advs, rm_punc=args.p)

