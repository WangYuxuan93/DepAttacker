import pickle
import argparse
import json

def load_conll(f):
	data = []
	sents = f.read().strip().split("\n\n")
	for sent in sents:
		data.append([line.strip().split("\t") for line in sent.strip().split("\n")])
	return data

def build_vocab(datas):
	word_cnt = {}
	for data in datas:
		for sent in data:
			for line in sent:
				word = line[1]
				if word not in word_cnt:
					word_cnt[word] = 1
				else:
					word_cnt[word] += 1
	word_cnt = sorted(word_cnt, key=word_cnt.get, reverse=True)
	vocab = {}
	i = 0
	for word in word_cnt:
		vocab[word] = i
		i += 1
	return vocab

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input conll file")
parser.add_argument("output", type=str, default="vocab.json", help="output path for vocab")
#parser.add_argument("--use_id", action="store_true", help="whether use id to store")
parser.add_argument("--save_binary", action="store_true", help="whether save in binary")
args = parser.parse_args()

files = args.input.split(":")
datas = []
for file in files:
	with open(file, 'r') as fi:
		datas.append(load_conll(fi))

vocab = build_vocab(datas)
print ("Saving: vocab length: ", len(vocab))
with open(args.output, 'w') as fo:
	fo.write(json.dumps(vocab, indent=4))
