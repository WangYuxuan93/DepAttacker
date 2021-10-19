import pickle
import argparse
import json

def load_conll(f):
	data = []
	sents = f.read().strip().split("\n\n")
	for sent in sents:
		data.append([line.strip().split("\t") for line in sent.strip().split("\n")])
	return data

def build_vocab(data):
	word_cnt = {}
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
parser.add_argument("--vocab", type=str, help="input vocab (json)")
parser.add_argument("--out_vocab", type=str, default="vocab.json", help="output path for vocab")
parser.add_argument("--out_lemma", type=str, default="lemma.pkl", help="output path for lemma")
#parser.add_argument("--use_id", action="store_true", help="whether use id to store")
parser.add_argument("--save_binary", action="store_true", help="whether save in binary")
args = parser.parse_args()


with open(args.input, 'r') as fi:
	data = load_conll(fi)
if args.vocab is None:
	vocab = build_vocab(data)
	print ("Saving: vocab length: ", len(vocab))
	with open(args.out_vocab, 'w') as fo:
		fo.write(json.dumps(vocab, indent=4))
else:
	with open(args.vocab, 'r') as fi:
		vocab = json.loads(fi.read().strip())
	print ("Loading: vocab length: ", len(vocab))

dict = {w: i for (w, i) in vocab.items()}
inv_dict = {i: w for (w, i) in dict.items()}

word_candidate={}
trains = [[line[1] for line in sent] for sent in data]

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

#if args.use_id:
#	train_text = [[inv_dict[t] for t in tt] for tt in trains]
#else:
#	train_text = trains

NNS={}
NNPS={}
JJR={}
JJS={}
RBR={}
RBS={}
VBD={}
VBG={}
VBN={}
VBP={}
VBZ={}
inv_NNS={}
inv_NNPS={}
inv_JJR={}
inv_JJS={}
inv_RBR={}
inv_RBS={}
inv_VBD={}
inv_VBG={}
inv_VBN={}
inv_VBP={}
inv_VBZ={}
s_ls=['NNS','NNPS','JJR','JJS','RBR','RBS','VBD','VBG','VBN','VBP','VBZ']
s_noun=['NNS','NNPS']
s_verb=['VBD','VBG','VBN','VBP','VBZ']
s_adj=['JJR','JJS']
s_adv=['RBR','RBS']

word_pos_pairs = [[(line[1],line[3]) for line in sent] for sent in data]
for idx in range(len(word_pos_pairs)):
	if idx % 1000 == 0:
		print(idx)
	#text=train_text[idx]
	pos_tags = word_pos_pairs[idx]
	for i in range(len(pos_tags)):
		pair=pos_tags[i]
		if pair[1] in s_ls:
			if pair[1][:2]=='NN':
				w=wnl.lemmatize(pair[0],pos='n')
			elif pair[1][:2]=='VB':
				w = wnl.lemmatize(pair[0], pos='v')
			elif pair[1][:2]=='JJ':
				w = wnl.lemmatize(pair[0], pos='a')
			else:
				w = wnl.lemmatize(pair[0], pos='r')
			eval('inv_'+pair[1])[w]=pair[0]
			eval(pair[1])[pair[0]]=w
f=open(args.out_lemma,'wb')
pickle.dump((NNS,NNPS,JJR,JJS,RBR,RBS,VBD,VBG,VBN,VBP,VBZ,inv_NNS,inv_NNPS,inv_JJR,inv_JJS,inv_RBR,inv_RBS,inv_VBD,inv_VBG,inv_VBN,inv_VBP,inv_VBZ),f)
