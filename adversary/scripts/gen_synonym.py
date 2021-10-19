import pickle
#import OpenHowNet
from nltk.corpus import wordnet as wn
from functools import partial
import spacy
nlp = spacy.load('en_core_web_sm')
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vocab", type=str, help="input vocab (json)")
parser.add_argument("output", type=str, help="output path for word candidates")
#parser.add_argument("--out_vocab", type=str, default="vocab.json", help="output path for vocab")
#parser.add_argument("--lemma", type=str, default="lemma.pkl", help="input path for lemma")
#parser.add_argument("--save_binary", action="store_true", help="whether save in binary")
args = parser.parse_args()

word_candidate = {}

with open(args.vocab, 'r') as fi:
	vocab = json.loads(fi.read().strip())
print ("Loading: vocab length: ", len(vocab))
inv_vocab = {i: w for (w, i) in vocab.items()}

s_ls = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
s_noun = ['NN', 'NNS', 'NNP', 'NNPS']
s_verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
s_adj = ['JJ', 'JJR', 'JJS']
s_adv = ['RB', 'RBR', 'RBS']


def _synonym_prefilter_fn(token, synonym):
  lemma = nlp(token)[0].lemma
  if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
          synonym.lemma == lemma) or (  # token and synonym are the same
          token.lower() == 'be')):  # token is be
      return False
  else:
      return True

def get_synonyms(token):
  wordnet_synonyms = []
  synsets = wn.synsets(token, pos=None)
  for synset in synsets:
      wordnet_synonyms.extend(synset.lemmas())
  #print ("synsets:\n", synsets)
  synonyms = []
  for wordnet_synonym in wordnet_synonyms:
      spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))#[0]
      if len(spacy_synonym) > 1: continue # the synonym produced is a phrase
      synonyms.append(spacy_synonym[0])
  #print ("synonyms:\n", synonyms)
  #tags = [s.tag_ for s in synonyms]
  #print ("tags:\n", [(s,t) for s,t in zip(synonyms, tags)])
  synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)
  syn_list = [s for s in synonyms]
  #print ("filtered synonyms:\n", syn_list)
  if len(syn_list) > 0:
  	word_candidate[token] = {}
  	for pos in s_ls:
  		word_candidate[token][pos] = []
  	for syn in syn_list:
  		tag = syn.tag_
  		#print ("tok:{}, cand:{}, tag:{}".format(token, syn.text, syn.tag_))
  		if tag in s_ls and syn.text not in word_candidate[token][tag]:
  			word_candidate[token][tag].append(syn.text)

for w1,i1 in vocab.items():
	if i1 % 1000 == 0:
		print(i1)
	get_synonyms(w1)

if args.output.endswith(".json"):
	with open(args.output, 'w') as fo:
		json.dump(word_candidate, fo, indent=4)
else:
	with open(args.output, 'wb') as fo:
		pickle.dump(word_candidate, fo)



