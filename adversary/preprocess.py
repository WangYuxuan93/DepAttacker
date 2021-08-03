# coding=utf-8
import argparse
import nltk
from nltk.corpus import wordnet as wn
import spacy
nlp = spacy.load('en_core_web_sm')
import language_check
import sys
import codecs
import stanza

from functools import partial
import numpy as np
import torch
import random
import os
import json
import pickle
import string

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

try:
    import tensorflow_hub as hub
    import tensorflow as tf
except:
    print ("Can not import tensorflow_hub!")
try:
    from allennlp.modules.elmo import batch_to_ids
except:
    print ("can not import batch_to_ids!")

from neuronlp2.io import get_logger
from neuronlp2.io.common import PAD, ROOT, END
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io import common
from adversary.lm.bert import Bert
#from adversary.attack import convert_tokens_to_ids
from neuronlp2.io.common import DIGIT_RE

#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

stopwords = set(
        [
            "'s",
            "'re",
            "'ve",
            "a",
            "about",
            "above",
            "across",
            "after",
            "afterwards",
            "again",
            "against",
            "ain",
            "all",
            "almost",
            "alone",
            "along",
            "already",
            "also",
            "although",
            "am",
            "among",
            "amongst",
            "an",
            "and",
            "another",
            "any",
            "anyhow",
            "anyone",
            "anything",
            "anyway",
            "anywhere",
            "are",
            "aren",
            "aren't",
            "around",
            "as",
            "at",
            "be",
            "back",
            "been",
            "before",
            "beforehand",
            "behind",
            "being",
            "below",
            "beside",
            "besides",
            "between",
            "beyond",
            "both",
            "but",
            "by",
            "can",
            "cannot",
            "could",
            "couldn",
            "couldn't",
            "d",
            "do",
            "did",
            "didn",
            "didn't",
            "does",
            "doesn",
            "doesn't",
            "don",
            "don't",
            "down",
            "due",
            "during",
            "either",
            "else",
            "elsewhere",
            "empty",
            "enough",
            "even",
            "ever",
            "everyone",
            "everything",
            "everywhere",
            "except",
            "first",
            "for",
            "former",
            "formerly",
            "from",
            "had",
            "hadn",
            "hadn't",
            "has",
            "hasn",
            "hasn't",
            "have",
            "haven",
            "haven't",
            "he",
            "hence",
            "her",
            "here",
            "hereafter",
            "hereby",
            "herein",
            "hereupon",
            "hers",
            "herself",
            "him",
            "himself",
            "his",
            "how",
            "however",
            "hundred",
            "i",
            "if",
            "in",
            "indeed",
            "into",
            "is",
            "isn",
            "isn't",
            "it",
            "it's",
            "its",
            "itself",
            "just",
            "latter",
            "latterly",
            "least",
            "ll",
            "may",
            "me",
            "meanwhile",
            "mightn",
            "mightn't",
            "mine",
            "more",
            "moreover",
            "most",
            "mostly",
            "must",
            "mustn",
            "mustn't",
            "my",
            "myself",
            "n't",
            "namely",
            "needn",
            "needn't",
            "neither",
            "never",
            "nevertheless",
            "next",
            "no",
            "nobody",
            "none",
            "noone",
            "nor",
            "not",
            "nothing",
            "now",
            "nowhere",
            "o",
            "of",
            "off",
            "on",
            "once",
            "one",
            "only",
            "onto",
            "or",
            "other",
            "others",
            "otherwise",
            "our",
            "ours",
            "ourselves",
            "out",
            "over",
            "per",
            "please",
            "s",
            "same",
            "shan",
            "shan't",
            "she",
            "she's",
            "should've",
            "shouldn",
            "shouldn't",
            "somehow",
            "something",
            "sometime",
            "somewhere",
            "such",
            "t",
            "than",
            "that",
            "that'll",
            "the",
            "their",
            "theirs",
            "them",
            "themselves",
            "then",
            "thence",
            "there",
            "thereafter",
            "thereby",
            "therefore",
            "therein",
            "thereupon",
            "these",
            "they",
            "this",
            "those",
            "through",
            "throughout",
            "thru",
            "thus",
            "to",
            "too",
            "toward",
            "towards",
            "under",
            "unless",
            "until",
            "up",
            "upon",
            "used",
            "ve",
            "was",
            "wasn",
            "wasn't",
            "we",
            "were",
            "weren",
            "weren't",
            "what",
            "whatever",
            "when",
            "whence",
            "whenever",
            "where",
            "whereafter",
            "whereas",
            "whereby",
            "wherein",
            "whereupon",
            "wherever",
            "whether",
            "which",
            "while",
            "whither",
            "who",
            "whoever",
            "whole",
            "whom",
            "whose",
            "why",
            "with",
            "within",
            "without",
            "won",
            "won't",
            "would",
            "wouldn",
            "wouldn't",
            "y",
            "yet",
            "you",
            "you'd",
            "you'll",
            "you're",
            "you've",
            "your",
            "yours",
            "yourself",
            "yourselves",
        ]
    )

def recover_word_case(word, reference_word):
    """ Makes the case of `word` like the case of `reference_word`. Supports 
        lowercase, UPPERCASE, and Capitalized. """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word

class Preprocessor(object):
    def __init__(self, candidates, vocab, synonyms,
                filter_every_n_sents=1,
                generators=['synonym', 'sememe', 'embedding'],
                tagger="stanford", punct_set=[], 
                cached_path=None, 
                knn_path=None, max_knn_candidates=50, 
                cand_mlm=None, temperature=1.0, top_k=100, top_p=None, 
                n_mlm_cands=50, mlm_cand_file=None, 
                device=None, symbolic_root=True, symbolic_end=False, mask_out_root=False, 
                batch_size=32):
        super(Preprocessor, self).__init__()
        logger = get_logger("Attacker")
        logger.info("##### Attacker Type: {} #####".format(self.__class__.__name__))

        self.candidates = candidates
        self.synonyms = synonyms
        self.word2id = vocab
        self.generators = generators
        self.tagger = tagger
        self.punct_set = punct_set

        self.cached_path = cached_path
        self.filter_every_n_sents = filter_every_n_sents

        logger.info("Generators: {}".format(generators))
        logger.info("POS tagger: {}".format(tagger))
        logger.info("Filter every {} sents".format(filter_every_n_sents))
        if cached_path is not None:
            logger.info("Loading cached candidates from: %s" % cached_path)
            self.cached_cands = json.load(open(cached_path, 'r', encoding="utf-8"))
        else:
            self.cached_cands = None
        if self.tagger == 'stanza':
            self.stanza_tagger = stanza.Pipeline(lang='en', processors='tokenize,pos', 
                                                    tokenize_pretokenized=True, use_gpu=True)
        elif self.tagger == 'stanford':
            jar = 'stanford-postagger-2018-10-16/stanford-postagger.jar'
            model = 'stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger'
            logger.info("Loading stanford tagger from: %s" % model)
            self.stanford_tagger = nltk.tag.StanfordPOSTagger(model, jar, encoding='utf8')
        self.id2word = {i:w for (w,i) in vocab.items()}

        if self.cached_cands is None and 'embedding' in self.generators:
            logger.info("Loading knn from: {}".format(knn_path))
            self.load_knn_path(knn_path)
        else:
            self.nn = None

        if 'mlm' in self.generators:
            self.n_mlm_cands = n_mlm_cands
            logger.info("Loading MLM generator from: {}".format(cand_mlm))
            self.mlm_cand_model = Bert(cand_mlm, device=device, temperature=temperature, top_k=top_k, top_p=top_p)
            #print ("BERT:cand_mlm={}\ntemp={},top_k={},top_p={}".format(cand_mlm, temperature, top_k, top_p))
            self.mlm_cand_model.model.eval() 
        else:
            self.mlm_cand_model = None
            self.n_mlm_cands = None
        
        self.device = device
        self.symbolic_root = symbolic_root
        self.symbolic_end = symbolic_end
        self.mask_out_root = mask_out_root
        self.batch_size = batch_size
        #self.stop_words = nltk.corpus.stopwords.words('english')
        #if 'stop_words' in self.filters:
        #    logger.info("Init stop word list.")
        #    self.stop_words = stopwords
        #else:
        logger.info("Empty stop word list.")
        self.stop_words = []
        self.stop_tags = ['PRP','PRP$','DT','CC','IN','CD','UH','WDT','WP','WP$','-LRB-','-RRB-','.','``',"\'\'",':',',','?',';']

        self.max_knn_candidates = max_knn_candidates
        
        if self.cached_cands is None and 'embedding' in self.generators and self.nn is None:
            print ("Must input embedding path for embedding generator!")
            exit()
        if self.cached_cands is None and 'mlm' in self.generators and self.mlm_cand_model is None:
            print ("Must input bert path or cached mlm cands for mlm generator!")
            exit()

    def load_knn_path(self, path):
        word_embeddings_file = "paragram.npy"
        word_list_file = "wordlist.pickle"
        nn_matrix_file = "nn.npy"
        cos_sim_file = "cos_sim.p"
        word_embeddings_file = os.path.join(path, word_embeddings_file)
        word_list_file = os.path.join(path, word_list_file)
        nn_matrix_file = os.path.join(path, nn_matrix_file)
        cos_sim_file = os.path.join(path, cos_sim_file)

        self.word_embeddings = np.load(word_embeddings_file)
        self.word_embedding_word2index = np.load(word_list_file, allow_pickle=True)
        self.nn = np.load(nn_matrix_file)
        with open(cos_sim_file, "rb") as f:
            self.cos_sim_mat = pickle.load(f)

        # Build glove dict and index.
        self.word_embedding_index2word = {}
        for word, index in self.word_embedding_word2index.items():
            self.word_embedding_index2word[index] = word

    def _get_knn_words(self, word):
        """ Returns a list of possible 'candidate words' to replace a word in a sentence 
            or phrase. Based on nearest neighbors selected word embeddings.
        """
        try:
            word_id = self.word_embedding_word2index[word.lower()]
            nnids = self.nn[word_id][1 : self.max_knn_candidates + 1]
            candidate_words = []
            for i, nbr_id in enumerate(nnids):
                nbr_word = self.word_embedding_index2word[nbr_id]
                #candidate_words.append(recover_word_case(nbr_word, word))
                candidate_words.append(nbr_word)
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []

    def get_word_cos_sim(self, a, b):
        """ Returns the cosine similarity of words with IDs a and b."""
        if a not in self.word_embedding_word2index or b not in self.word_embedding_word2index:
            return None
        if isinstance(a, str):
            a = self.word_embedding_word2index[a]
        if isinstance(b, str):
            b = self.word_embedding_word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            cos_sim = self.cos_sim_mat[a][b]
        except KeyError:
            e1 = self.word_embeddings[a]
            e2 = self.word_embeddings[b]
            e1 = torch.tensor(e1)
            e2 = torch.tensor(e2)
            cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2).numpy()
            self.cos_sim_mat[a][b] = cos_sim
        return cos_sim

    def _word2id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return None

    def _id2word(self, id):
        if id in self.id2word:
            return self.id2word[id]
        else:
            return None

    def get_punct_mask(self, tokens, tags):
        assert len(tokens) == len(tags)
        punct_mask = np.ones(len(tokens))
        for i, tag in enumerate(tags):
            if tag in self.punct_set:
                punct_mask[i] = 0
        return punct_mask

    def get_batch(self, input_ids):
        # (cand_size+1, seq_len)
        data_size = input_ids.size()[0]
        for start_idx in range(0, data_size, self.batch_size):
            excerpt = slice(start_idx, start_idx + self.batch_size)
            yield input_ids[excerpt, :]

    def cos_sim(self, e1, e2):
        e1 = torch.tensor(e1)
        e2 = torch.tensor(e2)
        cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2)
        return cos_sim.numpy()

    def get_syn_cands(self, token, tag):
        if token not in self.synonyms:
            return []
        if token in self.stop_words:
            return []
        if tag in self.synonyms[token]:
            return self.synonyms[token][tag]
        else:
            return []

    def get_sem_cands(self, token, tag):
        tag_list = ['JJ', 'NN', 'RB', 'VB']
        if self._word2id(token) not in range(1, 50000):
            return []
        if token in self.stop_words:
            return []
        if tag[:2] not in tag_list:
            return []
        if tag[:2] == 'JJ':
            pos = 'adj'
        elif tag[:2] == 'NN':
            pos = 'noun'
        elif tag[:2] == 'RB':
            pos = 'adv'
        else:
            pos = 'verb'
        if pos in self.candidates[self._word2id(token)]:
            return [self._id2word(neighbor) for neighbor in self.candidates[self._word2id(token)][pos]]
        else:
            return []

    def get_emb_cands(self, tokens, tag, idx):
        return self._get_knn_words(tokens[idx])

    def get_mlm_cands(self, tokens, tag, idx, sent_id=None):
        return self._get_mlm_cands(tokens, idx, n=self.n_mlm_cands, sent_id=sent_id)

    def _get_mlm_cands(self, tokens, idx, n=50, sent_id=None):
        original_word = tokens[idx]
        tmps = tokens.copy()
        tmps[idx] = self.mlm_cand_model.MASK_TOKEN
        masked_text = ' '.join(tmps)
        #print ("masked_text:", masked_text)
        candidates = self.mlm_cand_model.predict(masked_text, target_word=original_word, n=n)

        return [candidate[0] for candidate in candidates]

    def update_cand_set(self, token, cand_set, cands, lower_set):
        for c in cands:
            #if c.lower() not in lower_set and c.lower() != token.lower():
            if c.lower() != token.lower():
                cand_set.append(c)
                #cand_set.append(c)
                #lower_set.add(c.lower())
        return cand_set#, lower_set
    """
    def _get_candidate_set_from_cache(self, tokens, tag, idx, sent_id=None):
        token = tokens[idx]
        if token.lower() in self.stop_words:
            return []
        if tag in self.stop_tags:
            return []
        if token == PAD or token == ROOT:
            return []
        candidate_set = []
        lower_set = set()
        name_map = {'sememe':'sem_cands', 'synonym':'syn_cands', 'embedding':'emb_cands',
                    'mlm':'mlm_cands'}

        cands = self.cached_cands[sent_id]['tokens'][idx]
        assert cands['token'] == tokens[idx]
        for name in name_map:
            if name in self.generators:
                # generate mlm cands dynamically
                if name == 'mlm' and self.dynamic_mlm_cand:
                    mlm_cands = self.get_mlm_cands(tokens.copy(), tag, idx, sent_id=sent_id)
                    self.update_cand_set(token, candidate_set, mlm_cands, lower_set)
                else:
                    self.update_cand_set(token, candidate_set, cands[name_map[name]], lower_set)
        return candidate_set

        def get_candidate_set(self, tokens, tag, idx, sent_id=None, cache=False):
        if self.cached_cands is not None:
            cand_set = self._get_candidate_set_from_cache(tokens, tag, idx, sent_id=sent_id)
            cache_data = None
        else:
            cand_set, cache_data = self._get_candidate_set(tokens, tag, idx, sent_id=sent_id, cache=cache)

        return cand_set, cache_data
    """

    def post_process(self, tokens, cands, tag, idx):
        # replace with cased version, same format as original token
        cased_cands = [recover_word_case(c, tokens[idx]) for c in cands]
        return self.pos_filter(tokens, cased_cands, tag, idx)

    def pos_filter(self, tokens, cands, tag, idx):
        if not cands:
            return []
        filtered_cands = []
        tmps = tokens.copy()
        #print ("token={}, tag={}".format(tokens[idx], tag))
        cand_sents = [tokens[:idx]+[c]+tokens[idx+1:] for c in cands]
        if self.tagger == 'stanza':
            sent_tags = self.stanza_tagger(cand_sents).sentences
            cand_tags = [x.words[idx].xpos for x in sent_tags]
        elif self.tagger == 'nltk':
            #cand_tag = nltk.pos_tag([cand.lower()])[0][1]
            cand_tags = [nltk.pos_tag(x)[idx][1] for x in cand_sents]
        elif self.tagger == 'spacy':
            #cand_tag = nlp(cand.lower())[0].tag_
            cand_tag = [nlp(' '.join(x))[idx].tag_ for x in cand_sents]
        elif self.tagger == 'stanford':
            sent_tags = self.stanford_tagger.tag_sents(cand_sents)
            cand_tags = [x[idx][1] for x in sent_tags]

        for cand_tag, cand in zip(cand_tags, cands):
            tmps[idx] = cand
            if cand_tag == tag:
                filtered_cands.append(cand)
            
        return filtered_cands

    def _get_candidate_set(self, tokens, tag, idx, sent_id=None):
        token = tokens[idx]
        cache_data = {'sem_cands':[], 'syn_cands':[], 'emb_cands':[],
                          'mlm_cands':[]}
        if token.lower() in self.stop_words:
            return cache_data
        if tag in self.stop_tags:
            return cache_data
        if token == PAD or token == ROOT:
            return cache_data
        #candidate_set = []
        lower_set = set()
        #print ("origin token: ", token)
        if 'sememe' in self.generators:
            sem_cands = self.get_sem_cands(token, tag)
            sem_cands = [recover_word_case(c, tokens[idx]) for c in sem_cands]
            #sem_cands = self.post_process(tokens.copy(), sem_cands, tag, idx)
            #self.update_cand_set(token, candidate_set, sem_cands, lower_set)
            #print ("sememe:", sem_cands)
        else:
            sem_cands = []
        if 'synonym' in self.generators:
            syn_cands = self.get_syn_cands(token, tag)
            syn_cands = [recover_word_case(c, tokens[idx]) for c in syn_cands]
            #self.update_cand_set(token, candidate_set, syn_cands, lower_set)
            #print ("syn:", syn_cands)
        else:
            syn_cands = []
        if 'embedding' in self.generators:
            emb_cands = self.get_emb_cands(tokens.copy(), tag, idx)
            emb_cands = [recover_word_case(c, tokens[idx]) for c in emb_cands]
            #self.update_cand_set(token, candidate_set, emb_cands, lower_set)
            #print ("knn cands:\n", emb_cands)
        else:
            emb_cands = []
        if 'mlm' in self.generators:
            mlm_cands = self.get_mlm_cands(tokens.copy(), tag, idx, sent_id=sent_id)
            #print ("token:{}, mlm_cands:{}".format(tokens[idx],mlm_cands))
            mlm_cands = [recover_word_case(c, tokens[idx]) for c in mlm_cands]
            #self.update_cand_set(token, candidate_set, mlm_cands, lower_set)
        else:
            mlm_cands = []

        cache_data = {'sem_cands':sem_cands, 'syn_cands':syn_cands, 'emb_cands':emb_cands,
                      'mlm_cands':mlm_cands}
            
        return cache_data


    def get_cache_for_sentence(self, tokens, tags, sent_id=None, debug=False):
        """
        Input:
            tokens: List[str], (seq_len)
            tags: List[str], (seq_len)
            heads: List[int], (seq_len)
            rel_ids: List[int], (seq_len)
        Output:
        """
        adv_tokens = tokens.copy()
        punct_mask = self.get_punct_mask(tokens, tags)

        x_len = len(tokens)
        tag_list = ['JJ', 'NN', 'RB', 'VB']
        cand_cache = []
        #stop_words = nltk.corpus.stopwords.words('english')
        for i in range(x_len):
            #print (adv_tokens[i], self._word2id(adv_tokens[i]))
            cache_data = self._get_candidate_set(adv_tokens, tags[i], i, sent_id=sent_id)
            cache_data['id'] = i
            cache_data['token'] = tokens[i]
            cand_cache.append(cache_data)

        return cand_cache

    def filter_cache_with_pos(self, tmp_cache, tmp_tags, debug=False):
        cand_types = ['sem_cands','syn_cands','emb_cands','mlm_cands']
        cand_sents = []
        # gather candidate sentence list for stanford tagger
        # for each sentence
        for cache in tmp_cache:
            tokens = [x['token'] for x in cache]
            # for each token
            for token_cache in cache:
                idx = token_cache['id']
                # for each cand type
                for type in cand_types:
                    tmp_sents = [tokens[:idx]+[c]+tokens[idx+1:] for c in token_cache[type]]
                    cand_sents.extend(tmp_sents)

        # apply stanford tagger
        sent_tags = self.stanford_tagger.tag_sents(cand_sents)
        offset = 0
        new_cache = []
        for cache, tags in zip(tmp_cache, tmp_tags):
            tokens = [x['token'] for x in cache]
            new_sent_cache = []
            # for each token
            for token_cache in cache:
                idx = token_cache['id']
                new_token_cache = {}
                # for each cand type
                for type in cand_types:
                    cand_tokens = token_cache[type]
                    #print (sent_tags[offset:offset+len(cand_tokens)])
                    cand_tags = [x[idx][1] for x in sent_tags[offset:offset+len(cand_tokens)]]
                    if debug:
                        print ("idx={},tag={},type={}".format(idx, tags[idx],type))
                        print ("cand_tags:", cand_tags)
                        print ("cand_tokens:", cand_tokens)
                    new_token_cache[type] = []
                    for token, cand_tag in zip(cand_tokens, cand_tags):
                        if cand_tag == tags[idx]:
                            new_token_cache[type].append(token)
                    offset += len(cand_tokens)
                new_token_cache['id'] = idx
                new_token_cache['token'] = token_cache['token']
                new_sent_cache.append(new_token_cache)
            new_cache.append(new_sent_cache)
        return new_cache

    def get_cache(self, sents, debug=False):
        """
        Input:
            sents: List[(tokens, tags)]
        Output:
        """
        cache = []
        tmp_cache = []
        tmp_tags = []
        start = 0
        for sent_id, (tokens, tags) in enumerate(sents):
            sent_cache = self.get_cache_for_sentence(tokens, tags, sent_id)
            tmp_cache.append(sent_cache)
            tmp_tags.append(tags)
            if sent_id % self.filter_every_n_sents == 0:
                if sent_id % (100*self.filter_every_n_sents) == 0:
                    print ("%d.."%sent_id, end="")
                    sys.stdout.flush()
                filtered_cache = self.filter_cache_with_pos(tmp_cache, tmp_tags)
                tmp_cache = []
                tmp_tags = []
                for c in filtered_cache:
                    cache.append({'sent_id':len(cache), 'tokens': c})
            
        if tmp_cache:
            filtered_cache = self.filter_cache_with_pos(tmp_cache, tmp_tags)
            tmp_cache = []
            tmp_tags = []
            for c in filtered_cache:
                cache.append({'sent_id':len(cache), 'tokens': c})
        print ("")
        return cache

def read_conll(filename):
    sents = []
    with open(filename, 'r') as f:
        data = f.read().strip().split("\n\n")
        for sent in data:
            lines = sent.strip().split("\n")
            tokens = [ROOT]+[x.strip().split('\t')[1] for x in lines]
            tags = [ROOT]+[x.strip().split('\t')[3] for x in lines]
            sents.append((tokens, tags))
    return sents


def main(args):
    print ("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    punct_set = set(args.punctuation)
    if args.cand.endswith('.json'):
        cands = json.load(open(args.cand, 'r'))
        candidates = {int(i):dic for (i,dic) in cands.items()}
    else:
        candidates = pickle.load(open(args.cand, 'rb'))
        vocab = json.load(open(args.vocab, 'r'))
        synonyms = json.load(open(args.syn, 'r'))
    generators = args.generators.split(":")
    #logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))
    processor = Preprocessor(candidates, vocab, synonyms, generators=generators,
                        tagger=args.tagger,
                        punct_set=punct_set,
                        cached_path=args.cached_path, 
                        knn_path=args.knn_path, 
                        max_knn_candidates=args.max_knn_candidates,
                        cand_mlm=args.cand_mlm, temperature=args.temp, 
                        top_k=args.top_k, top_p=args.top_p, 
                        n_mlm_cands=args.n_mlm_cands, mlm_cand_file=args.mlm_cand_file,
                        device=device,
                        batch_size=args.adv_batch_size,
                        filter_every_n_sents=args.filter_every_n_sents)
    sents = read_conll(args.test)
    print ("Total {} sents".format(len(sents)))
    cache = processor.get_cache(sents)
    with open(args.cand_cache_path, 'w') as cache_f:
        json.dump(cache, cache_f, indent=4)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--seed', type=int, default=666, help='Random seed for torch and numpy (-1 for random)')
    args_parser.add_argument('--config', type=str, help='config file')
    args_parser.add_argument('--vocab', type=str, help='vocab file for attacker')
    args_parser.add_argument('--cand', type=str, help='candidate file for attacker')
    args_parser.add_argument('--syn', type=str, help='synonym file for attacker')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    args_parser.add_argument('--eval_batch_size', type=int, default=256, help='Number of sentences in each batch while evaluating')

    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--cand_mlm', help='path for mlm candidate generating')
    args_parser.add_argument('--mlm_cand_file', help='path for mlm candidate preprocessed json file')
    args_parser.add_argument('--temp', type=float, default=1.0, help='Temperature for mlm candidate generating')
    args_parser.add_argument('--n_mlm_cands', type=int, default=50, help='Select candidate number for mlm candidate generating')
    args_parser.add_argument('--top_k', type=int, default=100, help='Top candidate number for filtering mlm candidate generating')
    args_parser.add_argument('--top_p', type=float, default=None, help='Top proportion for filtering mlm candidate generating')
    args_parser.add_argument('--output_filename', type=str, help='output filename for parse')
    args_parser.add_argument('--adv_filename', type=str, help='output adversarial filename')
    args_parser.add_argument('--adv_gold_filename', type=str, help='output adversarial text with gold heads & rels')
    args_parser.add_argument('--adv_batch_size', type=int, default=16, help='Number of sentences in adv lm each batch')
    args_parser.add_argument('--knn_path', type=str, help='knn embedding path for adversarial attack')
    args_parser.add_argument('--max_knn_candidates', type=int, default=50, help='max knn candidate number')
    args_parser.add_argument('--generators', type=str, default='synonym:sememe:embedding', help='generators for word substitution')
    args_parser.add_argument('--tagger', choices=['stanza', 'nltk', 'spacy', 'stanford'], default='stanza', help='POS tagger for POS checking in KNN embedding candidates')
    args_parser.add_argument('--cached_path', type=str, default=None, help='input cached file for preprocessed candidate cache file')
    args_parser.add_argument('--cand_cache_path', type=str, default=None, help='output filename for candidate cache file')
    args_parser.add_argument('--filter_every_n_sents', type=int, default=1, help='filter n sents every batch')
    args = args_parser.parse_args()
    main(args)
