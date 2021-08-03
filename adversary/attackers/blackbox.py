# coding=utf-8
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

import io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
ISPRINT=False
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

class BlackBoxAttacker(object):
    def __init__(self, model, candidates, vocab, synonyms, filters=['word_sim', 'sent_sim', 'lm', 'train'],
                generators=['synonym', 'sememe', 'embedding'], max_mod_percent=0.05, 
                tagger="nltk", ensemble=False, punct_set=[], beam=1, normalize_digits=False,
                cached_path=None, train_vocab=None,
                knn_path=None, max_knn_candidates=50, sent_encoder_path=None,
                min_word_cos_sim=0.8, min_sent_cos_sim=0.8, 
                cand_mlm=None, dynamic_mlm_cand=False, temperature=1.0, top_k=100, top_p=None, 
                n_mlm_cands=50, mlm_cand_file=None,
                adv_lms=None, rel_ratio=0.5, fluency_ratio=0.2,
                ppl_inc_thres=20 ,alphabets=None, tokenizer=None, 
                device=None, lm_device=None, symbolic_root=True, symbolic_end=False, mask_out_root=False, 
                batch_size=32, random_backoff=False, wordpiece_backoff=False):
        super(BlackBoxAttacker, self).__init__()
        logger = get_logger("Attacker")
        logger.info("##### Attacker Type: {} #####".format(self.__class__.__name__))
        self.model = model
        self.ensemble = ensemble
        self.candidates = candidates
        self.synonyms = synonyms
        self.word2id = vocab
        self.filters = filters
        self.generators = generators
        self.tagger = tagger
        self.punct_set = punct_set
        self.max_mod_percent = max_mod_percent
        self.cached_path = cached_path
        self.dynamic_mlm_cand = dynamic_mlm_cand
        self.beam = beam
        self.normalize_digits = normalize_digits
        logger.info("Filters: {}".format(filters))
        logger.info("Generators: {}".format(generators))
        logger.info("Max modification percentage: {}".format(max_mod_percent))
        logger.info("POS tagger: {}".format(tagger))
        logger.info("Normalize digits: {}".format(normalize_digits))
        if cached_path is not None:
            logger.info("Loading cached candidates from: %s" % cached_path)
            self.cached_cands = json.load(open(cached_path, 'r', encoding="utf-8"))
        else:
            self.cached_cands = None
            if self.tagger == 'stanza':
                self.stanza_tagger = stanza.Pipeline(lang='en', processors='tokenize,pos', 
                                                        tokenize_pretokenized=True, use_gpu=True)
            elif self.tagger == 'stanford':
                jar = '/users2/yxwang/work/experiments/adv/stanford-postagger-2018-10-16/stanford-postagger.jar'
                model = '/users2/yxwang/work/experiments/adv/stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger'
                logger.info("Loading stanford tagger from: %s" % model)
                self.stanford_tagger = nltk.tag.StanfordPOSTagger(model, jar, encoding='utf8')
        self.id2word = {i:w for (w,i) in vocab.items()}
        if 'train' in self.filters and train_vocab is not None:
            logger.info("Loading train vocab for filter from: %s" % train_vocab)
            self.train_vocab = json.load(open(train_vocab, 'r',encoding="utf-8"))
        else:
            self.train_vocab = None
        if ('word_sim' in self.filters or (self.cached_cands is None and 'embedding' in self.generators)) and knn_path is not None:
            logger.info("Loading knn from: {}".format(knn_path))
            self.load_knn_path(knn_path)
            logger.info("Min word cosine similarity: {}".format(min_word_cos_sim))
        else:
            self.nn = None
        if 'sent_sim' in self.filters and sent_encoder_path is not None:
            logger.info("Loading sent encoder from: {}".format(sent_encoder_path))
            self.sent_encoder = hub.load(sent_encoder_path)
            logger.info("Min sent cosine similarity: {}".format(min_sent_cos_sim))
        else:
            self.sent_encoder = None
        if 'lm' in self.filters and adv_lms is not None:
            self.adv_tokenizer, self.adv_lm = adv_lms
            #logger.info("Min per ppl increase: {}".format(ppl_inc_thres))
        else:
            self.adv_tokenizer, self.adv_lm = None, None
        if 'grammar' in self.filters:
            self.grammar_checker = language_check.LanguageTool('en-US')
        else:
            self.grammar_checker = None

        if 'mlm' in self.generators and (self.dynamic_mlm_cand or self.cached_cands is None):
            self.n_mlm_cands = n_mlm_cands
            if mlm_cand_file is not None and not self.dynamic_mlm_cand:
                self.mlm_cand_dict = json.load(open(mlm_cand_file, 'r',encoding="utf-8"))
                logger.info("Loading MLM candidates from: {} ({} sentences)".format(mlm_cand_file, len(self.mlm_cand_dict)))
                self.mlm_cand_model = None
            elif cand_mlm is not None:
                logger.info("Loading MLM generator from: {}".format(cand_mlm))
                logger.info("Generate MLM dynamically: {}".format(self.dynamic_mlm_cand))
                self.mlm_cand_model = Bert(cand_mlm, device=device, temperature=temperature, top_k=top_k, top_p=top_p)
                #print ("BERT:cand_mlm={}\ntemp={},top_k={},top_p={}".format(cand_mlm, temperature, top_k, top_p))
                self.mlm_cand_model.model.eval() 
                self.mlm_cand_dict = None
        else:
            self.mlm_cand_model = None
            self.mlm_cand_dict = None
            self.n_mlm_cands = None
        if alphabets is not None:
            if self.ensemble:
                self.word_alphabets, self.char_alphabets, self.pos_alphabets, self.rel_alphabet, self.pretrained_alphabets = alphabets
                self.pretrained_alphabet = self.pretrained_alphabets[0]
            else:
                self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.rel_alphabet, self.pretrained_alphabet = alphabets
        self.tokenizer = tokenizer
        self.device = device
        self.lm_device = lm_device
        self.symbolic_root = symbolic_root
        self.symbolic_end = symbolic_end
        self.mask_out_root = mask_out_root
        assert rel_ratio >= 0 and rel_ratio <= 1
        self.rel_ratio = rel_ratio
        self.fluency_ratio = fluency_ratio
        #self.max_perp_diff_per_token = max_perp_diff_per_token
        self.ppl_inc_thres = ppl_inc_thres
        self.batch_size = batch_size
        #self.stop_words = nltk.corpus.stopwords.words('english')
        if 'stop_words' in self.filters:
            logger.info("Init stop word list.")
            self.stop_words = stopwords
        else:
            logger.info("Empty stop word list.")
            self.stop_words = []
        self.stop_tags = ['PRP','PRP$','DT','CC','IN','CD','UH','WDT','WP','WP$','-LRB-','-RRB-','.','``',"\'\'",':',',','?',';']
        self.random_backoff = random_backoff
        self.wordpiece_backoff = wordpiece_backoff
        if self.wordpiece_backoff and self.tokenizer is None:
            print ("Wordpiece backoff requires encoder with tokenizer!")
            exit()
        
        logger.info("Relation ratio:{}, Fluency ratio:{}".format(rel_ratio, fluency_ratio))
        #logger.info("Max ppl difference per token:{}, ppl diff threshold:{}".format(max_perp_diff_per_token, ppl_inc_thres))
        logger.info("Max ppl inc threshold:{}".format(ppl_inc_thres))
        logger.info("Random backoff:{}, Wordpice backoff:{}".format(self.random_backoff, self.wordpiece_backoff))
        self.max_knn_candidates = max_knn_candidates
        self.min_word_cos_sim = min_word_cos_sim
        self.min_sent_cos_sim = min_sent_cos_sim
        
        if 'train' in self.filters and self.train_vocab is None:
            print ("Must input train vocab path for train filter!")
            exit()
        if 'word_sim' in self.filters and self.nn is None:
            print ("Must input embedding path for word cos sim filter!")
            exit()
        if 'sent_sim' in self.filters and self.sent_encoder is None:
            print ("Must input sentence encoder path for sent cos sim filter!")
            exit()
        if 'lm' in self.filters and self.adv_lm is None:
            print ("Must input language model (gpt2) path for lm filter!")
            exit()
        if self.cached_cands is None and 'embedding' in self.generators and self.nn is None:
            print ("Must input embedding path for embedding generator!")
            exit()
        if self.cached_cands is None and 'mlm' in self.generators and self.mlm_cand_model is None and self.mlm_cand_dict is None:
            print ("Must input bert path or cached mlm cands for mlm generator!")
            exit()
        if 'mlm' in self.generators and self.dynamic_mlm_cand and self.mlm_cand_model is None:
            print ("Using dynamic mlm candidates, Must input bert path for mlm generator!")
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

    def str2id(self, tokens, tags):
        #word_ids = [[self.word_alphabet.get_index(x) for x in s] for s in tokens]
        if self.ensemble:
            num_models = len(self.word_alphabets)
            word_ids = [[] for _ in range(num_models)]
            bpes, first_idx = [], []
            if self.model.hyps['input']['use_pos']:
                tag_ids = [[] for _ in range(num_models)]
            else:
                tag_ids = None
            for i in range(num_models):
                for s in tokens:
                    word_list = []
                    for x in s:
                        x = DIGIT_RE.sub("0", x) if self.normalize_digits else x
                        word_list.append(self.word_alphabets[i].get_index(x))
                    word_ids[i].append(word_list)
                if self.model.hyps['input']['use_pos']:
                    tag_ids[i] = [[self.pos_alphabets[i].get_index(x) for x in s] for s in tags]

                if self.tokenizer[i] is not None:
                    sub_bpes, sub_first_idx = convert_tokens_to_ids(self.tokenizer[i], tokens)
                    sub_bpes = sub_bpes.to(self.device)
                    sub_first_idx = sub_first_idx.to(self.device)
                else:
                    sub_bpes, sub_first_idx = None, None
                bpes.append(sub_bpes)
                first_idx.append(sub_first_idx)
        else:
            word_ids = []
            for s in tokens:
                word_list = []
                for x in s:
                    x = DIGIT_RE.sub("0", x) if self.normalize_digits else x
                    word_list.append(self.word_alphabet.get_index(x))
                word_ids.append(word_list)
            if self.model.hyps['input']['use_pos']:
                tag_ids = [[self.pos_alphabet.get_index(x) for x in s] for s in tags]
            else:
                tag_ids = None

            if self.model.pretrained_lm != "none":
                bpes, first_idx = convert_tokens_to_ids(self.tokenizer, tokens)
                bpes = bpes.to(self.device)
                first_idx = first_idx.to(self.device)
            else:
                bpes, first_idx = None, None
        #pre_ids = [[self.pretrained_alphabet.get_index(x) for x in s] for s in tokens]
        pre_ids = []
        for s in tokens:
            pre_list = []
            for w in s:
                pid = self.pretrained_alphabet.get_index(w)
                if pid == 0:
                    pid = self.pretrained_alphabet.get_index(w.lower())
                pre_list.append(pid)
            pre_ids.append(pre_list)
        
        if not self.model.hyps['input']['use_char']:
            chars = None
        if not self.model.lan_emb_as_input:
            lan_id = None
        if self.model.use_elmo:
            elmo_inputs = batch_to_ids(tokens)
            elmo_inputs = elmo_inputs.to(self.device)
        else:
            elmo_inputs = None

        data_size = len(tokens)
        max_length = max([len(s) for s in tokens])
        if self.ensemble:
            wid_inputs = [np.empty([data_size, max_length], dtype=np.int64) for _ in range(num_models)]
            pid_inputs = [np.empty([data_size, max_length], dtype=np.int64) for _ in range(num_models)]
        else:
            wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
            pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        pre_inputs = np.empty([data_size, max_length], dtype=np.int64)
        tid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        
        masks = np.zeros([data_size, max_length], dtype=np.float32)

        for i in range(data_size):
            if self.ensemble:
                inst_size = len(word_ids[0][i])
                for j in range(num_models):
                    wids = word_ids[j][i]
                    # word ids
                    wid_inputs[j][i, :inst_size] = wids
                    wid_inputs[j][i, inst_size:] = PAD_ID_WORD
                    # pos ids
                    if tag_ids is not None:
                        pids = tag_ids[j][i]
                        pid_inputs[j][i, :inst_size] = pids
                        pid_inputs[j][i, inst_size:] = PAD_ID_TAG
            else:
                wids = word_ids[i]
                inst_size = len(wids)
                # word ids
                wid_inputs[i, :inst_size] = wids
                wid_inputs[i, inst_size:] = PAD_ID_WORD
                
                # pos ids
                if tag_ids is not None:
                    pids = tag_ids[i]
                    pid_inputs[i, :inst_size] = pids
                    pid_inputs[i, inst_size:] = PAD_ID_TAG
            
            preids = pre_ids[i]
            # pretrained ids
            pre_inputs[i, :inst_size] = preids
            pre_inputs[i, inst_size:] = PAD_ID_WORD
            # masks
            if self.symbolic_end:
                # mask out the end token
                masks[i, :inst_size-1] = 1.0
            else:
                masks[i, :inst_size] = 1.0
            #for j, wid in enumerate(wids):
            #    if word_alphabet.is_singleton(wid):
            #        single[i, j] = 1
        if self.mask_out_root:
            masks[:,0] = 0

        if self.ensemble:
            words = [torch.from_numpy(wid_input).to(self.device) for wid_input in wid_inputs]
            pos = [torch.from_numpy(pid_input).to(self.device) for pid_input in pid_inputs]
        else:
            words = torch.from_numpy(wid_inputs).to(self.device)
            pos = torch.from_numpy(pid_inputs).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)
        pres = torch.from_numpy(pre_inputs).to(self.device)

        for start_idx in range(0, data_size, self.batch_size):
            excerpt = slice(start_idx, start_idx + self.batch_size)
            if self.ensemble:
                b_words = [word[excerpt, :] for word in words] 
                b_pos = [pos_[excerpt, :] for pos_ in pos] 
                if chars is not None:
                    b_chars = chars[excerpt, :]
                else:
                    b_chars = [None] * num_models

                b_bpes = [sub_bpes[excerpt,:] if sub_bpes is not None else None for sub_bpes in bpes]
                b_first_idx = [sub_first_idx[excerpt,:] if sub_first_idx is not None else None for sub_first_idx in first_idx]
            else:
                b_words = words[excerpt, :]
                b_pos = pos[excerpt, :]
                if chars is not None:
                    b_chars = chars[excerpt, :]
                else:
                    b_chars = None

                if bpes is not None:
                    b_bpes = bpes[excerpt, :]
                else:
                    b_bpes = None
                if first_idx is not None:
                    b_first_idx = first_idx[excerpt, :]
                else:
                    b_first_idx = None

            b_pres = pres[excerpt, :]
            b_masks = masks[excerpt, :]
            if elmo_inputs is not None:
                b_elms = elmo_inputs[excerpt, :]
            else:
                b_elms = None
            b_lan_id = None
            yield b_words, b_pres, b_chars, b_pos, b_masks, b_bpes, b_first_idx, b_elms, b_lan_id

    def get_prediction(self, tokens, tags):
        """
        Input:
            tokens: List[List[str]], (batch, seq_len)
            tags: List[List[str]], (batch, seq_len)
        Output:
            heads_pred: (batch, seq_len)
            rels_pred: (batch, seq_len)
        """
        self.model.eval()
        heads_pred_list, rels_pred_list = [], []
        with torch.no_grad():
            if self.model.hyps['model'] == 'Biaffine':
                for words, pres, chars, pos, masks, bpes, first_idx, input_elmo, lan_id in self.str2id(tokens, tags):
                    heads_pred, rels_pred = self.model.decode(words, pres, chars, pos, mask=masks, 
                            bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id, 
                            leading_symbolic=common.NUM_SYMBOLIC_TAGS)
                    heads_pred_list.append(heads_pred)
                    rels_pred_list.append(rels_pred)
            elif self.model.hyps['model'] == 'StackPointer':
                for words, pres, chars, pos, masks, bpes, first_idx,input_elmo, lan_id in self.str2id(tokens, tags):
                    heads_pred, rels_pred = self.model.decode(words, pres, chars, pos, mask=masks, beam=self.beam,
                            bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id, 
                            leading_symbolic=common.NUM_SYMBOLIC_TAGS)
                    heads_pred_list.append(heads_pred)
                    rels_pred_list.append(rels_pred)
        heads_pred = np.concatenate(heads_pred_list, axis=0)
        rels_pred = np.concatenate(rels_pred_list, axis=0)
        return heads_pred, rels_pred

    def gen_importance_batch(self, tokens, tags):
        """
        Input:
            tokens: List[str], (seq_len)
            tags: List[str], (seq_len)
        Output:
            batch_tokens: List[List[str]], (batch, seq_len)
            batch_tags: List[List[str]], (batch, seq_len)
        """
        if self.ensemble:
            unk_token = '<_UNK>'
            for tokenizer in self.tokenizer:
                if tokenizer is not None:
                    unk_token = tokenizer.unk_token
                    break
        else:
            if self.model.pretrained_lm != "none" and not self.model.use_elmo:
                unk_token = self.tokenizer.unk_token
            else: # this is defined in alphabet.py
                unk_token = '<_UNK>'
        batch_len = len(tokens)+1-self.symbolic_root
        batch_tokens = [tokens.copy() for _ in range(batch_len)]
        batch_tags = [tags.copy() for _ in range(batch_len)]
        unk_id = 1 if self.symbolic_root else 0
        for i in range(1, batch_len):
            batch_tokens[i][unk_id] = unk_token
            unk_id += 1
        return batch_tokens, batch_tags

    def get_punct_mask(self, tokens, tags):
        assert len(tokens) == len(tags)
        punct_mask = np.ones(len(tokens))
        for i, tag in enumerate(tags):
            if tag in self.punct_set:
                punct_mask[i] = 0
        return punct_mask

    def calc_importance(self, batch_tokens, batch_tags, heads, rel_ids, punct_mask, debug=False):
        """
        Input:
            batch_tokens: List[List[str]], (batch, seq_len), the first line should be the original seq
            batch_tags: List[List[str]], (batch, seq_len), the first line should be the original seq
            heads: List[int], (seq_len)
            rel_ids: List[int], (seq_len)
            punct_mask: np.array(seq_len)
        Output:
            importance: List[int], (seq_len), importance of each seq
            word_rank: List[int], (seq_len), word id ranked by importance
        """
        heads_pred, rels_pred = self.get_prediction(batch_tokens, batch_tags)
        punct_mask_ = np.tile(punct_mask, (len(heads_pred),1))
        heads_gold = np.tile(np.array(heads), (len(heads_pred),1))
        heads_change_mask = np.where(heads_pred != heads_gold, 1, 0)
        # this should minus the diff between original prediction (line 0) and gold
        # mask out punctuations
        heads_change = (heads_change_mask * punct_mask_).sum(axis=1)
        heads_change = heads_change - heads_change[0]
        if debug:
            print (batch_tokens)
            print ("punct_mask:\n", punct_mask_)
            print ("gold heads:\n", heads_gold)
            print ("pred heads:\n", heads_pred)
            print ("mask:\n", heads_change_mask)
            print ("heads change:\n", heads_change)

        rels_gold = np.tile(np.array(rel_ids), (len(rels_pred),1))
        rels_change_mask = np.where(rels_pred != rels_gold, 1, 0)
        # if the arc is wrong, the rel must be wrong
        rels_change_mask = np.where(rels_change_mask+heads_change_mask>0, 1, 0)
        # this should minus the diff between original prediction (line 0) and gold
        # mask out punctuations
        rels_change = (rels_change_mask * punct_mask_).sum(axis=1)
        rels_change = rels_change - rels_change[0]
        if debug:
            print ("gold rels:\n", rel_ids)
            print ("pred rels:\n", rels_pred)
            print ("mask:\n", rels_change_mask)
            print ("rels change:\n", rels_change)
        
        importance = (1-self.rel_ratio) * heads_change + self.rel_ratio * rels_change
        return importance, heads_change, rels_change

    def calc_word_rank(self, tokens, tags, heads, rel_ids, punct_mask, debug=False):
        batch_tokens, batch_tags = self.gen_importance_batch(tokens, tags)
        importance, _, _ = self.calc_importance(batch_tokens, batch_tags, heads, rel_ids, punct_mask, debug)
        word_rank = (-importance).argsort()
        if debug:
            print ("importance:\n", importance)
            print ("word_rank:\n", word_rank)
        return word_rank

    def gen_cand_batch(self, tokens, cands, idx, tags):
        """
        Input:
            tokens: List[str], (seq_len)
            tags: List[str], (seq_len)
        Output:
            batch_tokens: List[List[str]], (batch, seq_len)
            batch_tags: List[List[str]], (batch, seq_len)
        """
        batch_len = len(cands)+1
        batch_tokens = [tokens.copy() for _ in range(batch_len)]
        batch_tags = [tags.copy() for _ in range(batch_len)]
        for i in range(1, batch_len):
            batch_tokens[i][idx] = cands[i-1]
        return batch_tokens, batch_tags

    def get_change_score(self, tokens, cands, idx, tags, heads, rel_ids, punct_mask, debug=False):
        batch_tokens, batch_tags = self.gen_cand_batch(tokens, cands, idx, tags)
        # (cand_size+1), the 1st is the original sentence
        change_score, head_change, rel_change = self.calc_importance(batch_tokens, batch_tags, heads, rel_ids, punct_mask, debug)
        # ignore the original sent
        change_score = change_score[1:]
        if debug:
            #print ("batch tokens:\n", batch_tokens)
            print ("importance:\n", change_score)
            #print ("word_rank:\n", word_rank)
        return change_score, head_change[1:], rel_change[1:]

    def get_batch(self, input_ids):
        # (cand_size+1, seq_len)
        data_size = input_ids.size()[0]
        for start_idx in range(0, data_size, self.batch_size):
            excerpt = slice(start_idx, start_idx + self.batch_size)
            yield input_ids[excerpt, :]

    def calc_perplexity(self, tokens):
        if self.symbolic_root:
            lines = [' '.join(t[1:]) for t in tokens]
        else:
            lines = [' '.join(t) for t in tokens]
        batch_encoding = self.adv_tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=128)
        examples = [torch.tensor(b,dtype=torch.long) for b in batch_encoding["input_ids"]]
        input_ids = torch.nn.utils.rnn.pad_sequence(examples, batch_first=True)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        perp_list = []
        for batch in self.get_batch(input_ids):
            batch = batch.to(self.lm_device)
            outputs = self.adv_lm(batch)
            # (batch_size, seq_len, voc_size)
            logits = outputs[0]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            # (batch, seq_len)
            loss = loss_fct(shift_logits.transpose(1,2), shift_labels)
            # (batch)
            loss = loss.mean(-1)
            perplexity = torch.exp(loss).cpu().numpy()
            perp_list.append(perplexity)
        perplexity = np.concatenate(perp_list, axis=0)
        return perplexity

    def get_perp_diff(self, tokens, cands, idx, debug=False):
        batch_tokens, _ = self.gen_cand_batch(tokens, cands, idx, tokens)
        # (cand_size+1), the first is the original sentence
        perplexity = self.calc_perplexity(batch_tokens)
        if debug:
            for perp, tokens in zip(perplexity, batch_tokens):
                print ("sent (perp={}):\n".format(perp), " ".join(tokens[:idx])+" </"+tokens[idx]+"/> "+" ".join(tokens[idx+1:]))
        # (cand_size)
        perp_diff = perplexity[1:] - perplexity[0]
        return perp_diff

    def filter_cands_with_lm(self, tokens, cands, idx, debug=False):
        new_cands, new_perp_diff = [], []
        clean_toks = tokens
        # (cand_size)
        #perp_diff = self.get_perp_diff(tokens, cands, idx)
        perp_diff = self.get_perp_diff(clean_toks, cands, idx)
        for i in range(len(cands)):
            if perp_diff[i] <= self.ppl_inc_thres:
                new_cands.append(cands[i])
                new_perp_diff.append(perp_diff[i])
        return new_cands, np.array(new_perp_diff), perp_diff

    def filter_cands_with_word_sim(self, token, cands, debug=False):
        new_cands= []
        new_sims, all_sims = [], []
        # (cand_size)
        for i in range(len(cands)):
            sim = self.get_word_cos_sim(token.lower(), cands[i].lower())
            all_sims.append(sim)
            if sim is not None and sim >= self.min_word_cos_sim:
                new_cands.append(cands[i])
                new_sims.append(sim)
        return new_cands, new_sims, all_sims

    def list2str(self, tokens):
        str = ""
        no_blank_toks = ['.',',','!',':',';','?',"'s", "n't"]
        for tok in tokens:
            if tok in no_blank_toks:
                str += tok
            else:
                str += ' ' + tok
        return str

    def filter_cands_with_grammar_checker(self, tokens, cands, idx, debug=False):
        new_cands = []
        all_errors = []
        batch_tokens, _ = self.gen_cand_batch(tokens, cands, idx, tokens)
        # remove the virtual ROOT symbol
        if self.symbolic_root:
            batch_tokens = [toks[1:] for toks in batch_tokens]
        #batch_sents = [' '.join(toks) for toks in batch_tokens]
        batch_sents = [self.list2str(toks) for toks in batch_tokens]
        origin_matches = self.grammar_checker.check(batch_sents[0])
        num_origin_err = len(origin_matches)
        origin_errors = [(match.fromx, match.tox, match.msg.encode("utf-8")) for match in origin_matches]
        # (cand_size)
        for i in range(1,len(batch_sents)):
            matches = self.grammar_checker.check(batch_sents[i])
            # only keep substitutes that do not increase grammar error
            if len(matches) <= num_origin_err:
                new_cands.append(cands[i-1])
            error = []
            for match in matches:
                error_tok = batch_sents[i][match.fromx: match.tox]
                if match not in origin_matches:
                    error.append((match.fromx, match.tox, error_tok, match.msg.encode("utf-8")))
            all_errors.append(error)
        return new_cands, all_errors, origin_errors, batch_sents[0]

    # only allow cands that have appeared in training set
    def filter_cands_with_train_vocab(self, cands, debug=False):
        new_cands = []
        for c in cands:
            if c in self.train_vocab or c.lower() in self.train_vocab:
                new_cands.append(c)
        return new_cands

    def filter_cands(self, tokens, cands, idx, debug=False):
        new_cands = []
        #print ("raw cands:", cands)
        # filter out stop words from candidates
        for cand in cands:
            if cand not in self.stop_words:
                new_cands.append(cand)
        cands = new_cands
        #print ("cands:", cands)
        if "train" in self.filters:
            cands = self.filter_cands_with_train_vocab(cands)
            if len(cands) == 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), no cand from train, continue".format(idx, tokens[idx]))
                return cands, None
            else:
                if ISPRINT:
                    print ("--------------------------")
                    print ("Idx={}({})".format(idx, tokens[idx]))
                    print ("cands from train:", cands)
        all_cands = cands.copy()
        if "word_sim" in self.filters:
            cands, w_sims, all_w_sims = self.filter_cands_with_word_sim(tokens[idx], cands)
            if len(cands) == 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), all word_sim less than min, continue".format(idx, tokens[idx]))
                    print ("word_sims:", *zip(all_cands, all_w_sims)) #jeffrey

                return cands, None
            else:
                if ISPRINT:
                    print ("--------------------------")
                    print ("Idx={}({})".format(idx, tokens[idx]))
                    print ("word_sims:", *zip(all_cands, all_w_sims)) #jeffrey

        all_cands = cands.copy()
        if "sent_sim" in self.filters:
            cands, s_sims, all_s_sims = self.filter_cands_with_sent_sim(tokens, cands, idx)
            if len(cands) == 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), all sent_sim less than min, continue".format(idx, tokens[idx]))
                    print ("sent_sim:", *zip(all_cands, all_s_sims))
                return cands, None
            else:
                if ISPRINT:
                    print ("--------------------------")
                    print ("Idx={}({})".format(idx, tokens[idx]))
                    print ("sent_sim:", *zip(all_cands, all_s_sims))
        # filter with grammar checker
        all_cands = cands.copy()
        if "grammar" in self.filters:
            cands, all_errors, origin_errors, origin_str = self.filter_cands_with_grammar_checker(tokens, cands, idx, debug==2)
            if len(cands) == 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("origin sent:", origin_str)
                    print ("origin errors:", origin_errors)
                    print ("Idx={}({}), all fail grammar checker, continue".format(idx, tokens[idx]))
                    print ("errors:", *zip(all_cands, all_errors))
                return cands, None
            else:
                if ISPRINT:
                    print ("--------------------------")
                    print ("origin sent:", origin_str)
                    print ("origin errors",origin_errors)
                    print ("Idx={}({})".format(idx, tokens[idx]))
                    print ("errors:", *zip(all_cands, all_errors)) # jeffrey

        # filter with language model
        all_cands = cands.copy()
        perp_diff = None
        if "lm" in self.filters:
            cands, perp_diff, all_perp_diff = self.filter_cands_with_lm(tokens, cands, idx, debug==2)
            if len(cands) == 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), all perp_diff above thres, continue".format(idx, tokens[idx]))
                    print ("ppl_diff:", *zip(all_cands, all_perp_diff))
                return cands, None
            else:
                if ISPRINT:
                    print ("--------------------------")
                    print ("Idx={}({})".format(idx, tokens[idx]))
                    print ("ppl_diff:", *zip(all_cands, all_perp_diff))

        return cands, perp_diff

    def cos_sim(self, e1, e2):
        e1 = torch.tensor(e1)
        e2 = torch.tensor(e2)
        cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2)
        return cos_sim.numpy()

    def filter_cands_with_sent_sim(self, tokens, cands, idx, debug=False):
        clean_toks = tokens
        batch_tokens, _ = self.gen_cand_batch(clean_toks, cands, idx, tokens)
        #batch_tokens, _ = self.gen_cand_batch(tokens, cands, idx, tokens)
        sents = [' '.join(toks) for toks in batch_tokens]
        # sent-0 is original sent
        with tf.device('/cpu:0'):
            sent_embeds = self.sent_encoder(sents).numpy()
        #print ("sent_sim batch toks:\n", batch_tokens)
        new_cands= []
        new_sims, all_sims = [], []
        # (cand_size)
        for i in range(1,len(cands)+1):
            sim = self.cos_sim(sent_embeds[i], sent_embeds[0])
            all_sims.append(sim)
            if sim >= self.min_sent_cos_sim:
                new_cands.append(cands[i-1])
                new_sims.append(sim)
        return new_cands, new_sims, all_sims

    def get_best_cand(self, score, change_score):
        cand_rank = (-score).argsort()
        for i in range(len(score)):
            cand_idx = cand_rank[i]
            if change_score[cand_idx] > 0:
                return cand_idx
        return None

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
            """
            if self.tagger == 'stanza':
                data = self.stanza_tagger([tmps]).sentences[0]
                cand_tag = data.words[idx].xpos
            elif self.tagger == 'nltk':
                #cand_tag = nltk.pos_tag([cand.lower()])[0][1]
                cand_tag = nltk.pos_tag(tmps)[idx][1]
            elif self.tagger == 'spacy':
                #cand_tag = nlp(cand.lower())[0].tag_
                cand_tag = nlp(' '.join(tmps))[idx].tag_
            elif self.tagger == 'stanford':
                cand_tag = self.stanford_tagger.tag(tmps)[idx][1]
            """
            #print ("cand={}, tag={}".format(cand, cand_tag))
            
        return filtered_cands

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
        # load directly from preprocessed file
        if not self.dynamic_mlm_cand and self.mlm_cand_dict is not None:
            if self.symbolic_root and idx == 0: return []
            sent_mlm_cands = self.mlm_cand_dict[str(sent_id)]
            #if self.symbolic_root:
            #    sent_mlm_cands = [{"orig":ROOT, "cands":[]}] + sent_mlm_cands
            assert len(sent_mlm_cands) == (len(tokens) - self.symbolic_root)
            mlm_cands = sent_mlm_cands[idx-self.symbolic_root]
            #print (idx, tokens[idx])
            #print (mlm_cands)
            assert mlm_cands["orig"] == tokens[idx]
            return mlm_cands["cands"]
        elif self.mlm_cand_model is not None:
            original_word = tokens[idx]
            tmps = tokens.copy()
            tmps[idx] = self.mlm_cand_model.MASK_TOKEN
            masked_text = ' '.join(tmps)

            candidates = self.mlm_cand_model.predict(masked_text, target_word=original_word, n=n)

            return [candidate[0] for candidate in candidates]

    def update_cand_set(self, token, cand_set, cands, lower_set):
        for c in cands:
            if c.lower() not in lower_set and c.lower() != token.lower():
                cand_set.append(recover_word_case(c, token))
                #cand_set.append(c)
                lower_set.add(c.lower())
        return cand_set, lower_set

    def _get_candidate_set_from_cache(self, tokens, tag, idx, sent_id=None):
        # 恢复正常之后记得修改
        offset = 0
        sent_id = sent_id + offset
        # *************************************
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

    def _get_candidate_set(self, tokens, tag, idx, sent_id=None, cache=False,debug=False):
        token = tokens[idx]
        if cache:
            cache_data = {'sem_cands':[], 'syn_cands':[], 'emb_cands':[],
                          'mlm_cands':[]}
        else:
            cache_data = None
        if token.lower() in self.stop_words:
            return [], cache_data
        if tag in self.stop_tags:
            return [], cache_data
        if token == PAD or token == ROOT:
            return [], cache_data
        candidate_set = []
        lower_set = set()
        if debug:
            print ("origin token: ", token)
        if 'sememe' in self.generators:
            sem_cands = self.get_sem_cands(token, tag)
            sem_cands = self.post_process(tokens.copy(), sem_cands, tag, idx
)
            if debug:
                print ("sememe:", sem_cands)
            self.update_cand_set(token, candidate_set, sem_cands, lower_set)
            #print ("sememe:", sem_cands)
        else:
            sem_cands = []
        if 'synonym' in self.generators:
            syn_cands = self.get_syn_cands(token, tag)
            syn_cands = self.post_process(tokens.copy(), syn_cands, tag, idx)
            if debug:
                print ("syn:", syn_cands)
            self.update_cand_set(token, candidate_set, syn_cands, lower_set)
            #print ("syn:", syn_cands)
        else:
            syn_cands = []
        if 'embedding' in self.generators:
            emb_cands = self.get_emb_cands(tokens.copy(), tag, idx)
            emb_cands = self.post_process(tokens.copy(), emb_cands, tag, idx)
            if debug:
                print ("knn cands:\n", emb_cands)
            self.update_cand_set(token, candidate_set, emb_cands, lower_set)
            #print ("knn cands:\n", emb_cands)
        else:
            emb_cands = []
        if 'mlm' in self.generators:
            mlm_cands = self.get_mlm_cands(tokens.copy(), tag, idx, sent_id=sent_id)
            #if debug:
            print ("token:{}, mlm_cands:{}".format(tokens[idx], mlm_cands))
            mlm_cands = self.post_process(tokens.copy(), mlm_cands, tag, idx)
            self.update_cand_set(token, candidate_set, mlm_cands, lower_set)
            
        else:
            mlm_cands = []
        if cache:
            cache_data = {'sem_cands':sem_cands, 'syn_cands':syn_cands, 'emb_cands':emb_cands,
                          'mlm_cands':mlm_cands}
            
        return candidate_set, cache_data
    
    def get_candidate_set(self, tokens, tag, idx, sent_id=None, cache=False):
        if self.cached_cands is not None:
            cand_set = self._get_candidate_set_from_cache(tokens, tag, idx, sent_id=sent_id)
            cache_data = None
        else:
            cand_set, cache_data = self._get_candidate_set(tokens, tag, idx, sent_id=sent_id, cache=cache)

        return cand_set, cache_data

    def attack(self, tokens, tags, heads, rel_ids, sent_id=None, debug=False, cache=False):
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
        word_rank = self.calc_word_rank(tokens, tags, heads, rel_ids, punct_mask, debug==2)
        x_len = len(tokens)
        tag_list = ['JJ', 'NN', 'RB', 'VB']
        neigbhours_list = []
        cand_cache = []
        #stop_words = nltk.corpus.stopwords.words('english')
        if not self.dynamic_mlm_cand:
            for i in range(x_len):
                #print (adv_tokens[i], self._word2id(adv_tokens[i]))
                cands, cache_data = self.get_candidate_set(adv_tokens, tags[i], i, sent_id=sent_id, cache=cache)
                neigbhours_list.append(cands)
                if cache and self.cached_path is None:
                    cache_data['id'] = i
                    cache_data['token'] = tokens[i]
                    cand_cache.append(cache_data)
            neighbours_len = [len(x) for x in neigbhours_list]
            #print (neigbhours_list)
            if np.sum(neighbours_len) == 0:
                return None, cand_cache
        

        change_edit_ratio = -1
        total_change_score = 0
        total_head_change = 0
        total_rel_change = 0
        total_perp_diff = 0.0
        total_score = 0
        num_edit = 0
        #max_perp_diff = x_len * self.max_perp_diff_per_token
        # at least allow one modification
        max_mod_token = max(1, int(x_len * self.max_mod_percent))

        #cand_cache = []
        if "mlm" in self.generators and self.dynamic_mlm_cand and cache and self.cached_path is None:
            cand_cache = [[] for _ in range(len(tokens))]

        if debug == 3:
            #print ("tokens:\n", adv_tokens)
            print ("importance rank:\n", word_rank)

        for idx in word_rank:
            idx = int(idx)
            if "mlm" in self.generators and self.dynamic_mlm_cand:
                cands, cache_data = self.get_candidate_set(adv_tokens, tags[idx], idx, sent_id=sent_id, cache=cache)
                if cache and self.cached_path is None:
                    cache_data['id'] = idx
                    cache_data['token'] = tokens[idx]
                    cand_cache[idx] = cache_data

                if len(cands) == 0:
                    if debug == 3:
                        print ("--------------------------")
                        print ("Idx={}({}), no cands, continue".format(idx, tokens[idx]))
                    continue
            else:
                cands = neigbhours_list[idx]
            # skip the edit for ROOT
            if self.symbolic_root and idx == 0: continue

            cands, perp_diff = self.filter_cands(adv_tokens.copy(), cands, idx, debug=debug)
            if len(cands) == 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), all cands filtered out, continue".format(idx, tokens[idx]))
                continue
            all_cands = cands.copy()
            if "lm" in self.filters:
                blocked_perp_diff = np.where(perp_diff>0, perp_diff, 0)
            # (cand_size)
            change_score, head_change, rel_change = self.get_change_score(adv_tokens, cands, idx, tags, heads, rel_ids, punct_mask, debug==2)
            change_rank = (-change_score).argsort()
            # this means the biggest change is 0
            if change_score[change_rank[0]] <= 0:
                if change_score[change_rank[0]] < 0 or (not self.random_backoff and not self.wordpiece_backoff):
                    if debug == 3:
                        print ("--------------------------")
                        print ("Idx={}({}), no cand can make change, continue".format(idx, tokens[idx]))
                        print ("change_scores:", *zip(cands, change_score))
                else:
                    if num_edit >= max_mod_token:
                        continue
                    num_nochange_sub = 0
                    for i in range(len(change_rank)):
                        if change_score[change_rank[i]] == 0:
                            num_nochange_sub += 1
                        else:
                            break
                    if self.wordpiece_backoff:
                        nochange_cand_ids = change_rank[:num_nochange_sub]
                        max_wp_len = 0
                        chosen_idx = -1
                        nochange_cands = []
                        nochange_wp_lens = []
                        for j in nochange_cand_ids:
                            wp_len = len(self.tokenizer.tokenize(cands[j]))
                            nochange_cands.append(cands[j])
                            nochange_wp_lens.append(wp_len)
                            if wp_len > max_wp_len:
                                max_wp_len = wp_len
                                chosen_idx = j
                        adv_tokens[idx] = cands[chosen_idx]
                        num_edit += 1
                        if "lm" in self.filters:
                            total_perp_diff += blocked_perp_diff[chosen_idx]
                        if debug == 3:
                            print ("--------------------------")
                            print ("Idx={}({}), wp backoff chose:{} since no cand makes change".format(idx, tokens[idx], cands[chosen_idx]))
                            print ("wordpiece lens:", *zip(nochange_cands, nochange_wp_lens))

                    elif self.random_backoff:
                        # only choose the subs that will not reduce the error
                        chosen_rank_idx = random.randint(0, num_nochange_sub-1)
                        chosen_idx = change_rank[chosen_rank_idx]
                        adv_tokens[idx] = cands[chosen_idx]
                        num_edit += 1
                        if "lm" in self.filters:
                            total_perp_diff += blocked_perp_diff[chosen_idx]
                        if debug == 3:
                            print ("--------------------------")
                            print ("Idx={}({}), random backoff chose:{} since no cand makes change\ncands:{}\nchange_scores:{}".format(idx, tokens[idx], cands[chosen_idx], cands, change_score))
                            if "lm" in self.filters:
                                print ("perp diff: {}".format(perp_diff))
                continue
            if "lm" in self.filters:
                # penalize the score for disfluency substitution
                # if the perplexity of new sent is lower than the original one, no bonus
                score = (1 - self.fluency_ratio) * change_score - self.fluency_ratio * blocked_perp_diff
            else:
                score = change_score
            best_cand_idx = self.get_best_cand(score, change_score)
            if best_cand_idx is None:
                if ISPRINT:
                    print ("--------------------------")
                    print ("Idx={}({}), can't find best cand, continue".format(idx, tokens[idx]))
                    print ("change_scores:", *zip(cands, change_score))
                if "lm" in self.filters:
                    if ISPRINT:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
                continue
            #cand_rank = (-score).argsort()
            best_cand = cands[best_cand_idx]
            best_c_score = change_score[best_cand_idx]
            best_score = score[best_cand_idx]
            new_ratio = (total_change_score + best_c_score) / (num_edit + 1)
            #if ("lm" in self.filters and total_perp_diff<=max_perp_diff) or (new_ratio > change_edit_ratio):
            if num_edit < max_mod_token:
                change_edit_ratio = new_ratio
                num_edit += 1
                total_change_score += best_c_score
                total_score += best_score
                total_head_change += head_change[best_cand_idx]
                total_rel_change += rel_change[best_cand_idx]
                adv_tokens[idx] = best_cand
                if "lm" in self.filters:
                    total_perp_diff += blocked_perp_diff[best_cand_idx]
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), chosen cand:{}, total_change_score:{}, change_edit_ratio:{}".format(
                            idx, tokens[idx], best_cand, total_change_score, change_edit_ratio))
                    print ("change_scores:", *zip(cands, change_score))
                    if "lm" in self.filters:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
            else:
                if debug == 3:
                    print ("------------Stopping------------")
                    print ("Idx={}({}), chosen cand:{}, total_change_score:{}, change_edit_ratio:{}".format(
                            idx, tokens[idx], best_cand, total_change_score, change_edit_ratio))
                    print ("change_scores:", *zip(cands, change_score))
                    if "lm" in self.filters:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
                break
        if adv_tokens == tokens:
            return None, cand_cache
        sent_str = ""
        for x,y in zip(tokens, adv_tokens):
            if x == y:
                sent_str += x + " "
            else:
                sent_str += y + " [ " + x + " ] "
        if ISPRINT:
            print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print ("Success attack (change: head:{}, rel:{}, score:{}), adv sent:\n{}".format(
                total_head_change, total_rel_change, total_change_score, sent_str))
        adv_infos = (num_edit, total_score, total_change_score, total_perp_diff,
                    total_head_change, total_rel_change)
        return (adv_tokens, adv_infos), cand_cache
