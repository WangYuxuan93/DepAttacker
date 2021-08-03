# -*- coding:utf-8 -*-

__author__ = 'max'

from collections import OrderedDict
import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
import gzip
import lzma
import os

from neuronlp2.io.logger import get_logger
from neuronlp2.io.alphabet import Alphabet
from neuronlp2.io.common import DIGIT_RE
from neuronlp2.io.common import PAD, ROOT, END


def load_embedding_dict(embedding, embedding_path, normalize_digits=True):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :return: embedding dict, embedding dimention, caseless
    """
    print("loading embedding: %s from %s" % (embedding, embedding_path))
    if embedding == 'word2vec':
        # loading word2vec
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim
    elif embedding == 'glove':
        # loading GloVe
        embedd_dim = -1
        embedd_dict = OrderedDict()
        with gzip.open(embedding_path, 'rt', encoding="utf-8") as file:
            file.readline()  # 忽略第一句
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'senna':
        # loading Senna
        embedd_dim = -1
        embedd_dict = OrderedDict()
        with gzip.open(embedding_path, 'rt') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'sskip':
        embedd_dim = -1
        embedd_dict = OrderedDict()

        with gzip.open(embedding_path, 'rt',encoding="utf-8") as file:
            # skip the first line
            file.readline()
            for line in file:
                line = line.strip()
                try:
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    if len(tokens) < embedd_dim:
                        continue

                    if embedd_dim < 0:
                        embedd_dim = len(tokens) - 1

                    embedd = np.empty([1, embedd_dim], dtype=np.float32)
                    start = len(tokens) - embedd_dim
                    word = ' '.join(tokens[0:start])
                    embedd[:] = tokens[start:]
                    word = DIGIT_RE.sub("0", word) if normalize_digits else word
                    embedd_dict[word] = embedd
                except UnicodeDecodeError:
                    continue
        return embedd_dict, embedd_dim
    elif embedding == 'xz':
        embedd_dim = -1
        embedd_dict = OrderedDict()

        with lzma.open(embedding_path, 'rt',encoding="utf-8") as file:
            # skip the first line
            file.readline()
            for line in file:
                line = line.strip()
                try:
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    if len(tokens) < embedd_dim:
                        continue

                    if embedd_dim < 0:
                        embedd_dim = len(tokens) - 1

                    embedd = np.empty([1, embedd_dim], dtype=np.float32)
                    start = len(tokens) - embedd_dim
                    word = ' '.join(tokens[0:start])
                    embedd[:] = tokens[start:]
                    word = DIGIT_RE.sub("0", word) if normalize_digits else word
                    embedd_dict[word] = embedd
                except UnicodeDecodeError:
                    continue
        return embedd_dict, embedd_dim
    elif embedding == 'polyglot':
        words, embeddings = pickle.load(open(embedding_path, 'rb'), encoding='latin1')
        _, embedd_dim = embeddings.shape
        embedd_dict = OrderedDict()
        for i, word in enumerate(words):
            embedd = np.empty([1, embedd_dim], dtype=np.float32)
            embedd[:] = embeddings[i, :]
            word = DIGIT_RE.sub("0", word) if normalize_digits else word
            embedd_dict[word] = embedd
        return embedd_dict, embedd_dim

    else:
        raise ValueError("embedding should choose from [word2vec, senna, glove, sskip, polyglot]")


def create_alphabet_from_embedding(alphabet_directory, embedd_dict=None, vocabs=None, max_vocabulary_size=100000,
                                    do_trim=True):
    _START_VOCAB = [PAD, ROOT, END]
    logger = get_logger("Create Pretrained Alphabets")
    pretrained_alphabet = Alphabet('pretrained', default_value=True)
    file = os.path.join(alphabet_directory, 'pretrained.json')
    if not os.path.exists(file):
        if not embedd_dict or not vocabs:
            print ("No embedd dict or vocabs for pretrained alphabet!")
            exit()
        logger.info("Creating Pretrained Alphabets: %s" % alphabet_directory)
        pretrained_alphabet.add(PAD)
        pretrained_alphabet.add(ROOT)
        pretrained_alphabet.add(END)

        pretrained_vocab = list(embedd_dict.keys())
        n_oov = 0
        if do_trim:
            logger.info("Trim pretrained vocab by data")
            for word in vocabs:
                if word in pretrained_vocab:
                    pretrained_alphabet.add(word)
                elif word.lower() in pretrained_vocab:
                    pretrained_alphabet.add(word.lower())
                elif word not in _START_VOCAB:
                    n_oov += 1
        else:
            logger.info("Not trim pretrained vocab by data")
            for word in pretrained_vocab:
                pretrained_alphabet.add(word)
            #for word in vocabs:
            #    if word not in pretrained_vocab and word.lower() not in pretrained_vocab:
            #        n_oov += 1
        #vocab_size = min(len(pretrained_vocab), max_vocabulary_size)
        logger.info("Loaded/Total Pretrained Vocab Size: %d/%d" % (pretrained_alphabet.size(),len(pretrained_vocab)))
        
        pretrained_alphabet.save(alphabet_directory)
    else:
        pretrained_alphabet.load(alphabet_directory)
        #pretrained_vocab = list(embedd_dict.keys())
        #vocab_size = min(len(pretrained_vocab), max_vocabulary_size)
        #assert pretrained_alphabet.size() == (vocab_size + 4)
        
    pretrained_alphabet.close()

    return pretrained_alphabet

def creat_language_alphabet(alphabet_directory, languages=None):
    logger = get_logger("Create Language Alphabets")
    lan_alphabet = Alphabet('language', default_value=True)
    file = os.path.join(alphabet_directory, 'language.json')
    if not os.path.exists(file):
        if not languages:
            print ("No languages for language alphabet!")
            exit()
        logger.info("Creating Language Alphabets: %s" % alphabet_directory)
        for l in languages:
            lan_alphabet.add(l)
        lan_alphabet.save(alphabet_directory)
    else:
        lan_alphabet.load(alphabet_directory)
    #print (lan_alphabet.items())
    logger.info("Total Languages: %d" % (lan_alphabet.size()))
        
    lan_alphabet.close()

    return lan_alphabet


class Oracle():
    def __init__(self):
        pass

    @staticmethod
    def get_init(annotated_sentence, directed_arc_indices, arc_tags):
        graph = {}
        for token_idx in range(len(annotated_sentence) + 1):
            graph[token_idx] = []
        for arc, arc_tag in zip(directed_arc_indices, arc_tags):
            graph[arc[0]].append((arc[1], arc_tag))
        N = len(graph)  # N-1 point, 1 root point
        # i:head_point j:child_point
        top_down_graph = [[] for i in range(N)]  # N-1 real point, 1 root point => N point

        # i:child_point j:head_point ->Bool
        # partial graph during construction
        sub_graph = [[False for i in range(N)] for j in range(N)]

        for i in range(N):
            for head_tuple_of_point_i in graph[i]:
                head = head_tuple_of_point_i[0]
                top_down_graph[head].append(i)
        actions = []
        stack = [0]
        buffer = []
        deque = []
        for i in range(N - 1, 0, -1):
            buffer.append(i)
        return graph,top_down_graph,sub_graph,stack,buffer,deque,actions

    @staticmethod
    def has_head(graph,w0, w1):
        if w0 <= 0:
            return False
        for w in graph[w0]:
            if w[0] == w1:
                return True
        return False

    @staticmethod
    def has_unfound_child(top_down_graph,sub_graph,w):
        for child in top_down_graph[w]:
            if not sub_graph[child][w]:
                return True
        return False

    # return if w has other head except the present one
    @staticmethod
    def has_other_head(graph,sub_graph,w):
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num + 1 < len(graph[w]):
            return True
        return False

    # return if w has any unfound head
    @staticmethod
    def lack_head(graph,sub_graph,w):
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num < len(graph[w]):
            return True
        return False

    # return if w has any unfound child in stack sigma
    # !!! except the top in stack
    @staticmethod
    def has_other_child_in_stack(top_down_graph,sub_graph,stack, w):
        if w <= 0:
            return False
        for c in top_down_graph[w]:
            if c in stack \
                    and c != stack[-1] \
                    and not sub_graph[c][w]:
                return True
        return False

    # return if w has any unfound head in stack sigma
    # !!! except the top in stack
    @staticmethod
    def has_other_head_in_stack(graph,sub_graph,stack, w):
        if w <= 0:
            return False
        for h in graph[w]:
            if h[0] in stack \
                    and h[0] != stack[-1] \
                    and not sub_graph[w][h[0]]:
                return True
        return False

    # return the relation between child: w0, head: w1
    @staticmethod
    def get_arc_label(graph,w0, w1):
        for h in graph[w0]:
            if h[0] == w1:
                return h[1]
    @staticmethod
    def get_oracle_actions_onestep(graph,top_down_graph,sub_graph, stack, buffer, deque, actions):
        b0 = buffer[-1] if len(buffer) > 0 else -1
        s0 = stack[-1] if len(stack) > 0 else -1

        if s0 > 0 and Oracle.has_head(graph,s0, b0):
            if not Oracle.has_unfound_child(top_down_graph,sub_graph,s0) and not Oracle.has_other_head(graph,sub_graph,s0):
                actions.append("LR:" + Oracle.get_arc_label(graph,s0, b0))
                stack.pop()
                sub_graph[s0][b0] = True
                return
            else:
                actions.append("LP:" + Oracle.get_arc_label(graph,s0, b0))
                deque.append(stack.pop())
                sub_graph[s0][b0] = True
                return

        elif s0 >= 0 and Oracle.has_head(graph,b0, s0):
            if not Oracle.has_other_child_in_stack(top_down_graph,sub_graph,stack, b0) and not Oracle.has_other_head_in_stack(graph,sub_graph,stack, b0):
                actions.append("RS:" + Oracle.get_arc_label(graph,b0, s0))
                while len(deque) != 0:
                    stack.append(deque.pop())
                stack.append(buffer.pop())
                sub_graph[b0][s0] = True
                return

            elif s0 > 0:
                actions.append("RP:" + Oracle.get_arc_label(graph,b0, s0))
                deque.append(stack.pop())
                sub_graph[b0][s0] = True
                return

        elif len(buffer) != 0 and not Oracle.has_other_head_in_stack(graph,sub_graph,stack, b0) \
                and not Oracle.has_other_child_in_stack(top_down_graph,sub_graph,stack, b0):
            actions.append("NS")
            while len(deque) != 0:
                stack.append(deque.pop())
            stack.append(buffer.pop())
            return

        elif s0 > 0 and not Oracle.has_unfound_child(top_down_graph,sub_graph,s0) and not Oracle.lack_head(graph,sub_graph,s0):
            actions.append("NR")
            stack.pop()
            return

        elif s0 > 0:
            actions.append("NP")
            deque.append(stack.pop())
            return

        else:
            actions.append('-E-')
            print('"error in oracle!"')
            return

    @staticmethod
    def get_action(graph,top_down_graph,sub_graph,stack,buffer,deque,actions):
        while len(buffer) != 0:
            Oracle.get_oracle_actions_onestep(graph,top_down_graph,sub_graph, stack, buffer, deque, actions)
        return actions

def get_oracle_actions(annotated_sentence, directed_arc_indices, arc_tags):
    graph = {}
    for token_idx in range(len(annotated_sentence) + 1):
        graph[token_idx] = []

    # construct graph given directed_arc_indices and arc_tags
    # key: id_of_point
    # value: a list of tuples -> [(id_of_head1, label),(id_of_head2, label)，...]
    for arc, arc_tag in zip(directed_arc_indices, arc_tags):
        graph[arc[0]].append((arc[1], arc_tag))

    N = len(graph)  # N-1 point, 1 root point

    # i:head_point j:child_point
    top_down_graph = [[] for i in range(N)]  # N-1 real point, 1 root point => N point

    # i:child_point j:head_point ->Bool
    # partial graph during construction
    sub_graph = [[False for i in range(N)] for j in range(N)]

    for i in range(N):
        for head_tuple_of_point_i in graph[i]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)

    actions = []
    stack = [0]
    buffer = []
    deque = []

    for i in range(N - 1, 0, -1):
        buffer.append(i)

    # return if w1 is one head of w0
    def has_head(w0, w1):
        if w0 <= 0:
            return False
        for w in graph[w0]:
            if w[0] == w1:
                return True
        return False

    def has_unfound_child(w):
        for child in top_down_graph[w]:
            if not sub_graph[child][w]:
                return True
        return False

    # return if w has other head except the present one
    def has_other_head(w):
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num + 1 < len(graph[w]):
            return True
        return False

    # return if w has any unfound head
    def lack_head(w):
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num < len(graph[w]):
            return True
        return False

    # return if w has any unfound child in stack sigma
    # !!! except the top in stack
    def has_other_child_in_stack(stack, w):
        if w <= 0:
            return False
        for c in top_down_graph[w]:
            if c in stack \
                    and c != stack[-1] \
                    and not sub_graph[c][w]:
                return True
        return False

    # return if w has any unfound head in stack sigma
    # !!! except the top in stack
    def has_other_head_in_stack(stack, w):
        if w <= 0:
            return False
        for h in graph[w]:
            if h[0] in stack \
                    and h[0] != stack[-1] \
                    and not sub_graph[w][h[0]]:
                return True
        return False

    # return the relation between child: w0, head: w1
    def get_arc_label(w0, w1):
        for h in graph[w0]:
            if h[0] == w1:
                return h[1]

    def get_oracle_actions_onestep(sub_graph, stack, buffer, deque, actions):
        b0 = buffer[-1] if len(buffer) > 0 else -1
        s0 = stack[-1] if len(stack) > 0 else -1

        if s0 > 0 and has_head(s0, b0):
            if not has_unfound_child(s0) and not has_other_head(s0):
                actions.append("LR:" + get_arc_label(s0, b0))
                stack.pop()
                sub_graph[s0][b0] = True
                return
            else:
                actions.append("LP:" + get_arc_label(s0, b0))
                deque.append(stack.pop())
                sub_graph[s0][b0] = True
                return

        elif s0 >= 0 and has_head(b0, s0):
            if not has_other_child_in_stack(stack, b0) and not has_other_head_in_stack(stack, b0):
                actions.append("RS:" + get_arc_label(b0, s0))
                while len(deque) != 0:
                    stack.append(deque.pop())
                stack.append(buffer.pop())
                sub_graph[b0][s0] = True
                return

            elif s0 > 0:
                actions.append("RP:" + get_arc_label(b0, s0))
                deque.append(stack.pop())
                sub_graph[b0][s0] = True
                return

        elif len(buffer) != 0 and not has_other_head_in_stack(stack, b0) \
                and not has_other_child_in_stack(stack, b0):
            actions.append("NS")
            while len(deque) != 0:
                stack.append(deque.pop())
            stack.append(buffer.pop())
            return

        elif s0 > 0 and not has_unfound_child(s0) and not lack_head(s0):
            actions.append("NR")
            stack.pop()
            return

        elif s0 > 0:
            actions.append("NP")
            deque.append(stack.pop())
            return

        else:
            actions.append('-E-')
            print('"error in oracle!"')
            return

    while len(buffer) != 0:
        get_oracle_actions_onestep(sub_graph, stack, buffer, deque, actions)

    return actions