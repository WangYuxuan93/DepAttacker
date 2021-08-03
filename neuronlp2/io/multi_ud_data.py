__author__ = 'max'

import os.path
import numpy as np
from collections import defaultdict, OrderedDict
import torch
import re

from neuronlp2.io.reader import CoNLLXReader
from neuronlp2.io.alphabet import Alphabet
from neuronlp2.io.logger import get_logger
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io.common import ROOT, END, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE

# Special vocabulary symbols - we always put them at the start.
_START_VOCAB = [PAD, ROOT, END]
NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 140]


def create_alphabets(alphabet_directory, train_paths, data_paths=None, max_vocabulary_size=100000, embedd_dict=None,
                     min_occurrence=1, normalize_digits=True, pos_idx=4):

    def expand_vocab():
        vocab_set = set(vocab_list)
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if line.startswith('#'): continue
                    tokens = line.split('\t')
                    if re.match('[0-9]+[-.][0-9]+', tokens[0]): continue

                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    pos = tokens[pos_idx]
                    type = tokens[7]

                    pos_alphabet.add(pos)
                    type_alphabet.add(type)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', default_value=True, singleton=True)
    char_alphabet = Alphabet('character', default_value=True)
    pos_alphabet = Alphabet('pos')
    type_alphabet = Alphabet('type')
    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)

        char_alphabet.add(ROOT_CHAR)
        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)

        char_alphabet.add(END_CHAR)
        pos_alphabet.add(END_POS)
        type_alphabet.add(END_TYPE)

        vocab = defaultdict(int)
        for set_id, train_path in enumerate(train_paths):
            with open(train_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if line.startswith('#'): continue
                    tokens = line.split('\t')
                    if re.match('[0-9]+[-.][0-9]+', tokens[0]): continue

                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    vocab[word] += 1

                    pos = tokens[pos_idx]
                    pos_alphabet.add(pos)

                    type = tokens[7]
                    type_alphabet.add(type)
        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurrence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            assert isinstance(embedd_dict, OrderedDict)
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurrence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurrence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())
    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet


def read_data(source_paths: [str], word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
              pre_alphabet=None, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False,
              mask_out_root=False, pos_idx=4, lans=['en'], lan_alphabet=None):
    all_tensors = []
    data_sizes = []
    for set_id, source_path in enumerate(source_paths):
        data = []
        max_length = 0
        max_char_length = 0
        counter = 0
        src_words = []
        err_types = []
        print('Reading language %s data from %s' % (lans[set_id], source_path))
        reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, 
                              pre_alphabet=pre_alphabet, pos_idx=pos_idx)
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
        while inst is not None and (not max_size or counter < max_size):
            counter += 1
            if counter % 10000 == 0:
                print("reading data: %d" % counter)

            sent = inst.sentence
            #print (inst.sentence.words)
            data.append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, sent.pre_ids])
            src_words.append(sent.words)
            err_types.append([line[9] for line in sent.lines])
            max_len = max([len(char_seq) for char_seq in sent.char_seqs])
            if max_char_length < max_len:
                max_char_length = max_len
            if max_length < inst.length():
                max_length = inst.length()
            inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
        reader.close()

        print("Total number of data: %d" % counter)

        data_size = len(data)
        data_sizes.append(data_size)
        char_length = min(MAX_CHAR_LENGTH, max_char_length)
        wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        hid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        tid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        preid_inputs = np.empty([data_size, max_length], dtype=np.int64)

        masks = np.zeros([data_size, max_length], dtype=np.float32)
        single = np.zeros([data_size, max_length], dtype=np.int64)
        lengths = np.empty(data_size, dtype=np.int64)

        for i, inst in enumerate(data):
            wids, cid_seqs, pids, hids, tids, preids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            if pre_alphabet:
                preid_inputs[i, :inst_size] = preids
                preid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            if symbolic_end:
                # mask out the end token
                masks[i, :inst_size-1] = 1.0
            else:
                masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1
        if mask_out_root:
            masks[:,0] = 0

        words = torch.from_numpy(wid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks = torch.from_numpy(masks)
        single = torch.from_numpy(single)
        lengths = torch.from_numpy(lengths)
        pres = torch.from_numpy(preid_inputs)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'PRETRAINED': pres, 
                       'LANG': lan_alphabet.get_index(lans[set_id]),
                       'SRC': np.array(src_words), 'ERR_TYPE': np.array(err_types)}
        all_tensors.append(data_tensor)
    return all_tensors, data_sizes


def read_bucketed_data(source_paths: [str], word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
                       pre_alphabet=None, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False,
                       mask_out_root=False, pos_idx=4, lans=['en'], lan_alphabet=None):
    
    all_tensors = []
    all_bucket_sizes = []
    for set_id, source_path in enumerate(source_paths):
        data = [[] for _ in _buckets]
        max_char_length = [0 for _ in _buckets]
        counter = 0
        src_words = [[] for _ in _buckets]
        err_types = [[] for _ in _buckets]
        print('Reading language %s data from %s' % (lans[set_id], source_path))
        reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, 
                              pre_alphabet=pre_alphabet, pos_idx=pos_idx)
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
        while inst is not None and (not max_size or counter < max_size):
            counter += 1
            if counter % 10000 == 0:
                print("reading data: %d" % counter)

            inst_size = inst.length()
            sent = inst.sentence
            for bucket_id, bucket_size in enumerate(_buckets):
                if inst_size < bucket_size:
                    data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, sent.pre_ids])
                    src_words[bucket_id].append(sent.words)
                    err_types[bucket_id].append([line[9] for line in sent.lines])
                    max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                    if max_char_length[bucket_id] < max_len:
                        max_char_length[bucket_id] = max_len
                    break

            inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
        reader.close()

        print("Total number of data: %d" % counter)

        bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
        all_bucket_sizes.append(bucket_sizes)
        data_tensors = []
        for bucket_id in range(len(_buckets)):
            bucket_size = bucket_sizes[bucket_id]
            if bucket_size == 0:
                data_tensors.append((1, 1))
                continue

            bucket_length = _buckets[bucket_id]
            char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id])
            wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
            pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            preid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

            masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
            single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
            lengths = np.empty(bucket_size, dtype=np.int64)

            for i, inst in enumerate(data[bucket_id]):
                wids, cid_seqs, pids, hids, tids, preids = inst
                inst_size = len(wids)
                lengths[i] = inst_size
                # word ids
                wid_inputs[i, :inst_size] = wids
                wid_inputs[i, inst_size:] = PAD_ID_WORD
                if pre_alphabet:
                    preid_inputs[i, :inst_size] = preids
                    preid_inputs[i, inst_size:] = PAD_ID_WORD
                for c, cids in enumerate(cid_seqs):
                    cid_inputs[i, c, :len(cids)] = cids
                    cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
                cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
                # pos ids
                pid_inputs[i, :inst_size] = pids
                pid_inputs[i, inst_size:] = PAD_ID_TAG
                # type ids
                tid_inputs[i, :inst_size] = tids
                tid_inputs[i, inst_size:] = PAD_ID_TAG
                # heads
                hid_inputs[i, :inst_size] = hids
                hid_inputs[i, inst_size:] = PAD_ID_TAG
                # masks
                if symbolic_end:
                    # mask out the end token
                    masks[i, :inst_size-1] = 1.0
                else:
                    masks[i, :inst_size] = 1.0
                for j, wid in enumerate(wids):
                    if word_alphabet.is_singleton(wid):
                        single[i, j] = 1
            if mask_out_root:
                masks[:,0] = 0

            words = torch.from_numpy(wid_inputs)
            chars = torch.from_numpy(cid_inputs)
            pos = torch.from_numpy(pid_inputs)
            heads = torch.from_numpy(hid_inputs)
            types = torch.from_numpy(tid_inputs)
            masks = torch.from_numpy(masks)
            single = torch.from_numpy(single)
            lengths = torch.from_numpy(lengths)
            pres = torch.from_numpy(preid_inputs)

            data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                           'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'PRETRAINED': pres,
                           'SRC': np.array(src_words[bucket_id], dtype=object), 'ERR_TYPE': np.array(err_types[bucket_id]),
                           'LANG': lan_alphabet.get_index(lans[set_id])}
            data_tensors.append(data_tensor)
        all_tensors.append(data_tensors)
    return all_tensors, all_bucket_sizes
