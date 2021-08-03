__author__ = 'max'

import os.path
import numpy as np
from collections import defaultdict, OrderedDict
import torch
import re

from neuronlp2.io.reader import CoNLLXReader
from neuronlp2.io.reader_daniel import CoNLLXReaderSDP
from neuronlp2.io.alphabet import Alphabet
from neuronlp2.io.logger import get_logger
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io.common import ROOT, END, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE

# Special vocabulary symbols - we always put them at the start.
_START_VOCAB = [PAD, ROOT, END]
NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 140]


def _generate_stack_inputs(heads, types, prior_order):
    # child_ids = _obtain_child_index_for_left2right(heads)

    debug = False

    stacked_heads = []
    stacked_types = []
    children = []  # [0 for _ in range(len(heads)-1)]
    siblings = []
    previous = []
    next = []
    skip_connect = []
    prev = [0 for _ in range(len(heads))]
    sibs = [0 for _ in range(len(heads))]
    # newheads = [-1 for _ in range(len(heads))]
    # newheads[0]=0
    # stack = [0]
    stack = [1]
    position = 1

    grandpa = [0, 0]
    previous_head = []
    previous_secondhead = []

    for child in range(len(heads)):
        # child 为token的序号
        if child == 0: continue  # 虚拟root节点忽略

        for h in heads[child]:  # ordered_heads:
            stacked_heads.append(child)
            if child == len(heads) - 1:
                next.append(0)
            else:
                next.append(child + 1)
            previous.append(child - 1)
            # head=heads[child]
            head = h
            # newheads[child]=head
            siblings.append(sibs[head])
            skip_connect.append(prev[head])
            prev[head] = position
            children.append(head)
            sibs[head] = child

            previous_head.append(grandpa[-1])
            previous_secondhead.append(grandpa[-2])
            grandpa.append(head)
            if child == head:
                grandpa = [0, 0]

        for t in types[child]:
            # stacked_types.append(types[child])
            stacked_types.append(t)
        position += 1

    previous = []
    next = []
    for x in previous_head:
        previous.append(x)
    # previous.append(0)
    for x in previous_secondhead:
        next.append(x)

    if debug: exit(0)
    # stacked_heads有歧义，应该为stacked_position 和 stacked_types对应。
    # children每个stacked_position的对于的head
    # siblings ？ skip_connect? previous? next?
    return stacked_heads, children, siblings, stacked_types, skip_connect, previous, next


def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=100000, embedd_dict=None,
                     min_occurrence=1, normalize_digits=True, pos_idx=4, expand_with_pretrained=False,
                     log_name="Create Alphabets", task_type="dp"):
    
    def expand_vocab_with_pretrained():
        logger.info("Expanding word vocab with pretrained words")
        vocab_set = set(vocab_list)
        for word in embedd_dict:
            if word not in vocab_set:
                vocab_set.add(word)
                vocab_list.append(word)

    def expand_vocab():
        vocab_set = set(vocab_list)
        lemma_vocab_set = set(lemma_vocab_list)
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if line.startswith('#'): continue
                    #if re.match('[0-9]+[-.][0-9]+', line): continue
                    tokens = line.split('\t')
                    if re.match('[0-9]+[-.][0-9]+', tokens[0]): continue

                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    pos = tokens[pos_idx]
                    # <<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    if task_type =="dp":
                        type = tokens[7]
                        type_alphabet.add(type)
                    elif task_type=="sdp":
                        type = []
                        for x in tokens[8].split("|"):
                            if x != '_':
                                type.append(x.split(":")[1])
                        for sub_type in type:
                            type_alphabet.add(sub_type)
                    # >>>>>>>>>>>>>>>>>>>> end >>>>>>>>>>>>>>>>>>>>>>>>>>
                    pos_alphabet.add(pos)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)
                    lemma = DIGIT_RE.sub("0", tokens[2]) if normalize_digits else tokens[2]
                    if lemma not in lemma_vocab_set and (lemma in embedd_dict or lemma.lower() in embedd_dict):
                        lemma_vocab_set.add(lemma)
                        lemma_vocab_list.append(lemma)

    logger = get_logger(log_name)
    word_alphabet = Alphabet('word', default_value=True, singleton=True)
    char_alphabet = Alphabet('character', default_value=True)
    pos_alphabet = Alphabet('pos',keep_growing=True)
    type_alphabet = Alphabet('type',keep_growing=True)
    lemma_alphabet = Alphabet('lemma',default_value=True,singleton=True)
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
        lemma_vocab = dict()
        with open(train_path, 'r',encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith('#'): continue
                tokens = line.split('\t')
                if re.match('[0-9]+[-.][0-9]+', tokens[0]): continue
                #if re.match('[0-9]+[-.][0-9]+', line): continue

                for char in tokens[1]:
                    char_alphabet.add(char)

                word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

                pos = tokens[pos_idx]
                pos_alphabet.add(pos)
                # <<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                if task_type == "dp":
                    type = tokens[7]
                    type_alphabet.add(type)
                elif task_type == "sdp":
                    type = []
                    for x in tokens[8].split("|"):
                        if x != '_':
                            type.append(x.split(":")[1])
                    for sub_type in type:
                        type_alphabet.add(sub_type)
                # >>>>>>>>>>>>>>>>>>>> end >>>>>>>>>>>>>>>>>>>>>>>>>>
                # LEMMAS
                lemma = DIGIT_RE.sub("0", tokens[2]) if normalize_digits else tokens[2]
                if lemma in lemma_vocab:
                    lemma_vocab[lemma] += 1
                else:
                    lemma_vocab[lemma] = 1

        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurrence])
        lemma_singletons = set([lemma for lemma, count in lemma_vocab.items() if count <= min_occurrence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            assert isinstance(embedd_dict, OrderedDict)
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurrence
            for lemma in lemma_vocab.keys():
                if lemma in embedd_dict or lemma.lower() in embedd_dict:
                    lemma_vocab[lemma] += min_occurrence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurrence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        lemma_vocab_list = _START_VOCAB + sorted(lemma_vocab, key=lemma_vocab.get, reverse=True)
        logger.info("Total LEMMA Vocabulary Size: %d" % len(lemma_vocab_list))
        logger.info("Total LEMMA Singleton Size:  %d" % len(lemma_singletons))
        lemma_vocab_list = [lemma for lemma in lemma_vocab_list if lemma in _START_VOCAB or lemma_vocab[lemma] > min_occurrence]
        logger.info("Total LEMMA Vocabulary Size (w.o rare words): %d" % len(lemma_vocab_list))

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        if embedd_dict is not None and expand_with_pretrained:
            expand_vocab_with_pretrained()

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        if len(lemma_vocab_list) > max_vocabulary_size:
            lemma_vocab_list = lemma_vocab_list[:max_vocabulary_size]

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))
        for lemma in lemma_vocab_list:
            lemma_alphabet.add(lemma)
            if lemma in lemma_singletons:
                lemma_alphabet.add_singleton(lemma_alphabet.get_index(lemma))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)
        lemma_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)
        lemma_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()
    lemma_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("LEMMA Alphabet Size (Singleton): %d (%d)" % (lemma_alphabet.size(), lemma_alphabet.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())

    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet,lemma_alphabet


def read_data(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
              pre_alphabet=None, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False,
              mask_out_root=False, pos_idx=4):
    data = []
    max_length = 0
    max_char_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, 
                          pre_alphabet=pre_alphabet, pos_idx=pos_idx,keep_growing=True)

    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    src_words = []
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence
        #print (inst.sentence.words)
        data.append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, sent.pre_ids])
        src_words.append(sent.words)
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
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
                   'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'SRC': src_words,
                   'PRETRAINED': pres}
    return data_tensor, data_size


def read_bucketed_data(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
                       pre_alphabet=None, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False,
                       mask_out_root=False, pos_idx=4):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    src_words = [[] for _ in _buckets]
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
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
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
                       'SRC': np.array(src_words[bucket_id],dtype=object)}
        data_tensors.append(data_tensor)
    return data_tensors, bucket_sizes

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<
def read_bucketed_data_sdp(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
                       lemma_alphabet:Alphabet=None,pre_alphabet=None, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False,
                       mask_out_root=False, pos_idx=4,model_type=None):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    src_words = [[] for _ in _buckets]

    reader = CoNLLXReaderSDP(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,lemma_alphabet=lemma_alphabet,
                             pre_alphabet=pre_alphabet, pos_idx=pos_idx,model_type=model_type)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end) # Jeffrey: sentence will be transformed to a instance
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        #print (sent.words)
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                # 默认prior_order=Left2right
                stacked_heads, children, siblings, stacked_types, skip_connect, previous, next = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order=None)
                data[bucket_id].append([sent.word_ids,sent.lemma_ids, sent.char_id_seqs, inst.pos_ids, inst.heads,
                                        inst.type_ids, sent.pre_ids, stacked_heads, children, siblings,
                                        stacked_types, skip_connect, previous, next]) #Jeffrey: bucket principle
                src_words[bucket_id].append(sent.words)
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])  # Jeffrey: record the max sen length in every bucket
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len  # Jeffrey: record the max char length in every bucket
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))] # Jeffrey: sample size in evrey bucket
    data_tensors = []
    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensors.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id])
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        lid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.zeros([bucket_size, bucket_length, 17], dtype=np.int64)
        tid_inputs = np.zeros([bucket_size, bucket_length, 17], dtype=np.int64)
        preid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)



        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lemma_single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths_e = np.empty(bucket_size, dtype=np.int64)


        #for NewStackPtr
        final_length = 17 * (bucket_length - 1)  # 计算transition-base最长的step数
        if bucket_length < 17: final_length = bucket_length * (bucket_length - 1)
        stack_hid_inputs = np.empty([bucket_size, final_length], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, final_length], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, final_length], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, final_length], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, final_length], dtype=np.int64)
        previous_inputs = np.empty([bucket_size, final_length], dtype=np.int64)
        next_inputs = np.empty([bucket_size, final_length], dtype=np.int64)

        masks_d = np.zeros([bucket_size, final_length], dtype=np.float32)
        lengths_d = np.empty(bucket_size, dtype=np.int64)
        for i, inst in enumerate(data[bucket_id]):
            wids,lids,cid_seqs, pids, hids, tids, preids,stack_hids, chids, ssids, stack_tids, skip_ids, previous_ids, next_ids = inst
            inst_size = len(wids)
            lengths_e[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            lid_inputs[i, :inst_size] = lids
            lid_inputs[i, inst_size:] = PAD_ID_WORD
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

            # heads,type ids
            # for h, hid in enumerate(hids):
            #     for kk, x in enumerate(hid):
            #         hid_inputs[i, h, x] = 1
            #         # tid_inputs[i, h, x] = tids[h][kk]
            #     hid_inputs[i, h, inst_size:] = PAD_ID_TAG
            #     tid_inputs[i, h, inst_size:] = PAD_ID_TAG
            for h, hid in enumerate(hids):
                hid_inputs[i,h,:len(hid)] = hid
                hid_inputs[i,h,len(hid):] = PAD_ID_TAG
            for h,tid in enumerate(tids):
                tid_inputs[i,h,:len(tid)] = tid
                tid_inputs[i,h,len(tid):] = PAD_ID_TAG


            # lemma single
            for j, lid in enumerate(lids):
                if lemma_alphabet.is_singleton(lid):
                    lemma_single[i, j] = 1
            # masks
            if symbolic_end:
                # mask out the end token
                masks_e[i, :inst_size-1] = 1.0
            else:
                masks_e[i, :inst_size] = 1.0   # mask the padding

            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            inst_size_decoder = len(stack_hids)
            lengths_d[i] = inst_size_decoder
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # siblings
            ssid_inputs[i, :inst_size_decoder] = ssids
            ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # skip connects
            skip_connect_inputs[i, :inst_size_decoder] = skip_ids
            skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # ADDED
            previous_inputs[i, :inst_size_decoder] = previous_ids
            previous_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            next_inputs[i, :inst_size_decoder] = next_ids
            next_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            masks_d[i, :inst_size_decoder] = 1.0
        if mask_out_root:
            masks[:,0] = 0

        words = torch.from_numpy(wid_inputs)
        lemmas = torch.from_numpy(lid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks = torch.from_numpy(masks_e)
        single = torch.from_numpy(single)
        lemma_single_t = torch.from_numpy(lemma_single)
        lengths = torch.from_numpy(lengths_e)
        pres = torch.from_numpy(preid_inputs)

        stacked_heads = torch.from_numpy(stack_hid_inputs)
        children = torch.from_numpy(chid_inputs)
        siblings = torch.from_numpy(ssid_inputs)
        stacked_types = torch.from_numpy(stack_tid_inputs)
        skip_connect = torch.from_numpy(skip_connect_inputs)
        previous = torch.from_numpy(previous_inputs)
        next = torch.from_numpy(next_inputs)

        masks_d_t = torch.from_numpy(masks_d)
        lengths_d_t = torch.from_numpy(lengths_d)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'PRETRAINED': pres,
                       "Lemma_s":lemma_single_t,"lemmas":lemmas,
                       "stacked_heads":stacked_heads,"children":children,"siblings":siblings,"stacked_types":stacked_types,
                       "skip_connect":skip_connect,"previous":previous,"next":next,"masks_d_t":masks_d_t,"lenghts_d_t":lengths_d_t,
                       'SRC': np.array(src_words[bucket_id],dtype=object)}
        data_tensors.append(data_tensor)
    return data_tensors, bucket_sizes

def read_data_one_by_one(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
                       lemma_alphabet:Alphabet=None,pre_alphabet=None, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False,
                       mask_out_root=False, pos_idx=4,model_type=None):
    data = []
    max_char_length = []
    print('Reading data from %s' % source_path)
    counter = 0
    src_words = []
    word_len = []
    heads_num = []
    reader = CoNLLXReaderSDP(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,lemma_alphabet=lemma_alphabet,
                             pre_alphabet=pre_alphabet, pos_idx=pos_idx,model_type=model_type)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end) # Jeffrey: sentence will be transformed to a instance
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        #print (sent.words)

        # 默认prior_order=Left2right
        stacked_heads, children, siblings, stacked_types, skip_connect, previous, next = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order=None)
        # 统计heads的最大多少个
        heads_num.append(max([len(x) for x in inst.heads]))

        data.append([sent.word_ids,sent.lemma_ids, sent.char_id_seqs, inst.pos_ids, inst.heads,
                                inst.type_ids, sent.pre_ids, stacked_heads, children, siblings,
                                stacked_types, skip_connect, previous, next]) #Jeffrey: bucket principle
        src_words.append(sent.words)
        word_len.append(len(sent.words))
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])  # Jeffrey: record the max sen length in every bucket
        if max_len>MAX_CHAR_LENGTH:
            max_len = MAX_CHAR_LENGTH
        max_char_length.append(max_len)
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))] # Jeffrey: sample size in evrey bucket
    data_tensors = []
    for ind in range(len(data)):

        wid_inputs = np.empty([1,word_len[ind]], dtype=np.int64)
        lid_inputs = np.empty([1,word_len[ind]], dtype=np.int64)
        cid_inputs = np.empty([1,word_len[ind], max_char_length[ind]], dtype=np.int64)
        pid_inputs = np.empty([1,word_len[ind]], dtype=np.int64)
        hid_inputs = np.zeros([1,word_len[ind], heads_num[ind]], dtype=np.int64)
        tid_inputs = np.zeros([1,word_len[ind], heads_num[ind]], dtype=np.int64)
        preid_inputs = np.empty([1,word_len[ind]], dtype=np.int64)



        masks_e = np.zeros([1,word_len[ind]], dtype=np.float32)
        single = np.zeros([1,word_len[ind]], dtype=np.int64)
        lemma_single = np.zeros([1,word_len[ind]], dtype=np.int64)
        lengths_e = np.empty(1, dtype=np.int64)


        #for NewStackPtr
        final_length =  heads_num[ind] * (word_len[ind] - 1)  # 计算transition-base最长的step数
        stack_hid_inputs = np.empty([1, final_length], dtype=np.int64)
        chid_inputs = np.empty([1, final_length], dtype=np.int64)
        ssid_inputs = np.empty([1, final_length], dtype=np.int64)
        stack_tid_inputs = np.empty([1, final_length], dtype=np.int64)
        skip_connect_inputs = np.empty([1, final_length], dtype=np.int64)
        previous_inputs = np.empty([1, final_length], dtype=np.int64)
        next_inputs = np.empty([1, final_length], dtype=np.int64)

        masks_d = np.zeros([1, final_length], dtype=np.float32)
        lengths_d = np.empty(1, dtype=np.int64)

        wids,lids,cid_seqs, pids, hids, tids, preids,stack_hids, chids, ssids, stack_tids, skip_ids, previous_ids, next_ids = data[ind]
        inst_size = len(wids)
        lengths_e[0] = inst_size
        # word ids
        wid_inputs[0, :inst_size] = wids

        lid_inputs[0, :inst_size] = lids
        if pre_alphabet:
            preid_inputs = preids
        for c, cids in enumerate(cid_seqs):
            cid_inputs[0, c, :len(cids)] = cids
            cid_inputs[0, c, len(cids):] = PAD_ID_CHAR

        # pos ids
        pid_inputs[0, :inst_size] = pids


            # heads,type ids
            # for h, hid in enumerate(hids):
            #     for kk, x in enumerate(hid):
            #         hid_inputs[i, h, x] = 1
            #         # tid_inputs[i, h, x] = tids[h][kk]
            #     hid_inputs[i, h, inst_size:] = PAD_ID_TAG
            #     tid_inputs[i, h, inst_size:] = PAD_ID_TAG
        for h, hid in enumerate(hids):
            hid_inputs[0,h,:len(hid)] = hid
            hid_inputs[0,h,len(hid):] = PAD_ID_TAG
        for h,tid in enumerate(tids):
            tid_inputs[0,h,:len(tid)] = tid
            tid_inputs[0,h,len(tid):] = PAD_ID_TAG


        # lemma single
        for j, lid in enumerate(lids):
            if lemma_alphabet.is_singleton(lid):
                lemma_single[0, j] = 1
        # masks
        if symbolic_end:
            # mask out the end token
            masks_e[0, :inst_size-1] = 1.0
        else:
            masks_e[0, :inst_size] = 1.0   # mask the padding

        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[0, j] = 1

        inst_size_decoder = len(stack_hids)
        lengths_d[0] = inst_size_decoder
        stack_hid_inputs[0, :inst_size_decoder] = stack_hids
        stack_hid_inputs[0, inst_size_decoder:] = PAD_ID_TAG
        # children
        chid_inputs[0, :inst_size_decoder] = chids
        chid_inputs[0, inst_size_decoder:] = PAD_ID_TAG
        # siblings
        ssid_inputs[0, :inst_size_decoder] = ssids
        ssid_inputs[0, inst_size_decoder:] = PAD_ID_TAG
        # stacked types
        stack_tid_inputs[0, :inst_size_decoder] = stack_tids
        stack_tid_inputs[0, inst_size_decoder:] = PAD_ID_TAG
        # skip connects
        skip_connect_inputs[0, :inst_size_decoder] = skip_ids
        skip_connect_inputs[0, inst_size_decoder:] = PAD_ID_TAG
        # ADDED
        previous_inputs[0, :inst_size_decoder] = previous_ids
        previous_inputs[0, inst_size_decoder:] = PAD_ID_TAG
        next_inputs[0, :inst_size_decoder] = next_ids
        next_inputs[0, inst_size_decoder:] = PAD_ID_TAG
        masks_d[0, :inst_size_decoder] = 1.0
        if mask_out_root:
            masks_e[:,0] = 0

        words = torch.from_numpy(wid_inputs)
        lemmas = torch.from_numpy(lid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks = torch.from_numpy(masks_e)
        single = torch.from_numpy(single)
        lemma_single_t = torch.from_numpy(lemma_single)
        lengths = torch.from_numpy(lengths_e)
        pres = torch.from_numpy(preid_inputs)

        stacked_heads = torch.from_numpy(stack_hid_inputs)
        children = torch.from_numpy(chid_inputs)
        siblings = torch.from_numpy(ssid_inputs)
        stacked_types = torch.from_numpy(stack_tid_inputs)
        skip_connect = torch.from_numpy(skip_connect_inputs)
        previous = torch.from_numpy(previous_inputs)
        next = torch.from_numpy(next_inputs)

        masks_d_t = torch.from_numpy(masks_d)
        lengths_d_t = torch.from_numpy(lengths_d)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'PRETRAINED': pres,
                       "Lemma_s":lemma_single_t,"lemmas":lemmas,
                       "stacked_heads":stacked_heads,"children":children,"siblings":siblings,"stacked_types":stacked_types,
                       "skip_connect":skip_connect,"previous":previous,"next":next,"masks_d_t":masks_d_t,"lenghts_d_t":lengths_d_t,
                       'SRC': np.array(src_words[ind],dtype=object)}
        data_tensors.append(data_tensor)
    return data_tensors, bucket_sizes

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<
def read_data_sdp(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
              lemma_alphabet:Alphabet=None,pre_alphabet=None, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False,
              mask_out_root=False, pos_idx=4,model_type=None):
    data = []
    max_length = 0
    max_char_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReaderSDP(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,lemma_alphabet=lemma_alphabet,
                          pre_alphabet=pre_alphabet, pos_idx=pos_idx,model_type=model_type)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    src_words = []
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence
        #print (inst.sentence.words)
        stacked_heads, children, siblings, stacked_types, skip_connect, previous, next = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order=None)
        data.append([sent.word_ids,sent.lemma_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, sent.pre_ids,stacked_heads, children, siblings, stacked_types, skip_connect, previous,
                     next])
        src_words.append(sent.words)
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
    char_length = min(MAX_CHAR_LENGTH, max_char_length)
    wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    lid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    hid_inputs = np.zeros([data_size, max_length,17], dtype=np.int64)  # Jeffrey: 由empty 改成zeros
    tid_inputs = np.zeros([data_size, max_length,17], dtype=np.int64)
    lemma_single = np.zeros([data_size, max_length], dtype=np.int64)

    preid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    masks = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths = np.empty(data_size, dtype=np.int64)

    final_length = 17 * (max_length - 1)  # 计算transition-base最长的step数
    if max_length < 17: final_length = max_length * (max_length - 1)
    stack_hid_inputs = np.empty([data_size, final_length], dtype=np.int64)
    chid_inputs = np.empty([data_size, final_length], dtype=np.int64)
    ssid_inputs = np.empty([data_size, final_length], dtype=np.int64)
    stack_tid_inputs = np.empty([data_size, final_length], dtype=np.int64)
    skip_connect_inputs = np.empty([data_size, final_length], dtype=np.int64)
    previous_inputs = np.empty([data_size, final_length], dtype=np.int64)
    next_inputs = np.empty([data_size, final_length], dtype=np.int64)

    masks_d = np.zeros([data_size, final_length], dtype=np.float32)
    lengths_d = np.empty(data_size, dtype=np.int64)

    for i, inst in enumerate(data):

        wids, lids,cid_seqs, pids, hids, tids, preids,stack_hids, chids, ssids, stack_tids, skip_ids, previous_ids, next_ids = inst
        inst_size = len(wids)
        lengths[i] = inst_size
        # word ids
        wid_inputs[i, :inst_size] = wids
        wid_inputs[i, inst_size:] = PAD_ID_WORD
        lid_inputs[i, :inst_size] = lids
        lid_inputs[i, inst_size:] = PAD_ID_WORD
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
        # type ids ,heads
        for h, hid in enumerate(hids):
            hid_inputs[i, h, :len(hid)] = hid
            hid_inputs[i, h, len(hid):] = PAD_ID_TAG
        for h, tid in enumerate(tids):
            tid_inputs[i, h, :len(tid)] = tid
            tid_inputs[i, h, inst_size:] = PAD_ID_TAG
        # lemma single
        for j, lid in enumerate(lids):
            if lemma_alphabet.is_singleton(lid):
                lemma_single[i, j] = 1
        # masks
        if symbolic_end:
            # mask out the end token
            masks[i, :inst_size-1] = 1.0
        else:
            masks[i, :inst_size] = 1.0
        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[i, j] = 1

        inst_size_decoder = len(stack_hids)
        lengths_d[i] = inst_size_decoder
        stack_hid_inputs[i, :inst_size_decoder] = stack_hids
        stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # children
        chid_inputs[i, :inst_size_decoder] = chids
        chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # siblings
        ssid_inputs[i, :inst_size_decoder] = ssids
        ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # stacked types
        stack_tid_inputs[i, :inst_size_decoder] = stack_tids
        stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # skip connects
        skip_connect_inputs[i, :inst_size_decoder] = skip_ids
        skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # ADDED
        previous_inputs[i, :inst_size_decoder] = previous_ids
        previous_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        next_inputs[i, :inst_size_decoder] = next_ids
        next_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        masks_d[i, :inst_size_decoder] = 1.0

    if mask_out_root:
        masks[:,0] = 0

    words = torch.from_numpy(wid_inputs)
    lemmas = torch.from_numpy(lid_inputs)
    chars = torch.from_numpy(cid_inputs)
    pos = torch.from_numpy(pid_inputs)
    heads = torch.from_numpy(hid_inputs)
    types = torch.from_numpy(tid_inputs)
    masks = torch.from_numpy(masks)
    single = torch.from_numpy(single)
    lengths = torch.from_numpy(lengths)
    pres = torch.from_numpy(preid_inputs)
    lemma_single_t = torch.from_numpy(lemma_single)

    stacked_heads = torch.from_numpy(stack_hid_inputs)
    children = torch.from_numpy(chid_inputs)
    siblings = torch.from_numpy(ssid_inputs)
    stacked_types = torch.from_numpy(stack_tid_inputs)
    skip_connect = torch.from_numpy(skip_connect_inputs)
    previous = torch.from_numpy(previous_inputs)
    next = torch.from_numpy(next_inputs)

    masks_d_t = torch.from_numpy(masks_d)
    lengths_d_t = torch.from_numpy(lengths_d)
    data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'PRETRAINED': pres,
                       "Lemma_s":lemma_single_t,"lemmas":lemmas,
                       "stacked_heads":stacked_heads,"children":children,"siblings":siblings,"stacked_types":stacked_types,
                       "skip_connect":skip_connect,"previous":previous,"next":next,"masks_d_t":masks_d_t,"lenghts_d_t":lengths_d_t,
                       'SRC': src_words}
    return data_tensor, data_size
