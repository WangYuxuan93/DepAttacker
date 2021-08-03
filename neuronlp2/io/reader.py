__author__ = 'max'

from neuronlp2.io.instance import DependencyInstance, NERInstance
from neuronlp2.io.instance import Sentence
from neuronlp2.io.common import ROOT, ROOT_POS, ROOT_CHAR, ROOT_TYPE, END, END_POS, END_CHAR, END_TYPE
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH
import re

class CoNLLXReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, 
                pre_alphabet=None, pos_idx=4):
        self.__source_file = open(file_path, 'r',encoding="utf-8")
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet
        self.__pre_alphabet = pre_alphabet
        self.pos_idx = pos_idx

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            if line.startswith('#'):
                line = self.__source_file.readline()
                continue
            items = re.split('[\t \s]+',line)
            if re.match('[0-9]+[-.][0-9]+', items[0]):
                line = self.__source_file.readline()
                continue
            lines.append(items)
            if len(items)<2:
                print("zhilinError: %s"%line)
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []
        if self.__pre_alphabet:
            pres = []
            pre_ids = []
        else:
            pres = None
            pre_ids = None

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            heads.append(0)
            if self.__pre_alphabet:
                pres.append(ROOT)
                pre_ids.append(self.__pre_alphabet.get_index(ROOT))

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)
            word = tokens[1]
            pos = tokens[self.pos_idx]
            head = int(tokens[6])
            type = tokens[7]
            
            # save original word in words (data['SRC']), to recover this for normalize_digits=True
            words.append(word)
            word = DIGIT_RE.sub("0", word) if normalize_digits else word
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

            if self.__pre_alphabet:
                pres.append(word)
                id = self.__pre_alphabet.get_index(word)
                if id == 0:
                    id = self.__pre_alphabet.get_index(word.lower())
                pre_ids.append(id)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)
            if self.__pre_alphabet:
                pres.append(END)
                pre_ids.append(self.__pre_alphabet.get_index(END))

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs, pres=pres, pre_ids=pre_ids, lines=lines), postags, pos_ids, heads, types, type_ids)


class CoNLL03Reader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        chunk_tags = []
        chunk_ids = []
        ner_tags = []
        ner_ids = []

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[2]
            chunk = tokens[3]
            ner = tokens[4]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            chunk_tags.append(chunk)
            chunk_ids.append(self.__chunk_alphabet.get_index(chunk))

            ner_tags.append(ner)
            ner_ids.append(self.__ner_alphabet.get_index(ner))

        return NERInstance(Sentence(words, word_ids, char_seqs, char_id_seqs),
                           postags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class CoNLLXReaderSDP(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, pre_alphabet=None, pos_idx=4):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet
        self.__pre_alphabet = pre_alphabet
        self.pos_idx = pos_idx

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while (len(line) > 0 and len(line.strip()) == 0) or line.startswith('#'):
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            if line.startswith('#'):
                line = self.__source_file.readline()
                continue
            items = line.split()
            if re.match('[0-9]+[-.][0-9]+', items[0]):
                line = self.__source_file.readline()
                continue
            lines.append(items)
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []
        if self.__pre_alphabet:
            pres = []
            pre_ids = []
        else:
            pres = None
            pre_ids = None

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            types.append([ROOT_TYPE]) # Jeffrey: heads and types should be a list
            type_ids.append([self.__type_alphabet.get_index(ROOT_TYPE)])
            heads.append([0])
            if self.__pre_alphabet:
                pres.append(ROOT)
                pre_ids.append(self.__pre_alphabet.get_index(ROOT))

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)
            word = tokens[1]
            pos = tokens[self.pos_idx]
            headlist = []
            typelist = []
            for x in tokens[8].split("|"):
                if x != '_':
                    p = x.split(":")
                    headlist.append(int(p[0]))
                    typelist.append(p[1])

            # save original word in words (data['SRC']), to recover this for normalize_digits=True
            words.append(word)
            word = DIGIT_RE.sub("0", word) if normalize_digits else word
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(typelist)
            #  exception:
            temp=[]
            for type in typelist:
                try:
                    temp_type = self.__type_alphabet.get_index(type)
                    temp.append(temp_type)
                except:
                    temp_type = self.__type_alphabet.get_index(ROOT_TYPE)  # Jeffrey type不存在的情况
                    # temp_type = self.__type_alphabet.next_index
                    # self.__type_alphabet.next_index +=1
                    print("【ERROR arc_type:%s】"%type)
                    temp.append(temp_type)
            type_ids.append(temp)
            # type_ids.append([self.__type_alphabet.get_index(type) for type in typelist])

            heads.append(headlist)

            if self.__pre_alphabet:
                pres.append(word)
                id = self.__pre_alphabet.get_index(word)
                if id == 0:
                    id = self.__pre_alphabet.get_index(word.lower())
                pre_ids.append(id)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)
            if self.__pre_alphabet:
                pres.append(END)
                pre_ids.append(self.__pre_alphabet.get_index(END))

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs, pres=pres, pre_ids=pre_ids, lines=lines), postags, pos_ids, heads, types, type_ids)
