__author__ = 'max'
import re
import math

class CoNLL03Writer(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, chunk, predictions, targets, lengths):
        batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__word_alphabet.get_instance(word[i, j])
                p = self.__pos_alphabet.get_instance(pos[i, j])
                ch = self.__chunk_alphabet.get_instance(chunk[i, j])
                tgt = self.__ner_alphabet.get_instance(targets[i, j])
                pred = self.__ner_alphabet.get_instance(predictions[i, j])
                self.__source_file.write('%d %s %s %s %s %s\n' % (j + 1, w, p, ch, tgt, pred))
            self.__source_file.write('\n')


class CoNLLXWriter(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w',encoding="utf-8")

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, head, type, lengths,arc_loss=None, symbolic_root=False, symbolic_end=False, src_words=None, adv_words=None, heads_by_layer=None):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                adv_flag = '_'
                w = self.__word_alphabet.get_instance(word[i, j])
                if w == '<_UNK>' and src_words is not None:
                    w = src_words[i][j]
                # this is to deal with normalize_digits = True
                if re.match(r"\d", w):
                    w = src_words[i][j]
                if adv_words is not None and adv_words[i][j] != w:
                    adv_flag = '['+w+']'
                    w = adv_words[i][j]
                p = self.__pos_alphabet.get_instance(pos[i, j])
                t = self.__type_alphabet.get_instance(type[i, j])
                h = head[i, j]
                proba = "_"
                if arc_loss!=None:
                    proba = str(math.exp(arc_loss[i][h][j].item()))
                if heads_by_layer is not None:
                    layer = '#layer-'+str(heads_by_layer[i, j])
                    self.__source_file.write('%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t%s\t%s\n' % (j, w, p, p, h, t, layer, adv_flag))
                else:
                    if adv_flag !="_":
                        self.__source_file.write('%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t_\t%s\n' % (j, w, p, p, h, t, adv_flag))
                    else:
                        self.__source_file.write('%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t_\t%s\n' % (j, w, p, p, h, t, proba))
            self.__source_file.write('\n')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class CoNLLXWriterSDP(object):

    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()
    # <<<<<<<<<<<<<<<<<<<<<< sdp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def write(self, word, pos, head, type, lengths, symbolic_root=False, symbolic_end=False, src_words=None, adv_words=None, heads_by_layer=None):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                adv_flag = '_'
                w = self.__word_alphabet.get_instance(word[i, j])
                if w == '<_UNK>' and src_words is not None:
                    w = src_words[i][j]
                # this is to deal with normalize_digits = True
                if re.match(r"\d", w):
                    w = src_words[i][j]
                if adv_words is not None and adv_words[i][j] != w:
                    adv_flag = '['+w+']'
                    w = adv_words[i][j]
                p = self.__pos_alphabet.get_instance(pos[i, j])
                first_head = "_"
                first_type = "_"
                written = "_"
                for k in range(0, lengths[i]-end):
                    if head[i,j,k] == 1:
                        if first_head == "_":
                            first_type = self.__type_alphabet.get_instance(type[i, j, k])
                            first_head = str(k)
                            written = ":".join([first_head,first_type])
                        else:
                            t = self.__type_alphabet.get_instance(type[i, j, k])
                            h = str(k)
                            temp_written = ":".join([h, t])
                            written = written+"|"+temp_written

                if heads_by_layer is not None: # TODO: Jeffrey: accustom to sdp
                    layer = '#layer-'+str(heads_by_layer[i, j])
                    self.__source_file.write('%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t%s\t%s\n' % (j, w, p, p, h, t, layer, adv_flag))
                else:
                    self.__source_file.write('%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (j, w, '_', p, p, '_', first_head, first_type, written,'_'))
            self.__source_file.write('\n')