import argparse
import json
from collections import OrderedDict
import os, sys
import torch

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from adversary.lm.bert import Bert



def load_conll(f):
    data = []
    sents = f.read().strip().split("\n\n")
    for sent in sents:
        data.append([line.strip().split("\t") for line in sent.strip().split("\n")])
    return data

class MLM_Generator(object):

    def __init__(self, bert_path, device=None, temperature=1.0, top_k=100, top_p=None, n_mlm_cands=50):
        print ("Loading MLM generator from: {}".format(bert_path))
        self.cand_mlm_model = Bert(bert_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p)
        self.cand_mlm_model.model.eval()
        self.n_mlm_cands = n_mlm_cands

    def _get_mlm_cands(self, tokens, idx, n=50):
        original_word = tokens[idx]
        tmps = tokens.copy()
        tmps[idx] = self.cand_mlm_model.MASK_TOKEN
        masked_text = ' '.join(tmps)

        candidates = self.cand_mlm_model.predict(masked_text, target_word=original_word, n=n)

        return [candidate[0] for candidate in candidates]

    def generate(self, tokens, n=50):
        cands_list = []
        for i in range(len(tokens)):
            cands = self._get_mlm_cands(tokens, i, n=n)
            cands_list.append({"orig":tokens[i], "cands":cands})
        return cands_list

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Generating Candidates from BERT')
    args_parser.add_argument('--input', help='path for test file.', required=True)
    args_parser.add_argument('--bert_path', help='path for mlm candidate generating', required=True)
    args_parser.add_argument('--temp', type=float, default=1.0, help='Temperature for mlm candidate generating')
    args_parser.add_argument('--n_mlm_cands', type=int, default=50, help='Select candidate number for mlm candidate generating')
    args_parser.add_argument('--top_k', type=int, default=100, help='Top candidate number for filtering mlm candidate generating')
    args_parser.add_argument('--top_p', type=float, default=None, help='Top proportion for filtering mlm candidate generating')
    args_parser.add_argument('--output', type=str, help='output filename for parse')
    args_parser.add_argument('--cuda',action='store_true')
    args = args_parser.parse_args()

    with open(args.input, 'r') as f:
        data = load_conll(f)
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    generator = MLM_Generator(args.bert_path, device=device, temperature=args.temp, 
                            top_k=args.top_k, top_p=args.top_p, n_mlm_cands=args.n_mlm_cands)

    all_cands = OrderedDict()
    print ("total sent:", len(data))
    for i, sent in enumerate(data):
        if i % 100 == 0:
            print (i, "... ", end="")
            sys.stdout.flush()
        tokens = [line[1] for line in sent]
        cands_list = generator.generate(tokens, args.n_mlm_cands)
        all_cands[i] = cands_list
    json.dump(all_cands, open(args.output, 'w'), indent=4)
