# Source: https://arxiv.org/abs/1810.04805

try:
    import torch
    from transformers import BertTokenizer, BertForMaskedLM
except ImportError:
    # No installation required if not using this function
    pass

from adversary.lm.language_models import LanguageModels

class Bert(LanguageModels):
    START_TOKEN = '[CLS]'
    SEPARATOR_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    SUBWORD_PREFIX = '##'

    def __init__(self, model_path='bert-base-cased', temperature=1.0, top_k=None, top_p=None, device='cuda'):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p)
        self.model_path = model_path

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

    def id2token(self, _id):
        # id: integer format
        return self.tokenizer.convert_ids_to_tokens([_id])[0]

    def is_skip_candidate(self, candidate):
        return candidate[:2] == self.SUBWORD_PREFIX

    def predict(self, text, target_word=None, n=1):
        # Prepare inputs
        tokens = self.tokenizer.tokenize(text)
        tokens.insert(0, self.START_TOKEN)
        tokens.append(self.SEPARATOR_TOKEN)
        target_pos = tokens.index(self.MASK_TOKEN)

        token_inputs = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_inputs = [0] * len(token_inputs)
        mask_inputs = [1] * len(token_inputs)  # 1: real token, 0: padding token

        # Convert to feature
        token_inputs = torch.tensor([token_inputs]).to(self.device)
        segment_inputs = torch.tensor([segment_inputs]).to(self.device)
        mask_inputs = torch.tensor([mask_inputs]).to(self.device)

        # Prediction
        with torch.no_grad():
            outputs = self.model(token_inputs, segment_inputs, mask_inputs)
        target_token_logits = outputs[0][0][target_pos]

        # Selection
        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)

        results = self.pick(target_token_logits, target_word=target_word, n=n)
        return results
