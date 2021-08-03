try:
    import torch
    import torch.nn.functional as F
except ImportError:
    # No installation required if not using this function
    pass

from adversary.lm.filtering import *

class LanguageModels:
    def __init__(self, device=None, temperature=1.0, top_k=100, top_p=0.01, cache=True):
        try:
            self.device = 'cuda' if device is None and torch.cuda.is_available() else device
        except NameError:
            raise ImportError('Missed torch, transformers libraries. Install it via '
                              '`pip install torch transformers`')
        self.cache = cache
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def clean(self, text):
        return text.strip()

    def predict(self, text, target_word=None, n=1):
        raise NotImplementedError

    @classmethod
    def control_randomness(cls, logits, seed):
        temperature = seed['temperature']
        if temperature is not None:
            return logits / temperature
        return logits

    @classmethod
    def filtering(cls, logits, seed):
        top_k = seed['top_k']
        top_p = seed['top_p']

        if top_k is not None and 0 < top_k < len(logits):
            logits, idxes = filter_top_k(
                logits, top_k, -float('Inf'))
        if top_p is not None and 0 < top_p < 1:
            logits, idxes = nucleus_sampling(logits, top_p)

        return logits, idxes

    def pick(self, logits, target_word, n=1):
        candidate_ids, candidate_probas = self.prob_multinomial(logits, n=n+10)
        results = self.get_candidiates(candidate_ids, candidate_probas, target_word, n)

        return results

    def id2token(self, _id):
        raise NotImplementedError()

    def prob_multinomial(self, logits, n):
        # Convert to probability
        probas = F.softmax(logits, dim=-1)

        # Draw candidates
        # top_n_ids = torch.multinomial(probas, num_samples=n, replacement=False).tolist()
        """
        The torch.multinomial is rather slow, see https://github.com/pytorch/pytorch/issues/11931
        After changing it to numpy, the speed can accelerate a lot.
        Speed up: 3.221s/call -> 5e-6s/call
        """
        top_n_ids = np.random.choice(probas.size(0), n, False, probas.cpu().numpy()).tolist()
        
        probas = probas.cpu() if self.device == 'cuda' else probas
        probas = probas.cpu().detach().data.numpy()
        top_n_probas = [probas[_id] for _id in top_n_ids]

        return top_n_ids, top_n_probas

    def is_skip_candidate(self, candidate):
        return False

    def get_candidiates(self, candidate_ids, candidate_probas, target_word=None, n=1):
        # To have random behavior, NO sorting for candidate_probas.
        results = []
        for candidate_id, candidate_proba in zip(candidate_ids, candidate_probas):
            candidate_word = self.id2token(candidate_id)

            if candidate_word == '':
                continue

            if target_word is not None and candidate_word.lower() == target_word.lower():
                continue

            if self.is_skip_candidate(candidate_word):
                continue

            results.append((candidate_word, candidate_proba))

            if len(results) >= n:
                break

        return results
