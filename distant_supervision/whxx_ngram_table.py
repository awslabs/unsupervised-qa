import toml
import numpy as np

class WhxxNgramTable:
    def __init__(self, counts):
        self.counts = counts
        self.freqs = self._compute_freqs()

        # Taken from https://github.com/explosion/spaCy/blob/146dc2766a069c8fe9e91c801f9695c2e262f742/spacy/lang/en/syntax_iterators.py
        # self.nounchunk_labels = [
        #     "nsubj",
        #     "dobj",
        #     "nsubjpass",
        #     "pcomp",
        #     "pobj",
        #     "dative",
        #     "appos",
        #     "attr",
        #     "ROOT",
        #     "conj",
        # ]


    def __repr__(self):
        return str(self.__dict__)

    def _compute_freqs(self):
        freqs = {}
        for category, ngram2count in self.counts.items():
            ngram2freq = {}
            total = sum(ngram2count.values())

            for ngram, count in ngram2count.items():
                ngram2freq[ngram] = float(count) / total
            freqs[category] = ngram2freq
            assert(np.isclose(sum(ngram2freq.values()), 1.0))
        return freqs

    def rand_sample_ngram(self, np_rng, ner_category):
        if ner_category not in self.freqs:
            return 'what'

        ngram2freq = self.freqs[ner_category]
        ngram = np_rng.choice(list(ngram2freq.keys()), p=list(ngram2freq.values()))
        return ngram

    @classmethod
    def import_from_toml(cls, fptr):
        counts = toml.loads(fptr.read())
        return WhxxNgramTable(counts)
