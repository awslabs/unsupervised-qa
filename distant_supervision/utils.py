import string
import random
import toml
import secrets
import numpy as np
from collections import Counter

def read_default_config_toml():
    with open('resources/default_config.toml') as fh:
        return toml.loads(fh.read())

# Spacy isn't serializable but loading it is semi-expensive
class SpacyMagic(object):
    """
    Simple Spacy Magic to minimize loading time.
    >>> SpacyMagic.load("en")
    <spacy.en.English ...
    """
    _spacys = {}
    _counter = Counter()

    @classmethod
    def load(cls, name, lang, **kwargs):
        # e.g. load('my_en', 'en_core_web_sm', disable=['tagger', 'ner'])

        # spacy has memory leaks: https://github.com/explosion/spaCy/issues/3618
        # HACK: reload every K loads (via counter) to prevent leaks

        if name not in cls._spacys or cls._counter[name] % 10000 == 0:
            # only load once per thread
            import spacy
            cls._spacys[name] = spacy.load(lang, **kwargs)

        cls._counter[name] += 1
        return cls._spacys[name]

    @classmethod
    def load_en_disable_all(cls):
        name = 'en_disable_all'
        if name not in cls._spacys:
            import spacy
            nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser'])
            assert(len(nlp.pipeline) == 0)
            assert(len(nlp.pipe_names) == 0)
            cls._spacys[name] = nlp
        return cls._spacys[name]

    @classmethod
    def load_en_sentencizer(cls):
        # https://spacy.io/usage/linguistic-features#sbd-component
        name = 'en_sentencizer'
        if name not in cls._spacys:
            # only load once per thread
            from spacy.lang.en import English
            nlp = English()
            sbd = nlp.create_pipe('sentencizer')   # or: nlp.create_pipe('sbd')
            nlp.add_pipe(sbd)
            cls._spacys[name] = nlp
        return cls._spacys[name]


class ElasticsearchMagic(object):
    _es_objects = {}

    @classmethod
    def get_instance(cls, name, **kwargs):
        if name not in cls._es_objects:
            from elasticsearch6 import Elasticsearch
            cls._es_objects[name] = Elasticsearch(**kwargs)
        return cls._es_objects[name]


class ElasticsearchConfig:
    def __init__(self, *, hosts, index_name, doc_type):
        self.hosts = hosts
        self.index_name = index_name
        self.doc_type = doc_type

class RandomNumberGenerator:
    """
    Need `secrets` for seed because Spark initialize multiple workers at the same time
    that causes multiple RNG or random numbers to be the same!
    """

    def __init__(self):
        self.vanilla = self.get_random_number_generator()
        self.np = self.get_numpy_random_number_generator()

    def get_random_number_generator(self):
        rng = random.Random()
        rng.seed(secrets.randbits(128))
        return rng

    def get_numpy_random_number_generator(self):
        np_rng = np.random.RandomState(seed=secrets.randbits(32))
        return np_rng

def random_str(N):
    alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for i in range(N))

def find_all(a_str, sub):
    # https://stackoverflow.com/questions/4664850/find-all-occurrences-of-a-substring-in-python/19720214
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches
