import string
import copy
import unicodedata
import statistics
import re

from nltk.corpus import stopwords as nltk_stopwords
from distant_supervision.utils import SpacyMagic

STOPWORDS = set(nltk_stopwords.words('english'))
PUNCTS = set(string.punctuation)
DISCARD_WORD_SET = STOPWORDS | PUNCTS | set([''])
ULIM_CHAR_PER_SENTENCE = 500

class TextPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def get_phrases(*, entities, noun_chunks):
        """
        :param entities: list of pairs (ent_str, ent_category)
        :param noun_chunks: list of pairs
        """
        phrases = copy.deepcopy(entities)

        ent_str_set = set([ent_str.lower() for ent_str, _ in entities])
        discard_set = ent_str_set | STOPWORDS

        for nc in noun_chunks:
            nc_str, _ = nc  # ensure it's in the correct format (i.e. pairs)
            nc_str_lower = nc_str.lower()

            if nc_str_lower not in discard_set:
                phrases.append(nc)

        return phrases

    @staticmethod
    def unicode_normalize(text):
        """
        Resolve different type of unicode encodings.

        e.g. unicodedata.normalize('NFKD', '\u00A0') will return ' '
        """
        return unicodedata.normalize('NFKD', text)

    @classmethod
    def clean_and_tokenize_str(cls, s):
        tokens = set(re.split(r'\W+', s.lower()))
        tokens = tokens - DISCARD_WORD_SET
        return tokens

    def findall_substr(self, substr, full_str):
        """
        Respect word boundaries
        """
        return re.findall(r'\b{}\b'.format(re.escape(substr)), full_str)

    def is_similar(self, sent1, sent2, f1_cutoff, *, discard_stopwords):
        """
        Based on bag of words.

        :param discard_stopwords: remove stopwords and lowercasing
        """
        if sent1 == sent2:
            return True

        if discard_stopwords:
            tokens1 = self.clean_and_tokenize_str(sent1)
            tokens2 = self.clean_and_tokenize_str(sent2)
        else:
            tokens1 = set(sent1.strip().split())
            tokens2 = set(sent2.strip().split())

        eps = 1e-100

        score1 = float(len(tokens1 & tokens2)) / (len(tokens1) + eps)
        score2 = float(len(tokens1 & tokens2)) / (len(tokens2) + eps)

        f1 = statistics.harmonic_mean([score1, score2])
        return f1 >= f1_cutoff

    def word_tokenize(self, raw_text):
        nlp = SpacyMagic.load_en_disable_all()
        return [w.text for w in nlp(raw_text)]

    def compute_ner(self, text):
        """
        :return: e.g. [('today', 'DATE'), ('Patrick', 'PERSON')]
        """
        nlp = SpacyMagic.load('my_english_ner', 'en_core_web_sm', disable=['tagger', 'parser'])
        ents = [(ent.text, ent.label_) for ent in nlp(text).ents]
        return sorted(set(ents))

    def compute_ner_and_noun_chunks(self, text):
        """
        https://spacy.io/usage/linguistic-features#noun-chunks

        ents: [('today', 'DATE'), ('Patrick', 'PERSON')]
        noun_chunks: e.g. [('Autonomous cars', 'nsubj'), ('insurance liability', 'dobj')]

        :return: (ents, noun_chunks)
        """
        if len(text) > ULIM_CHAR_PER_SENTENCE:
            return [], []

        # spacy has memory leaks: https://github.com/explosion/spaCy/issues/3618
        nlp = SpacyMagic.load('my_english', 'en_core_web_sm', disable=[])
        doc = nlp(text)

        ents = [(ent.text, ent.label_) for ent in doc.ents]
        chunks = [(nc.text, nc.root.dep_) for nc in doc.noun_chunks]

        ents = sorted(set(ents))
        chunks = sorted(set(chunks))

        return ents, chunks

    def normalize_basic(self, text):
        """
        :param text: tokenized text string
        """
        tokens = [w for w in text.lower().split() if w not in DISCARD_WORD_SET]
        return ' ' . join(tokens)

    def sent_tokenize(self, raw_text, title):
        """
        :return: a list of ...
        """
        # There are different types of sentence segmentation. See
        # https://spacy.io/usage/linguistic-features#sbd for more details
        # The sentencizer is much faster, but not as good as DependencyParser
        # Alternatively, nlp = SpacyMagic.load('en_core_web_sm')  # using DependencyParser
        nlp = SpacyMagic.load_en_sentencizer()

        text_lst = re.split(r'[\n\r]+', raw_text)
        if title and text_lst[0] == title:
            # remove the first element if is the same as the title
            text_lst = text_lst[1:]

        sentences_agg = []
        for text in text_lst:
            doc = nlp(text)
            sentences = [sent.string.strip() for sent in doc.sents]
            sentences_agg.extend(sentences)
        return sentences_agg
