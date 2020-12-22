import json
from .data_models import Article
from .text_preprocessor import TextPreprocessor
from . import utils

import regex as re
from html.parser import HTMLParser


class PrepWikipedia:
    # Taken from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/prep_wikipedia.py

    PARSER = HTMLParser()
    BLACKLIST = set(['23443579', '52643645'])  # Conflicting disambig. pages

    @staticmethod
    def preprocess(article):
        # Take out HTML escaping WikiExtractor didn't clean
        for k, v in article.items():
            article[k] = PrepWikipedia.PARSER.unescape(v)

        # Filter some disambiguation pages not caught by the WikiExtractor
        if article['id'] in PrepWikipedia.BLACKLIST:
            return None
        if '(disambiguation)' in article['title'].lower():
            return None
        if '(disambiguation page)' in article['title'].lower():
            return None

        # Take out List/Index/Outline pages (mostly links)
        if re.match(r'(List of .+)|(Index of .+)|(Outline of .+)',
                    article['title']):
            return None

        # Return doc with `id` set to `title`
        return article


class InputParser:
    def __init__(self, *, num_partitions, debug_save):
        self.text_preprocessor = TextPreprocessor()
        self.num_partitions = num_partitions
        self.debug_save = debug_save

    def _load_json(self, json_str):
        dct = json.loads(json_str)
        return dct

    def _convert_jsonl_to_prdd(self, rdd):
        json_rdd = rdd.map(self._load_json)
        return json_rdd.map(lambda x: (x['id'], x))

    def _process_row(self, row):
        raw_row = row

        preprocessed_row = PrepWikipedia.preprocess(raw_row)
        if not preprocessed_row:
            return None

        article = Article()

        for k in ['text', 'title']:
            raw_row[k] = self.text_preprocessor.unicode_normalize(raw_row[k])

        article.import_from(raw_row)

        # article.tok = ' '.join(self.text_preprocessor.word_tokenize(article.text))

        sentence_str_lst = self.text_preprocessor.sent_tokenize(article.text, article.title)

        sentence_structs = []
        for sent_str in sentence_str_lst:
            ents, noun_chunks = self.text_preprocessor.compute_ner_and_noun_chunks(sent_str)

            if len(ents) + len(noun_chunks) == 0:
                # discard sentences that have neither
                continue

            sentence_structs.append(dict(
                id=utils.random_str(16),
                text=sent_str,
                noun_chunks=noun_chunks,
                ents=ents))

        article.sents = sentence_structs

        return article

    def tokenize_and_perform_rollup(self, raw_rdd):
        corpus_rdd = raw_rdd.map(self._load_json).repartition(self.num_partitions)

        rollup_rdd = corpus_rdd.map(lambda x: self._process_row(x)).filter(lambda x: x is not None)
        return rollup_rdd
