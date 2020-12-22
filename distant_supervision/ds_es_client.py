from operator import add
import json
from .text_preprocessor import TextPreprocessor
from .utils import ElasticsearchConfig, ElasticsearchMagic
from .data_models import Article

class ESIndexWriter:
    def __init__(self, output_dir, *, es_hosts, es_index_name, debug_save):
        self.output_dir = output_dir
        self.debug_save = debug_save
        self.text_preprocessor = TextPreprocessor()

        self.es_conf = ElasticsearchConfig(
            hosts=es_hosts,
            index_name=es_index_name,
            doc_type='doc')

    def _is_good_sentence(self, sent):
        # remove sentences that are very short (just do it based on number of characters)
        return len(sent) >= 10

    def _wrangle_articles(self, article_rdd):
        """
        Filter for good sentences,etc
        """
        def filter_sents(article):
            new_sents = []
            for sent_obj in article.sents:
                if not self._is_good_sentence(sent_obj.text):
                    continue

                new_sents.append(sent_obj)
            article.sents = new_sents
            return article

        return article_rdd.map(filter_sents)


    def _index_rdd_partition(self, article_lst):
        es_conf = self.es_conf
        es = ElasticsearchMagic.get_instance('singleton', hosts=[es_conf.hosts])

        def gendata():
            for article in article_lst:
                for sent_obj in article.sents:
                    text_body = sent_obj.text

                    yield {
                        '_op_type': 'create',  # `create` will fail on duplicate _id
                        "_index": es_conf.index_name,
                        "_type": es_conf.doc_type,
                        '_id': sent_obj.id,
                        "_source": {
                            'body': text_body,
                            'body_with_title': '{} \n {}'.format(article.title, text_body),
                            'article_id': article.id,
                            'article_title': article.title,
                            'entities': json.dumps(sent_obj.ents),
                            'noun_chunks': json.dumps(sent_obj.noun_chunks),
                        },
                    }

        from elasticsearch6 import helpers
        helpers.bulk(es, gendata(), request_timeout=60)

    def _perform_es_index(self, article_rdd):
        article_rdd.foreachPartition(
            lambda article_lst: self._index_rdd_partition(article_lst))

    def run_job(self, corpus_rdd, metric_fptr):
        total_nb_articles = corpus_rdd.count()
        print('Number of articles in corpus: {}'.format(total_nb_articles), file=metric_fptr)

        article_rdd = corpus_rdd.map(lambda x: Article.deserialize_json(x))
        article_rdd = self._wrangle_articles(article_rdd)

        nb_sentences_to_index = article_rdd.map(lambda x: len(x.sents)).reduce(add)
        print('Number of sentences to index: {}'.format(nb_sentences_to_index), file=metric_fptr)
        metric_fptr.flush()

        # TODO generate some statistics on number of characters

        self._create_es_index()
        self._perform_es_index(article_rdd)

    def _create_es_index(self):
        es_conf = self.es_conf
        es = ElasticsearchMagic.get_instance('singleton', hosts=[es_conf.hosts])

        # delete index if exists
        if es.indices.exists(index=es_conf.index_name):
            es.indices.delete(index=es_conf.index_name)

        settings = {
            "number_of_shards": 9,
            "number_of_replicas": 1,
            "similarity": {
                "default": {
                    "type": "BM25",
                    "k1": 0.1,  # default is 1.2. Value of 0.0 means that it only depends on IDF (not TF).
                    "b": 0.1,  # default is 0.75. Value of 0.0 disables length-normalization.
                }
            },
            "analysis": {
                "filter": {
                    "english_possessive_stemmer": {
                        "name": "possessive_english",
                        "type": "stemmer"
                    },
                    "english_stop": {
                        "stopwords": "_english_",
                        "type": "stop"
                    },
                    "kstem_stemmer": {
                        # kstem is less aggressive than porter, e.g. "dogs" => "dog" in porter, but not in kstem
                        "name": "light_english",
                        "type": "stemmer"
                    },
                    "english_porter_stemmer": {
                        "name": "english",  # porter, see StemmerTokenFilterFactory.java
                        "type": "stemmer"
                    }
                },
                "analyzer": {
                    "porter_eng_analyzer": {
                        # https://stackoverflow.com/questions/33945796/understanding-analyzers-filters-and-queries-in-elasticsearch
                        "filter": [
                            "standard",  # does nothing: https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-standard-tokenfilter.html
                            "asciifolding",
                            "english_possessive_stemmer",
                            "lowercase",
                            "english_stop",
                            "english_porter_stemmer"
                        ],
                        "tokenizer": "standard"
                    },
                    "kstem_eng_analyzer": {
                        "filter": [
                            "standard",
                            "asciifolding",
                            "english_possessive_stemmer",
                            "lowercase",
                            "english_stop",
                            "kstem_stemmer"
                        ],
                        "tokenizer": "standard"
                    },
                    "possessive_english_analyzer": {
                        # no stemming
                        "filter": [
                            "standard",
                            "asciifolding",
                            "english_possessive_stemmer",
                            "lowercase",
                            "english_stop",
                        ],
                        "tokenizer": "standard"
                    },
                    "standard_english_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_"
                    },
                }
            }
        }

        mappings_for_analyzed_text_field = {
            "type": "text",
            "index": True,
            "analyzer": "porter_eng_analyzer",
            "fields": {
                "possessive": {"type": "text", "analyzer": "possessive_english_analyzer"},
                "kstem": {"type": "text", "analyzer": "kstem_eng_analyzer"},
            },
        }

        mappings = {
            "doc": {
                "properties": {
                    "entities": {
                        "type": "text",  # json string
                        "index": False,
                    },
                    "noun_chunks": {
                        "type": "text",  # json string
                        "index": False,
                    },
                    "article_title": {
                        "type": "keyword",
                        "index": False,
                    },
                    "article_id": {
                        "type": "integer",
                        "index": True,
                    },
                    "body": mappings_for_analyzed_text_field,
                    "body_with_title": mappings_for_analyzed_text_field,
                }
            }
        }

        es.indices.create(es_conf.index_name, body=dict(
            mappings=mappings,
            settings=settings))

        es.indices.open(es_conf.index_name)
