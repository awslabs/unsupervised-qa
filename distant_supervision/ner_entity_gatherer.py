from .data_models import PhraseObj, PhraseMode
from .stat_computation import StatComputation
from . import utils
from operator import add
from .constants import ANSWER_NUM_CHARS_ULIM, NUM_ARTICLES_PER_ENTITY_LLIM
from logzero import logger as logging
import os

ENTITY_NCHARS_LLIM = 3

def _clean_ners(ner_set):
    """
    Remove entities/answers that are too long
    """
    logging.info('Number of NER entities/answers before filtering: {}'.format(len(ner_set)))

    new_ner_set = set([(phrase_str, phrase_category)
                        for phrase_str, phrase_category in ner_set
                        if len(phrase_str) <= ANSWER_NUM_CHARS_ULIM])

    logging.info('Number of NER entities/answers after filtering: {}'.format(len(new_ner_set)))
    return new_ner_set

def convert_ner_rdd_to_set(ner_rdd, nb_ner_ulim):
    # For NER file, we only use the entity + category information (not the article IDs)
    ner_set = set(ner_rdd.map(lambda x: (x.phrase_str, x.phrase_category)).collect())
    ner_set = _clean_ners(ner_set)

    if nb_ner_ulim:
        ner_lst = list(ner_set)
        rng = utils.RandomNumberGenerator()
        rng.vanilla.shuffle(ner_lst)
        ner_set = set(ner_lst[:nb_ner_ulim])

    return ner_set


class NerEntityGatherer:
    def __init__(self, *, output_dir, phrase_mode):
        self.stat_computation = StatComputation()
        self.output_dir = output_dir
        self.phrase_mode = phrase_mode

    def _get_unique_entity_pairs(self, article):
        """
        :return: a list of pairs [("0 to 6 years", "DATE"), ...]
        """
        phrase_pair_set = set()
        for sent in article.sents:
            phrase_tuple_lst = sent.get_phrases(self.phrase_mode)

            for (phrase_str, phrase_category) in phrase_tuple_lst:
                if len(phrase_str) < ENTITY_NCHARS_LLIM:
                    # only keep entities that have a minimum number of characters
                    continue
                phrase_pair_set.add((phrase_str, phrase_category))
        return list(phrase_pair_set)

    def gather_entities(self, article_rdd, metric_fptr):
        """
        :return: ner_rdd
        """

        entity_pair_rdd = article_rdd.flatMap(lambda x: self._get_unique_entity_pairs(x))

        # count number of articles per entity-pair
        entity_pair_count_prdd = entity_pair_rdd.map(lambda x: (x, 1)).reduceByKey(add).filter(
            lambda p2c: p2c[1] >= NUM_ARTICLES_PER_ENTITY_LLIM)

        entity_pair_count_prdd.cache()
        self._print_stats(metric_fptr, entity_pair_count_prdd)

        # p2c[0] returns the entity-pair (phrase_str, phrase_category)
        ner_rdd = entity_pair_count_prdd.map(lambda p2c: PhraseObj(p2c[0][0], p2c[0][1]))
        return ner_rdd

    def _print_stats(self, metric_fptr, entity_pair_count_prdd):
        nb_bins = 30
        buckets, histogram = entity_pair_count_prdd.map(lambda x: x[1]).histogram(nb_bins)

        # Number of entities with articles-count=[..., ...):
        print('Article counts histogram below is on the corpus, not the generated data', file=metric_fptr)
        print(('Since each article has more than one entity, the denominator of articles-count is not ' \
               'the same as number of articles.'), file=metric_fptr)
        self.stat_computation.help_print_histogram(
            metric_fptr, buckets, histogram, 'Histogram of gathered entities with articles-count=')

        self._stats_phrase_category_counts(metric_fptr, entity_pair_count_prdd)
        self._stats_top_occurring_entities(entity_pair_count_prdd)

    def _stats_phrase_category_counts(self, metric_fptr, entity_pair_count_prdd):
        phrase_category_counts = entity_pair_count_prdd.map(
            lambda p2c: (p2c[0][1], 1)
        ).reduceByKey(add).collectAsMap()

        self.stat_computation.print_phrase_category_counts(metric_fptr, phrase_category_counts, 'Number of gathered phrases')

    def _stats_top_occurring_entities(self, entity_pair_count_prdd):
        frequent_lst = entity_pair_count_prdd.takeOrdered(
            1000,
            key=lambda x: -x[1])
        with open(os.path.join(self.output_dir, 'frequently_occurring_phrases.txt'), 'w') as fptr:
            for line in frequent_lst:
                print(line, file=fptr)
