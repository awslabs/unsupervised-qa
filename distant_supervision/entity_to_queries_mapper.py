from logzero import logger as logging
from . import utils
from operator import add
from .data_models import PhraseObj, QueriesPerArticleObj

from .constants import CONTEXT_NUM_WORDS_ULIM, NUM_ENTITIES_PER_ARTICLE_TO_CONSIDER, NUM_ENTITIES_PER_ARTICLE_TO_KEEP
from .constants import NUM_ARTICLES_PER_ENTITY_ULIM, NUM_WORDS_IN_QUERY_SENTENCE_ULIM, NUM_OF_SENTENCES_ULIM

class EntityToQueriesMapper:
    def __init__(self, phrase_mode):
        self.phrase_mode = phrase_mode

    def _largest_index_exceeding_ulim_context(self, word_count_arr):
        # compute the largest index that could have approximately CONTEXT_NUM_WORDS_ULIM words
        reverse_accum = 0
        for idx in range(len(word_count_arr)-1, 0, -1):
            reverse_accum += word_count_arr[idx]
            if reverse_accum >= CONTEXT_NUM_WORDS_ULIM:
                return idx
        return 0

    def _get_valid_context_sentences(self, article, rng):
        """
        0  10
        1  5
        2  6
        3  7

        Let's say CONTEXT_NUM_WORDS_ULIM=10. Then The largest_inclusive_idx should be index of 2.
        We then randint(0,2)
        """
        word_count_arr = []
        for sent in article.sents:
            word_count_arr.append(len(sent.text.split()))  # using split() is only approximate because of punctuation

        assert len(word_count_arr) == len(article.sents)

        largest_inclusive_idx = self._largest_index_exceeding_ulim_context(word_count_arr)
        rnd_idx = rng.vanilla.randint(0, largest_inclusive_idx)

        good_sents = []
        accum_nb_words = 0
        for i in range(rnd_idx, len(article.sents)):
            good_sents.append(article.sents[i])
            accum_nb_words += word_count_arr[i]
            if accum_nb_words >= CONTEXT_NUM_WORDS_ULIM:
                break

        return good_sents

    def _get_all_phrases_from_sentence_list(self, sent_lst):
        phrase_set = set()
        for sent in sent_lst:
            phrase_set.update(sent.get_phrases(self.phrase_mode))
        return list(phrase_set)

    def _get_entity2qpa_list(self, article, ner_broadcast):
        rng = utils.RandomNumberGenerator()

        # good_sents is contiguous
        good_sents = self._get_valid_context_sentences(article, rng)

        # only use the first N sentences, instead of using article.text
        article_raw = ' '.join([sent.text for sent in good_sents])

        article_phrases = self._get_all_phrases_from_sentence_list(good_sents)

        candidate_phrase_pairs = set()
        for sent in good_sents:
            candidate_phrase_pairs.update(sent.get_phrases(self.phrase_mode))

        candidate_phrase_pairs = list(candidate_phrase_pairs & ner_broadcast.value)  # only keep ones that are in NER list

        rng.vanilla.shuffle(candidate_phrase_pairs)
        candidate_phrase_pairs = candidate_phrase_pairs[:NUM_ENTITIES_PER_ARTICLE_TO_CONSIDER]

        result_lst = []
        for phrase_str, phrase_category in candidate_phrase_pairs:
            # filtered sentences where the "answer" string is in there.
            # Also, keep only ones that have less than X number of words (others are likely an error).
            filtered_sents = [
                s for s in good_sents
                if (phrase_str, phrase_category) in s.get_phrases(self.phrase_mode) and len(s.text.split()) <= NUM_WORDS_IN_QUERY_SENTENCE_ULIM]

            if not filtered_sents:
                continue

            phrase = PhraseObj(phrase_str, phrase_category)

            rng.vanilla.shuffle(filtered_sents)
            filtered_sents = filtered_sents[:NUM_OF_SENTENCES_ULIM]  # only randomly take this many sentences

            qpa = QueriesPerArticleObj(
                article_id=article.id,
                article_title=article.title,
                article_raw=article_raw,  # do not use article.text here. We only use first/some N sentences
                article_phrases=article_phrases,
                filtered_sents=filtered_sents,  # filtered sentences where the "answer" string is in there
                phrase=phrase)
            result_lst.append(((phrase.phrase_str, phrase.phrase_category), [qpa]))

            if len(result_lst) >= NUM_ENTITIES_PER_ARTICLE_TO_KEEP:
                break
        return result_lst


    def get_entity_to_queries_v2(self, sc, article_rdd, ner_set, metric_fptr):
        """
        return a RDD with (entity, list of qpa)

        - List of QPAs are randomly selected/ordered.
        - V2 uses broadcast instead of explicit join
        - Take only first N sentences from an article. If we need more samples, consider splitting article
          to multiple passages.
        """

        print('Size of NER set: {}\n'.format(len(ner_set)), file=metric_fptr)

        ner_broadcast = sc.broadcast(ner_set)

        entity2qpa_prdd = article_rdd.flatMap(lambda x: self._get_entity2qpa_list(x, ner_broadcast))

        ner_broadcast.unpersist()

        entity2queries_prdd = entity2qpa_prdd.reduceByKey(add)

        # limit number of backfill articles. Some entities such as first is too common.
        entity2queries_prdd = entity2queries_prdd.map(lambda e2q: (e2q[0], e2q[1][:NUM_ARTICLES_PER_ENTITY_ULIM]))

        return entity2queries_prdd
