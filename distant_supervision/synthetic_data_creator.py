import os
import json

from logzero import logger as logging
from elasticsearch6.exceptions import RequestError
from pyspark import StorageLevel

from .entity_to_queries_mapper import EntityToQueriesMapper
from .constants import QUESTION_NUM_WORDS_ULIM
from .text_preprocessor import TextPreprocessor
from .utils import ElasticsearchMagic, ElasticsearchConfig, random_str, find_all
from . import utils
from .exceptions import DsDatasetCreationError
from .data_models import DsDatum, QuestionStyle, PhraseMode
from .question_generator import QuestionGenerator
from .ner_entity_gatherer import NerEntityGatherer, convert_ner_rdd_to_set
from .stat_computation import StatComputation

NUM_OF_HITS_FROM_ES = 1000

QUESTION_STYLES_FOR_JSONLINES = [QuestionStyle.TEMPLATE_WBA]


class SyntheticDataCreator:
    def __init__(self,
                 output_dir, *,
                 es_hosts,
                 es_index_name,
                 debug_save,
                 ulim_count,
                 nb_ner_ulim,
                 num_partitions,
                 nb_aux_qs_matches,
                 nb_aux_awc_matches,
                 phrase_mode,
                 whxx_ngram_table):

        self.output_dir = output_dir
        self.debug_save = debug_save
        self.num_partitions = num_partitions

        self.phrase_mode = phrase_mode

        self.ulim_count = ulim_count  # limit the number of results

        self.nb_ner_ulim = nb_ner_ulim

        self.nb_aux_qs_matches = nb_aux_qs_matches
        self.nb_aux_awc_matches = nb_aux_awc_matches

        self.whxx_ngram_table = whxx_ngram_table
        self.text_preprocessor = TextPreprocessor()
        self.question_generator = QuestionGenerator(self.whxx_ngram_table, self.text_preprocessor)

        self.es_conf = ElasticsearchConfig(
            hosts=es_hosts,
            index_name=es_index_name,
            doc_type='doc')


    def _compute_answer_start(self, *, answer_str, es_query, context):
        within_sent_start_pos_lst = find_all(es_query, answer_str)
        if not within_sent_start_pos_lst:
            raise DsDatasetCreationError('Cannot find start position for answer="{}" in es_query="{}"'.format(
                answer_str, es_query))

        sentence_start_pos_lst = find_all(context, es_query)  # should probably only have a single occurrence
        if not sentence_start_pos_lst:
            raise DsDatasetCreationError('Cannot find es_query="{}" in the following:\n{}'.format(
                es_query, context))

        start_pos_lst = []
        for sentence_start_pos in sentence_start_pos_lst:
            start_pos_lst.extend([pos + sentence_start_pos for pos in within_sent_start_pos_lst])

        for pos in start_pos_lst:
            # verify that it's correct
            if context[pos:pos + len(answer_str)] != answer_str:
                raise DsDatasetCreationError(
                    'inconsistent start_pos found {}'.format(
                        str(dict(
                            start_pos=pos,
                            sentence_start_pos_lst=sentence_start_pos_lst,
                            within_sent_start_pos_lst=within_sent_start_pos_lst,
                            answer_str=answer_str,
                            es_query=es_query,
                            context=context,
                        ))))

        return start_pos_lst

    def _make_styled_questions(self, *, qpa, es_hit, answer_str, rng):
        styled_questions = {}

        styled_questions[QuestionStyle.CLOZE_GENERIC] = self.question_generator.make_cloze_style(
            es_hit['_source']['body'],
            answer_str,
            '[MASK]')

        styled_questions[QuestionStyle.CLOZE_CATEGORY] = self.question_generator.make_cloze_style(
            es_hit['_source']['body'],
            answer_str,
            '[{}]'.format(qpa.phrase.phrase_category))

        templated_strings = self.question_generator.make_template_qg_styles(
            es_hit['_source']['body'],
            answer_str,
            qpa.phrase.phrase_category,
            rng)

        styled_questions.update(templated_strings)

        # unwrap the enum
        return {k.value: v for k, v in styled_questions.items()}

    def _construct_dataset_sample(self, *, qpa, hit, hit_phrases, es_query, es_rank, backfill_article, backfill_sent, rng):
        qid = random_str(16)

        context = qpa.article_raw

        answer_str = qpa.phrase.phrase_str
        answer_start_lst = self._compute_answer_start(
            answer_str=answer_str,
            es_query=es_query,
            context=context)

        if len(answer_start_lst) == 0:
            raise DsDatasetCreationError('Did not find any answer_start answer={}: {}\n{}'.format(
                answer_str, es_query, context))

        answers = [{'text': answer_str, 'answer_start': pos} for pos in answer_start_lst]

        styled_questions = self._make_styled_questions(
            qpa=qpa,
            es_hit=hit,
            answer_str=answer_str,
            rng=rng)

        datum = DsDatum(
            qid=qid,
            styled_questions=styled_questions,
            answers=answers,
            context=context)

        datum.meta = {
            "es_query": es_query,
            "es_rank": es_rank,
            "es_score": hit['_score'],
            "answer_phrase_category": qpa.phrase.phrase_category,
            "context": {
                "article_id": qpa.article_id,
                "article_title": qpa.article_title,
            },
            "question": {
                "article_id": int(hit['_source']['article_id']),
                "article_title": hit['_source']['article_title'],
                "phrases": hit_phrases,
            },
            "backfill_nb_articles": backfill_article,
            "backfill_nb_sents": backfill_sent,
        }

        return datum

    def _get_hit_phrases(self, es_hit):
        entities = [tuple(e) for e in json.loads(es_hit['_source']['entities'])]
        noun_chunks = [tuple(e) for e in json.loads(es_hit['_source']['noun_chunks'])]

        if self.phrase_mode is PhraseMode.NER_ONLY:
            phrases = self.text_preprocessor.get_phrases(entities=entities, noun_chunks=[])
        else:
            phrases = self.text_preprocessor.get_phrases(entities=entities, noun_chunks=noun_chunks)
        return set(phrases)


    def _obtain_retrieved_sentences_for_single_article(self, *, qpa, es, rng, backfill_article):
        """
        Return a list. Currently, it should return a list with zero or one element

        :param backfill_article: for debugging purpose. The index of backfill
        """

        filtered_sentences = qpa.filtered_sents
        rng.vanilla.shuffle(filtered_sentences)

        context_phrase_set = set(qpa.article_phrases)

        for backfill_sent, sent in enumerate(filtered_sentences):
            # Consider adding title to es_query
            es_query = sent.text

            es_query_phrases = set(sent.get_phrases(self.phrase_mode))

            request_body = {
                "query": {
                    "bool": {
                        "should": {
                            "match": {"body_with_title": es_query},
                        },
                        "must": {
                            "match": {"body_with_title": qpa.phrase.phrase_str},
                        },
                        "must_not": {
                            "match": {"article_id": str(qpa.article_id)}
                        }
                    }
                }
            }

            try:
                # maybe convert to using msearch (multi-search)
                results = es.search(
                    index=self.es_conf.index_name,
                    doc_type=self.es_conf.doc_type,
                    size=NUM_OF_HITS_FROM_ES,  # top-k
                    request_timeout=60,
                    body=request_body)
            except RequestError as ex:
                print('ES RequestError found: {}'.format(str(ex)))
                continue

            for hit_idx, hit in enumerate(results['hits']['hits']):
                retrieved_str = hit['_source']['body']
                hit_article_id = int(hit['_source']['article_id'])

                # TODO add create real hit_phrases
                hit_phrases = self._get_hit_phrases(hit)

                if hit_article_id == qpa.article_id:
                    # if the hit is from the same article as the query, skip
                    # we already check this in ES, but just to be safe
                    continue

                if len(hit_phrases) < 2:
                    # if number of entities in hit (question) is less than 2, it's likely that it's unanswerable
                    # Should have at least 2 entities
                    continue

                if (qpa.phrase.phrase_str, qpa.phrase.phrase_category) not in hit_phrases:
                    continue

                if len(hit_phrases & es_query_phrases) < self.nb_aux_qs_matches + 1:
                    # needs to +1 to nb_aux_qs_matches because there is already a match for phrase_str
                    continue

                if len(hit_phrases & context_phrase_set) < self.nb_aux_awc_matches + self.nb_aux_qs_matches + 1:
                    # needs to add self.nb_aux_qs_matches + 1 for both the phrase_str and aux_qs matches
                    continue

                if len(retrieved_str.split()) > QUESTION_NUM_WORDS_ULIM:
                    # used naive splitting based on spaces
                    continue

                nb_entity_occurrences = len(self.text_preprocessor.findall_substr(qpa.phrase.phrase_str, retrieved_str))
                if nb_entity_occurrences != 1:
                    # We already check existence in ES query, but we want only occurrence of *once* here.
                    # This is to simplify conversion to question-style
                    continue

                if self.text_preprocessor.is_similar(es_query, retrieved_str, 0.95, discard_stopwords=False):
                    # recall that retrieved_str (from ES hit) is actually a sentence
                    # If the two sentence are too similar, then it's likely a plagiarized sentence
                    continue

                if not self.text_preprocessor.is_similar(retrieved_str, es_query, 0.30, discard_stopwords=True):
                    # if there is almost no overlap, skip
                    continue

                # the "hit" is used to generate the question
                dataset_sample = self._construct_dataset_sample(
                    rng=rng,
                    qpa=qpa,
                    hit=hit,
                    hit_phrases=list(hit_phrases),
                    es_query=es_query,
                    es_rank=hit_idx,
                    backfill_article=backfill_article,
                    backfill_sent=backfill_sent)

                return [dataset_sample]


    def _obtain_retrieved_sentences(self, entity2queries, es, rng):
        """
        :return: either empty list or a list with a single element
        """
        qpa_lst = entity2queries[1]
        rng.vanilla.shuffle(qpa_lst)

        for backfill_article, qpa in enumerate(qpa_lst):
            retrieved_objs = self._obtain_retrieved_sentences_for_single_article(
                qpa=qpa,
                es=es,
                rng=rng,
                backfill_article=backfill_article)
            if retrieved_objs:
                return retrieved_objs
        return []


    def _compute_ds_data_by_partition(self, entity2queries_lst):
        # Do not move RNG as instance variables. You want a different seed for each partition
        rng = utils.RandomNumberGenerator()

        es = ElasticsearchMagic.get_instance(
            'singleton',
            hosts=[self.es_conf.hosts],
            timeout=60)

        for entity2queries in entity2queries_lst:
            yield self._obtain_retrieved_sentences(entity2queries, es, rng)

    def _split_by_style_and_write(self, ds_data_rdd):
        # Only add to this list if we want to run the dataset directly using BERT
        # If the data would be used for style-transfer, you can just use core_data format.
        qstyle_lst = QUESTION_STYLES_FOR_JSONLINES

        for qstyle in qstyle_lst:
            ds_data_rdd.map(lambda x: x.jsonify_single_style(qstyle)).saveAsTextFile(
                os.path.join(self.output_dir, qstyle.value))

    def _calculate_phrase_rdd(self, article_rdd, metric_fptr):
        ner_entity_gatherer = NerEntityGatherer(
            output_dir=self.output_dir,
            phrase_mode=self.phrase_mode)
        ner_rdd = ner_entity_gatherer.gather_entities(article_rdd, metric_fptr)

        return ner_rdd

    def _perform_subsample(self, rdd, subsample_frac):
        if subsample_frac < 0.99:
            return rdd.sample(False, subsample_frac)
        return rdd

    def _perform_subsample_by_count(self, rdd, subsample_count, *, tot_count=None):
        extra_frac = 2.0  # if extra_frac = 1.10, sample for 10% more data

        if tot_count is None:
            tot_count = rdd.count()
        frac = float(subsample_count) / tot_count

        if frac >= 0.99:
            return rdd

        return self._perform_subsample(rdd, frac * extra_frac)

    def run_job(self, sc, article_rdd, phrase_rdd, metric_fptr):
        total_nb_articles = article_rdd.count()
        print('Number of articles in corpus: {}\n'.format(total_nb_articles), file=metric_fptr)

        if phrase_rdd is None:
            phrase_rdd = self._calculate_phrase_rdd(article_rdd, metric_fptr)

        phrase_set = convert_ner_rdd_to_set(phrase_rdd, self.nb_ner_ulim)

        if self.ulim_count is not None:
            subsampled_article_rdd = self._perform_subsample_by_count(
                article_rdd,
                self.ulim_count,
                tot_count=total_nb_articles)
        else:
            subsampled_article_rdd = article_rdd

        if self.debug_save:
            subsampled_article_rdd.cache()  # only do this in debug_save
            print('Number of subsampled articles: {}\n'.format(subsampled_article_rdd.count()), file=metric_fptr)

        entity2queries_prdd = EntityToQueriesMapper(self.phrase_mode).get_entity_to_queries_v2(
            sc, subsampled_article_rdd, phrase_set, metric_fptr)

        ds_data_rdd_nonflat = entity2queries_prdd.mapPartitions(lambda x: self._compute_ds_data_by_partition(x))
        ds_data_rdd = ds_data_rdd_nonflat.flatMap(lambda x: x)

        ds_data_rdd.persist(StorageLevel.MEMORY_AND_DISK)

        # need to separate by different style and save it to separate folders
        self._split_by_style_and_write(ds_data_rdd)

        ds_data_rdd.map(lambda x: x.jsonify()).saveAsTextFile(os.path.join(self.output_dir, 'core_data'))

        StatComputation().print_output_stats(ds_data_rdd, metric_fptr)



"""
{
    “qid”: “xxxxxxxxxx”,
    “question”: “xxxxxxxxx”,
    “context”: “xxxxxxxx”,
    "answers": [
        {
            “answer_start”: 177,
            “text”: “xxxxxxx”,
        },
    ],
    “meta”: {
        "es_query": "....",
        "es_rank": ...,
        "es_score": ...,
        "answer_phrase_category": "PERSON"
        "context": {
          "article_id": xxxxx,
          "article_title": "xxxxx",
        },
        "question": {
          "article_id": xxxxx,
          "article_title": "xxxxx",
        }
    }
}
"""
