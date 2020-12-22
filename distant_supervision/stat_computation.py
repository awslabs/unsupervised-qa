from operator import add
from .data_models import QuestionStyle

class StatComputation:
    def __init__(self):
        pass

    def help_print_histogram(self, metric_fptr, buckets, histogram, description):
        tot = sum(histogram)

        for i in range(len(histogram)):
            print('{}[{:.3f}, {:.3f}): {} / {} ({:.2f}%)'.format(
                description,
                buckets[i],
                buckets[i + 1],
                histogram[i],
                tot,
                100.0 * histogram[i] / tot), file=metric_fptr)
        print(file=metric_fptr)
        metric_fptr.flush()

    def _print_context_stats(self, ds_data_rdd, metric_fptr):
        nb_bins = 10
        buckets, histogram = ds_data_rdd.map(lambda x: len(x.context.split())).histogram(nb_bins)

        self.help_print_histogram(metric_fptr, buckets, histogram, 'Histogram of contexts with words-count=')

    def _print_question_stats(self, ds_data_rdd, metric_fptr):
        nb_bins = 10
        buckets, histogram = ds_data_rdd.map(
            lambda x: len(x.styled_questions[QuestionStyle.CLOZE_GENERIC.value].split())
        ).histogram(nb_bins)

        self.help_print_histogram(metric_fptr, buckets, histogram, 'Histogram of questions with words-count=')

    def _print_answer_stats(self, ds_data_rdd, metric_fptr):
        nb_bins = 10
        buckets, histogram = ds_data_rdd.map(
            lambda x: len(x.answers[0]['text'])
        ).histogram(nb_bins)

        self.help_print_histogram(metric_fptr, buckets, histogram, 'Histogram of answers with chars-count=')

    def _print_backfill_stats(self, ds_data_rdd, metric_fptr):
        default_nb_bins = 10

        for field, nb_bins in [
                ('backfill_nb_sents', 5),
                ('backfill_nb_articles', default_nb_bins),
                ('es_rank', default_nb_bins),
                ('es_score', default_nb_bins)]:
            buckets, histogram = ds_data_rdd.map(
                lambda x: x.meta[field]
            ).histogram(nb_bins)

            self.help_print_histogram(metric_fptr, buckets, histogram, 'Histogram of meta field {}='.format(field))

    def _print_article_diversity_stats_helper(self, meta_field, ds_data_rdd, metric_fptr):
        nb_bins = 10

        row_counts = ds_data_rdd.map(lambda x: (x.meta[meta_field]['article_id'], 1)).reduceByKey(add).values()
        buckets, histogram = row_counts.histogram(nb_bins)

        self.help_print_histogram(
            metric_fptr, buckets, histogram,
            'Article diversity: number of {}.article_id with example-count='.format(meta_field))

    def _print_article_diversity_stats(self, ds_data_rdd, metric_fptr):
        for meta_field in ['question', 'context']:
            self._print_article_diversity_stats_helper(meta_field, ds_data_rdd, metric_fptr)


    def print_phrase_category_counts(self, metric_fptr, phrase_category_counts, description):
        total_count = sum(phrase_category_counts.values())
        print(file=metric_fptr)
        for phrase_category, count in sorted(phrase_category_counts.items()):
            print('{} with NER category "{}": {} / {} ({:.2f}%)'.format(
                description,
                phrase_category,
                count,
                total_count,
                100.0 * count / total_count), file=metric_fptr)
        print(file=metric_fptr)
        metric_fptr.flush()

    def print_output_stats(self, ds_data_rdd, metric_fptr):
        print(file=metric_fptr)
        print('Count of ds_data_rdd: {}'.format(ds_data_rdd.count()), file=metric_fptr)

        phrase_category_counts = ds_data_rdd.map(
            lambda x: (x.meta['answer_phrase_category'], 1)
        ).reduceByKey(add).collectAsMap()

        self.print_phrase_category_counts(metric_fptr, phrase_category_counts, 'Number of synthetic samples')

        self._print_context_stats(ds_data_rdd, metric_fptr)
        self._print_question_stats(ds_data_rdd, metric_fptr)
        self._print_answer_stats(ds_data_rdd, metric_fptr)

        self._print_backfill_stats(ds_data_rdd, metric_fptr)

        self._print_article_diversity_stats(ds_data_rdd, metric_fptr)
