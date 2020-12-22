from pyspark import SparkContext
from contextlib import ExitStack
from operator import add
from collections import defaultdict
import argparse
import toml
import os
from logzero import logger as logging
import timy
import re
from distant_supervision.data_models import RcQuestion

"""
spark-submit --driver-memory 300G --master local[90] spark_scripts/stat_for_ner_category_to_wh_words.py --squadner ~/efs/data_nonsync/squad_qa/squad_ner/squad_ner_train.jsonl --output-dir outputs/whxx_ngram_table-`utcid`
"""

ULIM_NGRAM_SIZE = 3
TOPK = 3
TOML_NGRAM_SIZE = 2

class WhxxNgram:
    def __init__(self, ngram_size, ngram, count):
        self.ngram_size = ngram_size
        self.ngram = ngram
        self.count = count


def _extract_leading_ngrams(question):
    """
    :return: list of ((<NER category>, <number of words>, <ngram>), 1), e.g. returns ( ("PERSON", 2, "who is"), 1)
    """
    ner_category = question.answers[0]['ner_category']

    question_text = question.question.lower()
    question_text = re.sub(r'\W+', ' ', question_text)
    tokens = question_text.split()
    return [
        ((ner_category, n, ' '.join(tokens[:n])), 1)
        for n in range(1, ULIM_NGRAM_SIZE + 1)]

def _compute_count_per_category(question_rdd):
    """
    returns dictionary of category => count
    """

    category_count_prdd = question_rdd.map(lambda x: (x.answers[0]['ner_category'], 1)).reduceByKey(add)
    return category_count_prdd.collectAsMap()

def _add_to_toml(toml_dct, category, size2list):
    whxx_ngram_objs = size2list[TOML_NGRAM_SIZE]

    dct = {}
    for whxx_ngram in whxx_ngram_objs:
        dct[whxx_ngram.ngram] = whxx_ngram.count

    toml_dct[category] = dct

def _run_stat(question_rdd, metric_fptr, metric_per_category_fptr, toml_fptr, output_dir):
    count_per_category = _compute_count_per_category(question_rdd)

    unit_count_prdd = question_rdd.flatMap(_extract_leading_ngrams)

    count_prdd = unit_count_prdd.reduceByKey(add).filter(lambda x: x[1] >= 10)
    count_prdd.cache()
    count_prdd.saveAsTextFile(os.path.join(output_dir, 'count_prdd'))

    count_dct = count_prdd.collectAsMap()

    toml_dct = {}

    category2list = defaultdict(list)
    for k, count in count_dct.items():
        ner_category = k[0]
        whxx_ngram = WhxxNgram(k[1], k[2], count)

        category2list[ner_category].append(whxx_ngram)

    for category, whxx_ngram_lst in sorted(category2list.items()):
        size2list = defaultdict(list)  # ngram-size to list-of-whxx_ngram

        for whxx_ngram in whxx_ngram_lst:
            size2list[whxx_ngram.ngram_size].append(whxx_ngram)

        for size in size2list.keys():
            size2list[size] = sorted(size2list[size], key=lambda x: -x.count)

        for size, lst in sorted(size2list.items()):
            print('{} {}-gram'.format(category, size), file=metric_fptr)
            for whxx_ngram in lst[:TOPK]:
                print('count= {:4d} / {:4d} ({:5.2f}%) "{}"'.format(
                    whxx_ngram.count,
                    count_per_category[category],
                    100.0 * whxx_ngram.count / count_per_category[category],
                    whxx_ngram.ngram), file=metric_fptr)
            print(file=metric_fptr)

        _add_to_toml(toml_dct, category, size2list)

    for category, whxx_ngram_lst in sorted(category2list.items()):
        sorted_lst = sorted(whxx_ngram_lst, key=lambda x: -x.count)[:10]

        print('{}'.format(category), file=metric_per_category_fptr)
        for whxx_ngram in sorted_lst:
            print('count= {:4d} / {:4d} ({:5.2f}%) "{}"'.format(
                whxx_ngram.count,
                count_per_category[category],
                100.0 * whxx_ngram.count / count_per_category[category],
                whxx_ngram.ngram), file=metric_per_category_fptr)
        print(file=metric_per_category_fptr)

    print(toml.dumps(toml_dct), file=toml_fptr)


@timy.timer()
def main(sc):
    argp = argparse.ArgumentParser()
    argp.add_argument('--squadner', help='input path of squadner data', required=True)
    argp.add_argument('--output-dir', default="output/", help='', required=True)
    argp.add_argument('--num-partitions', type=int, default=1000, help='')
    argp.add_argument('--debug-save', help='for debugging purposes', action='store_true')
    args = argp.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with ExitStack() as stack:
        metric_filename = os.path.join(args.output_dir, 'metric.txt')
        metric_fptr = stack.enter_context(open(metric_filename, 'w'))

        metric_per_category_fptr = stack.enter_context(open(
            os.path.join(args.output_dir, 'metric_per_category.txt'), 'w'))

        toml_fptr = stack.enter_context(open(
            os.path.join(args.output_dir, 'whxx_ngram_table.toml'), 'w'))

        question_rdd = sc.textFile(args.squadner, minPartitions=args.num_partitions).map(RcQuestion.deserialize_json)
        _run_stat(question_rdd, metric_fptr, metric_per_category_fptr, toml_fptr, args.output_dir)
        logging.info('Output dir: {}'.format(args.output_dir))


if __name__ == '__main__':
    sc = SparkContext(
        appName="Stat for NER category to Whxx words")

    sc.addFile('distant_supervision', recursive=True)

    main(sc)

"""
{
    "qid": "57277c965951b619008f8b2b",
    "question": "What do people engage in after they've disguised themselves?",
    "context": "In Greece Carnival is also ...",
    "answers": [
        {
            "ner_category": "SOME_CATEGORY",
            "answer_start": 677,
            "text": "pranks and revelry"
        }
    ],
    "article_title": "Carnival"
}
"""
