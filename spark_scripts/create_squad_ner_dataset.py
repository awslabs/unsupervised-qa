from pyspark import SparkContext
from contextlib import ExitStack
import argparse
import os
from logzero import logger as logging
import timy
from distant_supervision.data_models import RcQuestion
from distant_supervision.squad_ner_creator import SquadNerCreator

"""
Debug mode:

spark-submit --driver-memory 300G --master local[90] scripts/create_squad_ner_dataset.py --squad-rc-dir ~/efs/data_nonsync/squad_qa/squad_rc/ --debug-save --output-dir outputs/debug-`utc2`


Full run

spark-submit --driver-memory 300G --master local[90] scripts/create_squad_ner_dataset.py --squad-rc-dir ~/efs/data_nonsync/squad_qa/squad_rc/ --output-dir outputs/squad-ner-`utc2`
"""


def _run_job(input_jsonl_filepath, channel, metric_fptr, args):
    question_rdd = sc.textFile(input_jsonl_filepath, minPartitions=args.num_partitions).map(RcQuestion.deserialize_json)

    job = SquadNerCreator(
        os.path.join(args.output_dir, 'squad_ner_{}'.format(channel)),
        num_partitions=args.num_partitions,
        debug_save=args.debug_save)
    job.run_job(sc, question_rdd, metric_fptr)


@timy.timer()
def main(sc):
    argp = argparse.ArgumentParser()
    argp.add_argument('--squad-rc-dir', help='input path of squad data', required=True)
    argp.add_argument('--output-dir', default="output/", help='', required=True)
    argp.add_argument('--num-partitions', type=int, default=1000, help='')
    argp.add_argument('--debug-save', help='for debugging purposes', action='store_true')
    args = argp.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    input_dct = {
        'train': 'squad_rc_train.jsonl',
        'dev': 'squad_rc_dev.jsonl',
        'test': 'squad_rc_test.jsonl',
    }
    if args.debug_save:
        # debug_save mode only runs for dev set
        input_dct = {'dev': input_dct['dev']}

    with ExitStack() as stack:
        for channel, jsonl_basename in input_dct.items():
            metric_filename = os.path.join(args.output_dir, 'metric_{}.txt'.format(channel))
            metric_fptr = stack.enter_context(open(metric_filename, 'w'))

            input_jsonl_filepath = os.path.join(args.squad_rc_dir, jsonl_basename)
            _run_job(input_jsonl_filepath, channel, metric_fptr, args)

        logging.info('Output directory: {}'.format(args.output_dir))


if __name__ == '__main__':
    sc = SparkContext(
        appName="Construct SQuAD-NER dataset")

    sc.addFile('distant_supervision', recursive=True)

    main(sc)
