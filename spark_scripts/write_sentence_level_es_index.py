from pyspark import SparkContext
from contextlib import ExitStack
import argparse
import os
import timy
from logzero import logger as logging
from distant_supervision.ds_es_client import ESIndexWriter

@timy.timer()
def main(sc):
    argp = argparse.ArgumentParser()
    argp.add_argument('--corpus', help='input path, e.g. foobar/rollup/', required=True)
    argp.add_argument('--output-dir', default="output/", help='')
    argp.add_argument('--es-hosts', help='', default=os.getenv('AES_HOSTS'))
    argp.add_argument('--es-index', help='', required=True)
    argp.add_argument('--num-partitions', type=int, default=1000, help='')
    argp.add_argument('--debug-save', help='for debugging purposes', action='store_true')
    args = argp.parse_args()

    corpus_rdd = sc.textFile(args.corpus, minPartitions=args.num_partitions)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with ExitStack() as stack:
        metric_filename = os.path.join(args.output_dir, 'metric.txt')
        metric_fptr = stack.enter_context(open(metric_filename, 'w'))

        print('args: {}'.format(args), file=metric_fptr)

        job = ESIndexWriter(
            args.output_dir,
            es_hosts=args.es_hosts,
            es_index_name=args.es_index,
            debug_save=args.debug_save)
        job.run_job(corpus_rdd, metric_fptr)

        logging.info('Output directory: {}'.format(args.output_dir))


if __name__ == '__main__':
    sc = SparkContext(appName="Write sentence-level ES index")
    sc.addFile('distant_supervision', recursive=True)

    main(sc)
