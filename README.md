# Template-Based Question Generation from Retrieved Sentences for Improved Unsupervised Question Answering

Code and synthetic data from our [ACL 2020 paper](https://www.aclweb.org/anthology/2020.acl-main.413/)

## Abstract

Question Answering (QA) is in increasing demand as the amount of information
available online and the desire for quick access to this content grows. A common
approach to QA has been to fine-tune a pretrained language model on a task-specific
labeled dataset. This paradigm, however, relies on scarce, and costly to obtain,
large-scale human-labeled data. We propose an unsupervised approach to training
QA models with generated pseudo-training data. We show that generating questions
for QA training by applying a simple template on a related, retrieved sentence
rather than the original context sentence improves downstream QA performance
by allowing the model to learn more complex context-question relationships.
Training a QA model on this data gives a relative improvement over a previous
unsupervised model in F1 score on the SQuAD dataset by about 14%, and 20% when
the answer is a named entity, achieving state-of-the-art performance on SQuAD
for unsupervised QA.

## Synthetic data

Generated synthetic data for the publication is located under `enwiki_synthetic/`

## Requirements

1. PySpark
2. ElasticSearch 6

## Instruction to generate retrieval-based synthetic data

Tokenize and perform NER:

```
spark-submit --master local[90] --driver-memory 200G spark_scripts/tokenize_and_ner_inputs.py --corpus=enwiki/clean/*/*.raw  --output outputs/sent-tok-rollup
```

Then we write the tokenized sentences to ElasticSearch index. This uses `AES_HOSTS` environment variable.

```
spark-submit --master local[90] --driver-memory 4G spark_scripts/write_sentence_level_es_index.py --corpus=outputs/sent-tok-rollup/rollup/ --es-index uqa-es-index --output outputs/write-es
```

Create QA synthetic dataset

```
spark-submit --master local[90] --driver-memory 300G spark_scripts/create_ds_synthetic_dataset.py --corpus=outputs/sent-tok-rollup/rollup/ --output outputs/synthetic-uqa-auxqs1awc1-`utcid` --aux-qs=1 --aux-awc=1 --ulim-count=500000
```


## Citation

You can cite our [paper](https://www.aclweb.org/anthology/2020.acl-main.413/):

```
@inproceedings{fabbri-etal-2020-template,
    title = "Template-Based Question Generation from Retrieved Sentences for Improved Unsupervised Question Answering",
    author = "Fabbri, Alexander  and
      Ng, Patrick  and
      Wang, Zhiguo  and
      Nallapati, Ramesh  and
      Xiang, Bing",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.413",
    doi = "10.18653/v1/2020.acl-main.413",
    pages = "4508--4513",
    abstract = "Question Answering (QA) is in increasing demand as the amount of information available online and the desire for quick access to this content grows. A common approach to QA has been to fine-tune a pretrained language model on a task-specific labeled dataset. This paradigm, however, relies on scarce, and costly to obtain, large-scale human-labeled data. We propose an unsupervised approach to training QA models with generated pseudo-training data. We show that generating questions for QA training by applying a simple template on a related, retrieved sentence rather than the original context sentence improves downstream QA performance by allowing the model to learn more complex context-question relationships. Training a QA model on this data gives a relative improvement over a previous unsupervised model in F1 score on the SQuAD dataset by about 14{\%}, and 20{\%} when the answer is a named entity, achieving state-of-the-art performance on SQuAD for unsupervised QA.",
}
```


## License

This project is licensed under the Apache-2.0 License.
