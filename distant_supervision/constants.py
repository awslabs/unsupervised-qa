NUM_OF_SENTENCES_ULIM = 5

NUM_ENTITIES_PER_ARTICLE_TO_CONSIDER = 30  # only in v2
NUM_ENTITIES_PER_ARTICLE_TO_KEEP = 5

NUM_WORDS_IN_QUERY_SENTENCE_ULIM = 100

NUM_ARTICLES_PER_ENTITY_LLIM = 2
NUM_ARTICLES_PER_ENTITY_ULIM = 30  # limit number of backfill articles

# NOTE that this is using naive split() rather than the BPE vocab count in BERT
# QUESTION_NUM_WORDS_ULIM = 100  # still keep fairly long sentences
# ANSWER_NUM_CHARS_ULIM = 100  # not using the parameters from zgw-exps max_answer_size
# The QUESTION_NUM_WORDS_ULIM had 50 in zgw-exps

QUESTION_NUM_WORDS_ULIM = 64  # same as RC setting in pytorch-bert (--max_query_length)
ANSWER_NUM_CHARS_ULIM = 30  # same as RC setting in pytorch-bert (--max_answer_length)

# https://github.com/google-research/bert/issues/66 (384 words)
# pytorch-bert uses 384 tokens, with stride
CONTEXT_NUM_WORDS_ULIM = 400
