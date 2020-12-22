
from operator import add

from .text_preprocessor import TextPreprocessor


class SquadNerCreatorError(Exception):
    pass


"""
{
    "qid": "57277c965951b619008f8b2b",
    "question": "What do people engage in after they've disguised themselves?",
    "context": "In Greece Carnival is also ...",
    "answers": [
        {
            "answer_start": 677,
            "ner_category": "SOME_CATEGORY",
            "text": "pranks and revelry"
        }
    ],
    "article_title": "Carnival"
}
"""

class SquadNerCreator:

    def __init__(self, output_dir, *, debug_save, num_partitions):
        self.output_dir = output_dir
        self.debug_save = debug_save
        self.num_partitions = num_partitions

        self.text_preprocessor = TextPreprocessor()

    def _process_row(self, question):
        """
        1. Run NER on `context` field
        2. Set `ner_category` for answers
        3. Discard answers that are not entities
        """

        # this returns a list, e.g. [('today', 'DATE'), ('Patrick', 'PERSON')]
        ent_lst = self.text_preprocessor.compute_ner(question.context)

        ent_dct = {}
        for ent_str, ner_category in ent_lst:
            # perform uncasing
            ent_dct[ent_str.lower()] = ner_category

        new_answers = []
        for ans_struct in question.answers:
            ans_text = ans_struct['text'].lower()  # perform uncasing
            if ans_text not in ent_dct:
                continue

            ans_struct['ner_category'] = ent_dct[ans_text]
            new_answers.append(ans_struct)

        question.answers = new_answers
        return question

    def _print_output_stats(self, question_rdd, metric_fptr):
        print('Count of new question_rdd: {}'.format(question_rdd.count()), file=metric_fptr)

        ner_category_counts = question_rdd.map(
            lambda x: (x.answers[0]['ner_category'], 1)
        ).reduceByKey(add).collectAsMap()

        total_count = sum(ner_category_counts.values())
        print(file=metric_fptr)
        for ner_category, count in sorted(ner_category_counts.items()):
            print('Number of samples with NER category "{}": {} / {} ({:.2f}%)'.format(
                ner_category,
                count,
                total_count,
                100.0 * count / total_count), file=metric_fptr)



    def run_job(self, sc, question_rdd, metric_fptr):
        print('Count of original question_rdd: {}'.format(question_rdd.count()), file=metric_fptr)

        question_rdd = question_rdd.map(lambda x: self._process_row(x)).filter(lambda x: len(x.answers) >= 1)

        final_output_dir = self.output_dir  # normally, we do os.path.join(self.output_dir, <...>)
        question_rdd.map(lambda x: x.jsonify()).saveAsTextFile(final_output_dir)

        self._print_output_stats(question_rdd, metric_fptr)
