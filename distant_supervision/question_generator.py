from .data_models import QuestionStyle
import re

class QuestionGeneratorError(Exception):
    pass


class QuestionGenerator:
    def __init__(self, whxx_ngram_table, text_preprocessor):
        self.whxx_ngram_table = whxx_ngram_table
        self.text_preprocessor = text_preprocessor

    def make_cloze_style(self, text, answer_str, mask):
        # replace ALL occurrences
        new_text = text.replace(answer_str, mask)

        if new_text == text:
            raise QuestionGeneratorError(
                'Failed to convert cloze style, did not replace anything with answer="{}": {}'.format(
                    answer_str, text))

        return new_text

    def _replace_with_question_mark_ending(self, text):
        return re.sub(r'\W*$', '?', text)

    def _post_questionify(self, text):
        text = text.strip()
        # capitalize() does not work here, because it lowercase the rest of the sentence
        text = text[0].upper() + text[1:]
        return self._replace_with_question_mark_ending(text)

    def _generate_template_awb(self, text, answer_str, sampled_ngram):
        """
        If cloze-style is “[FragmentA] [PERSON] [FragmentB]”, then:

        "[FragmentA], who [FragmentB]?" - AWB
        """

        # need to use \W+ to ensure word boundaries and not part of a word
        template = re.sub(
            r'\W+{}\W+'.format(re.escape(answer_str)),
            ', {} '.format(sampled_ngram),
            ' ' + text + ' ')

        # remove leading comma if the replacement was the first word
        template = re.sub(r'^,\s*', '', template)
        template = self._post_questionify(template)

        return template

    def _generate_template_wba(self, text, answer_str, sampled_ngram):
        """
        If cloze-style is “[FragmentA] [PERSON] [FragmentB]”, then:

        "Who [FragmentB] [FragmentA]?" - WBA
        """
        # need to use \W+ to ensure word boundaries and not part of a word
        template = re.sub(
            r'^(.*?)\W+{}\W+(.*?)\W*$'.format(re.escape(answer_str)),
            r'{} \2, \1'.format(sampled_ngram),
            ' ' + text + ' ')

        template = re.sub(r'\s+', ' ', template)  # regex above may have created double spaces
        template = self._post_questionify(template)

        # self._check_template(template, answer_str, text)

        return template

    def make_template_qg_styles(self, text, answer_str, ner_category, rng):
        if not self.text_preprocessor.findall_substr(answer_str, text):
            raise QuestionGeneratorError(
                'Failed to convert template QG style, answer="{}" not in question-text: {}'.format(
                    answer_str, text))

        sampled_ngram = self.whxx_ngram_table.rand_sample_ngram(rng.np, ner_category)

        styles = {
            QuestionStyle.TEMPLATE_AWB: self._generate_template_awb(text, answer_str, sampled_ngram),
            QuestionStyle.TEMPLATE_WBA: self._generate_template_wba(text, answer_str, sampled_ngram),
        }

        return styles
