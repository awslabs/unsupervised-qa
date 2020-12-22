import json
import copy
from enum import Enum
from .text_preprocessor import TextPreprocessor


class PhraseMode(Enum):
    NER_ONLY = 'ner_only'
    ALL = 'all'


class QuestionStyle(Enum):
    """
    If cloze-style is “[FragmentA] [PERSON] [FragmentB]”, then:

    - "Who [FragmentB] [FragmentA]?" - WBA
    - "[FragmentA], who [FragmentB]?" - AWB
    """
    CLOZE_CATEGORY = 'cloze_category_style'
    CLOZE_GENERIC = 'cloze_generic_style'
    TEMPLATE_WBA = 'template_wba_style'
    TEMPLATE_AWB = 'template_awb_style'


class RcQuestion:
    """
    RC Question data, e.g. squad_rc_train.jsonl

    {
        "qid": "57277c965951b619008f8b2b",
        "question": "What do people engage in after they've disguised themselves?",
        "context": "In Greece Carnival is also ...",
        "answers": [
            {
                "ner_category": "SOME_CATEGORY",  # optional, depending on whether it's squad-ner or original
                "answer_start": 677,
                "text": "pranks and revelry"
            }
        ],
        "article_title": "Carnival"
    }
    """

    def __init__(self, *, qid=None, question=None, context=None, article_title=None, answers=None):
        self.qid = qid
        self.question = question
        self.context = context
        self.answers = answers  # list of {'text': "foobar", "answer_start": 48}

        self.article_title = article_title

    def jsonify(self):
        return json.dumps(self.__dict__)

    @classmethod
    def deserialize_json(cls, json_str):
        dct = json.loads(json_str)
        question = RcQuestion()
        for k, v in dct.items():
            question.__dict__[k] = v
        return question


class Article:
    def __init__(self):
        pass

    def import_from(self, raw_row):
        self.text = raw_row['text']
        self.id = int(raw_row['id'])
        self.title = raw_row['title']

        self.sents = None

    @classmethod
    def deserialize_json(cls, json_str):
        dct = json.loads(json_str)
        article = Article()
        for k, v in dct.items():
            article.__dict__[k] = v

        new_sents = []
        for sent in article.sents:
            new_sents.append(Sentence(sent['id'], sent['text'], sent['ents'], sent['noun_chunks']))
        article.sents = new_sents

        return article

    def __repr__(self):
        return str(self.__dict__)

    def jsonify(self):
        return json.dumps(self.__dict__)

class Sentence:
    def __init__(self, id, text, ents, noun_chunks):
        """
        noun chunks: https://nlp.stanford.edu/software/dependencies_manual.pdf
        """
        self.id = id  # note that this is sentence ID, and not article_id
        self.text = text
        self.ents = [(e[0], e[1]) for e in ents]
        self.noun_chunks = [(e[0], e[1]) for e in noun_chunks]

    def get_phrases(self, phrase_mode):
        if phrase_mode is PhraseMode.NER_ONLY:
            # don't pass it any noun_chunks
            return TextPreprocessor.get_phrases(entities=self.ents, noun_chunks=[])
        else:
            return TextPreprocessor.get_phrases(entities=self.ents, noun_chunks=self.noun_chunks)

    def __repr__(self):
        return str(self.__dict__)


class PhraseObj:
    def __init__(self, phrase_str, phrase_category):
        self.phrase_str = phrase_str
        self.phrase_category = phrase_category

    @classmethod
    def import_from(cls, row):
        """
        Example format: [["0 to 6 years", "DATE"], [5809465, 53318614, 49544471, 27237145, 54568155]]
        """
        phrase_pair = row[0]

        phrase = cls(phrase_pair[0], phrase_pair[1])
        return phrase

    def __repr__(self):
        return str(self.__dict__)


class DsDatum:
    def __init__(self, qid, styled_questions, context, answers):
        self.qid = qid
        self.styled_questions = styled_questions
        self.context = context
        self.answers = answers
        self.meta = None

    def __repr__(self):
        return str(self.__dict__)

    def jsonify(self):
        return json.dumps(self.__dict__)

    def jsonify_single_style(self, question_style):
        dct = copy.deepcopy(self.__dict__)
        dct['question'] = self.styled_questions[question_style.value]
        del dct['styled_questions']

        return json.dumps(dct)


class QueriesPerArticleObj:
    def __init__(self, *, article_id, article_title, article_raw, article_phrases, filtered_sents, phrase):
        self.article_id = int(article_id)
        self.article_title = article_title
        self.article_raw = article_raw
        self.article_phrases = article_phrases
        self.filtered_sents = filtered_sents
        self.phrase = phrase # answer phrase

    def __repr__(self):
        return str(self.__dict__)
