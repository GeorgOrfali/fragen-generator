from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from nltk import tokenize
import nltk
import json


def read_text_from_page(page):
    rawText = ''
    count = 0
    for element in page:
        if isinstance(element, LTTextContainer) and count > 0:
            textContent = element.get_text()
            rawText = rawText + textContent
        count = count + 1
    resultArray = tokenize.sent_tokenize(rawText)
    return {'array': resultArray, 'raw': rawText}


class Summarizer:
    train_data = []
    lexrank = None
    data_path = Path("datenset/test.json")
    nltk.download('punkt')
    sentences = []

    def __init__(self, pdf):
        self.pdf = pdf
        self.load_data()

    def load_data(self):
        with self.data_path.open(mode='rt', encoding='utf-8') as f:
            data_temp = json.load(f)
        for i in range(len(data_temp['data'])):
            self.train_data.append(data_temp['data'][i]['sentence'])

        self.lexrank = LexRank(self.train_data, stopwords=STOPWORDS['de'])

    def extract_question_worthy_sentences(self):
        for i, page in enumerate(extract_pages(self.pdf)):
            pageText = read_text_from_page(page)
            summary = self.lexrank.rank_sentences(
                pageText['array'],
                threshold=None,
                fast_power_method=True,
            )
            if len(summary) > 0:
                index = summary.argmax(axis=0)
                self.sentences.append({
                    'sentence': pageText['array'][index],
                    'abstract': pageText['raw'],
                    'page': (i + 1)
                })

    def get_sentences(self):
        return self.sentences
