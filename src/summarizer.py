from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import json
import numpy as np

class Summarizer:
    train_data = []
    lexrank = None
    data_path = Path("datenset/evaluation/data.json")
    sentences = []
    tagger = None
    word_Count = 0

    def __init__(self, pdf, tagger):
        self.pdf = pdf
        self.load_data()
        self.tagger = tagger
        self.sentences = []

    def load_data(self):
        with self.data_path.open(mode='rt', encoding='utf-8') as f:
            data_temp = json.load(f)
        for i in range(len(data_temp['data'])):
            self.train_data.append(data_temp['data'][i]['sentence'])

        self.lexrank = LexRank(self.train_data, stopwords=STOPWORDS['de'])

    def extract_question_worthy_sentences(self):
        self.sentences = []
        allSentences = []
        for i, page in enumerate(extract_pages(self.pdf)):
            pageText = self.read_text_from_page(page)
            allSentences = allSentences + pageText['array']

        print('WORD COUNT IN PDF: ', self.word_Count)
        print('Sentence Amount extracted: ', len(allSentences))
        filteredSentences = list(set(self.textPreProcessing(allSentences)))
        #self.sentences = filteredSentences
        wordCountInAllSentences = []
        print('Start of WordCount: ', len(wordCountInAllSentences))
        for sentence in filteredSentences:
            wordCountInAllSentences.append(len(sentence.split(' ')))
        print('')
        print('Median of Word Count in Sentence: ', np.median(wordCountInAllSentences))
        print('Average of Word Count in Sentence: ', np.mean(wordCountInAllSentences))
        print('')
        print('Filtered Sentences after TextProcessing: ', len(filteredSentences))
        summary = self.lexrank.rank_sentences(
            filteredSentences,
            threshold=None,
            fast_power_method=True,
        )

        for index, rankSentence in enumerate(summary):
            if rankSentence > 1:
                self.sentences.append({
                    'sentence': filteredSentences[index],
                })
        print('Filtered Sentences after LexRank: ', len(self.sentences))

    def read_text_from_page(self, page):
        rawText = ''
        testArray = []
        Sentence = ''
        for element in page:
            if isinstance(element, LTTextContainer):
                textContent = element.get_text()
                self.word_Count = self.word_Count + len(textContent.split())
                if '•' in textContent or '▪' in textContent:
                    if Sentence == '':
                        Sentence = textContent
                    else:
                        if len(Sentence.split()) > 7:
                            # Sentence.replace('•', '').replace('▪', '')
                            testArray.append(Sentence)
                            Sentence = ''
                else:
                    if Sentence != '':
                        Sentence = Sentence + ' ' + textContent
                    else:
                        if len(textContent.split()) > 7:
                            testArray.append(textContent)
                            textContent = ''
        return {'array': testArray, 'raw': rawText}

    def get_sentences(self):
        return self.sentences

    def textPreProcessing(self, allSentences):
        result = []
        for sentence in allSentences:
            if '?' not in sentence:
                nomen = False
                verb = False
                adjektiv = False
                # sentence.replace('\n', ' ').replace('•', '').replace('▪', '')
                for word in sentence.split():
                    tag = self.tagger.analyze(word)
                    if 'NN' in tag[1]:
                        nomen = True
                    elif 'VV' in tag[1]:
                        verb = True
                    elif 'ADJ' in tag[1]:
                        adjektiv = True

                if nomen is True & verb is True & adjektiv is True:
                    result.append(sentence)
        return result
