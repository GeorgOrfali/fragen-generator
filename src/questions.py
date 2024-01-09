import locale
import re
import wn
import random

locale.setlocale(locale.LC_ALL, 'de_DE')
disqualifier = ('Der', 'Die', 'Das', 'Ein', 'Eine', 'Ich', 'Du', 'Er', 'Sie', 'Es', 'Wir', 'Ihr',
                'Sich', 'Mein', 'Dein', 'Sein', 'Unser', 'Euer', 'Ihre', 'Dieser', 'Jener', 'Derjenige', 'Derselbe',
                'Man', 'Welcher', 'Dies')


def get_frequency_of_words(abstract):
    counts = dict()
    words = abstract.split()

    for i, word in enumerate(words):
        checked_word = re.sub('[^a-zA-Z0-9ÄÖÜäöüß_-]', '', word)
        if word[0].isupper() and i > 0 and checked_word not in disqualifier and len(checked_word) > 3:
            if checked_word in counts:
                counts[checked_word] += 1
            else:
                counts[checked_word] = 1

    return dict(counts)


def get_keywords(sentence):
    word_ranking = sorted(
        get_frequency_of_words(sentence).items(),
        key=lambda x: x[1],
        reverse=True
    )

    return list(word_ranking)


class BlankQuestion:
    question = {}
    tagger = None

    def __init__(self, tagger):
        self.tagger = tagger

    def generate_question(self, sentence):
            blank_word = self.getBlankField(sentence['sentence'])
            distractor = self.generate_distractors(blank_word)
            newQuestion = sentence['sentence'].replace(blank_word, '/blank/', 1).replace('\n', ' ')
            question = {
                'type': 'LückenText',
                'question': newQuestion,
                'sentence': sentence,
                'answer': blank_word,
                'distractors': distractor,
            }
            self.question = question

    def get_question(self):
        return self.question

    def generate_distractors(self, word):
        synset = wn.synsets(word)
        if len(synset) > 0:
            distractors = synset[0].get_related()
            if len(distractors) > 0:
                if len(distractors[0].lemmas()) > 0:
                    return distractors[0].lemmas()
        return []

    def getBlankField(self, sentence):
        result = ''
        for word in sentence.split():
            tag = self.tagger.analyze(word)
            if tag[1] == 'NN':
                result = word
                synset = wn.synsets(word)
                if len(synset) > 0:
                    distractors = synset[0].get_related()
                    if len(distractors) > 0:
                        return result

        # Wenn Die obere Methode kein Ergebnis liefert dann wird einfach highestFrequency genutzt
        if result is '':
            highest_frequency = get_keywords(sentence)
            if len(highest_frequency) > 0:
                result = highest_frequency[0][0]

        return result


class TrueFalseQuestion:
    question = {}
    tagger = None

    def __init__(self, tagger):
        self.tagger = tagger

    def generate_true_question(self, sentence):
        question = {
            'type': 'Wahr-Falsch',
            'question': sentence['sentence'].replace('\n', ''),
            'answer': 'Wahr'
        }
        self.question = question

    def generate_false_question(self, sentence):
        oldWord = sentence['sentence'].replace('\n', ' ').replace('.', '').split(' ')
        newSentence = sentence['sentence'].replace('\n', ' ').replace('▪', ',')
        #pre check for keine
        skip = False
        for w in oldWord:
            if 'kein' in w:
                newSentence = newSentence.replace(w, re.sub(r'^.', '', w), 1)
                skip = True
                break

        if skip is False:
            for i, word in enumerate(oldWord):
                if word is not '':
                    tag = self.tagger.analyze(word)
                    if len(tag) > 1:
                        if 'ADJ' in tag[1]:
                            if i == 0:
                                newWord = 'Nicht ' + word
                            else:
                                if '.' in oldWord[i]:
                                    newWord = 'Nicht ' + word
                                else:
                                    newWord = 'nicht ' + word
                            #newWord = oldWord[i] + 'nicht'
                            newSentence = newSentence.replace(word, newWord, 1)
                            break

        question = {
            'type': 'Wahr-Falsch',
            'question': newSentence,
            'sentence': sentence,
            'answer': 'Falsch'
        }
        self.question = question

    def generate_question(self, sentence):
        decide = random.randint(0, 9)
        if decide > 4:
            self.generate_true_question(sentence)
        else:
            self.generate_false_question(sentence)



    def get_question(self):
        return self.question
