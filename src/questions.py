import locale
import re
import wn
import nltk
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

    def generate_question(self, sentence):
        highest_frequency = get_keywords(sentence['sentence'])

        if len(highest_frequency) > 0:
            blank_word = highest_frequency[0][0]
            newQuestion = sentence['sentence'].replace(blank_word, '/blank/', 1).replace('\n', ' ')
            question = {
                'type': 'LückenText',
                'question': newQuestion,
                'answer': blank_word,
                # 'originalData': sentence
            }
            self.question = question

    def get_question(self):
        return self.question


class TrueFalseQuestion:
    question = {}

    def generate_true_question(self, sentence):
        question = {
            'type': 'Wahr-Falsch',
            'question': sentence['sentence'].replace('\n', ''),
            'answer': 'Wahr'
        }
        self.question = question

    def generate_false_question(self, sentence):
        wn.config.allow_multithreading = True
        oldWord = sentence['sentence'].replace('\n', '').replace('.', '').split(' ')
        newWord = ''
        newSentence = sentence['sentence'].replace('\n', '').replace('▪', ',')
        for i in range(len(oldWord)):
            w = wn.synsets(oldWord[i], pos='a')
            if len(w) > 0:
                if i == 0:
                    newWord = 'Nicht ' + oldWord[i]
                else:
                    newWord = 'nicht ' + oldWord[i]
                newSentence = newSentence.replace(oldWord[i], newWord)
                break

        question = {
            'type': 'Wahr-Falsch',
            'question': newSentence,
            'answer': 'Falsch',
            # 'originalData': sentence
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
