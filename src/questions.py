import locale
import re

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


class BlankQuestion:
    question = {}

    def generate_question(self, sentence):
        word_ranking = sorted(
            get_frequency_of_words(sentence['sentence']).items(),
            key=lambda x: x[1],
            reverse=True
        )

        highest_frequency = list(word_ranking)

        if len(highest_frequency) > 0:
            blank_word = list(word_ranking)[0][0]
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

    def generate_question(self, sentence):
        question = {
            'type': 'Wahr-Falsch',
            'question': sentence['sentence'],
            'answer': 'True',
            # 'originalData': sentence
        }
        self.question = question

    def get_question(self):
        return self.question
