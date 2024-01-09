from flask import Flask, request, render_template, Response
from questions import BlankQuestion, TrueFalseQuestion
from summarizer import Summarizer
from flask_cors import CORS
import os
import tensorflow as tf
import wn
from generator import SingleChoice
from HanTa import HanoverTagger as ht
from jackxml import JACKXML

# Set Up the webserver
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route("/sentence")
def sentence():
    url = "pdf/experiment/Steuern.pdf"
    tagger = setup()

    return extract_sentences(url, tagger)

@app.route("/false")
def false():
    url = "pdf/experiment/DAS.pdf"
    tagger = setup()
    sentences = extract_sentences(url, tagger)
    tfq = TrueFalseQuestion(tagger)

    questions = []
    for sentence in sentences:
        tfq.generate_false_question(sentence)
        questions.append(tfq.get_question())

    return questions

@app.route("/fill")
def fill_in_the_blanc():
    url = "pdf/experiment/armut.pdf"
    tagger = setup()
    sentences = extract_sentences(url, tagger)
    blank_question_generator = BlankQuestion(tagger)

    questions = []
    for sentence in sentences:
        blank_question_generator.generate_question(sentence)
        questions.append(blank_question_generator.get_question())

    return questions

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/single")
def single_choice():
    url = "pdf/experiment/DAS.pdf"
    tagger = setup()
    QuestionRNN = tf.saved_model.load('generator')
    questions = []
    sentences = extract_sentences(url, tagger)
    for sentence in sentences:
        questions.append(generateSingleChoiceCase2(QuestionRNN, sentence['sentence']))

    print("Len Sentences: ", len(sentences))
    print("Len Questions: ", len(questions))
    sentences.clear()

    return questions

@app.route("/all")
def all():
    return generateAll("pdf/experiment/DAS.pdf")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file found', 400

    if 'aufgabenart'not in request.values:
        return 'No type of questions choosed!', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    file.save('uploads/' + file.filename)
    print("File Name of PDF: " + file.filename)
    return generate('uploads/' + file.filename, request.values['aufgabenart']), 200

@app.route('/xml', methods=['POST'])
def generate_xml():
    if 'aufgaben' not in request.values:
        return 'No type of questions choosed!', 400

    JACKXML().generate(request.values['aufgaben'])
    xml = open("generierte_uebung.xml", "r")
    response = Response(xml, mimetype='text/xml')
    response.headers['Content-type'] = 'application/force-download'
    return response

def generate(url, aufgabenart):
    tagger = setup()
    questions = []
    generator = None

    sentences = extract_sentences(url, tagger)

    if aufgabenart == 'w':
        generator = TrueFalseQuestion(tagger)
    elif aufgabenart == 'l':
        generator = BlankQuestion(tagger)
    elif aufgabenart == 's':
        generator = [tf.saved_model.load('generator'), BlankQuestion(tagger)]

    if aufgabenart != 'g':
        for s in sentences:
            if aufgabenart != 's':
                generator.generate_question(s)
                questions.append(generator.get_question())
            else:
                generator[1].generate_question(s)
                tempBlankQ = generator[1].get_question()
                questions.append(generateSingleChoice(generator[0], tempBlankQ))

    os.remove(url)
    return questions

def generateAll(url):
    tagger = setup()
    tfq = TrueFalseQuestion(tagger)
    blank_question_generator = BlankQuestion(tagger)
    print("URL: ", url)
    QuestionRNN = tf.saved_model.load('generator')
    print()
    questions = []
    sentences = extract_sentences(url, tagger)
    for sentence in sentences:
        tfq.generate_question(sentence)
        blank_question_generator.generate_question(sentence)

        questions.append(tfq.get_question())
        blankQuestion = blank_question_generator.get_question()
        questions.append(blankQuestion)
        questions.append(generateSingleChoice(QuestionRNN, blankQuestion))

    print("Len Sentences: ", len(sentences))
    print("Len Questions: ", len(questions))
    sentences.clear()

    return questions


def setup():
    try:
        wn.config.allow_multithreading = True
        wn.download('odenet')
    except:
        print("")

    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    return tagger


def extract_sentences(url, tagger):
    summary = Summarizer(url, tagger)
    summary.extract_question_worthy_sentences()
    return summary.get_sentences()


def generateSingleChoice(mlModel, blankQuestion):
    inputSentence = blankQuestion['question'].replace('/blank/', '<A>') + ' <A> ' + blankQuestion['answer']
    questionEncode = mlModel.generate(tf.constant([inputSentence]))
    question = SingleChoice().decode(questionEncode, [inputSentence])
    return {
        'answer': blankQuestion['answer'],
        'distractors': blankQuestion['distractors'],
        'question': question[0],
        'sentence': blankQuestion['sentence'],
        'type': 'Single Choice',
    }


def generateSingleChoiceCase2(mlModel, sentence):
    #inputSentence = blankQuestion['question'].replace('/blank/', '<A>') + ' <A> ' + blankQuestion['answer']
    disqualifier = ('Der', 'Die', 'Das', 'Ein', 'Eine', 'Ich', 'Du', 'Er', 'Sie', 'Es', 'Wir', 'Ihr',
                    'Sich', 'Mein', 'Dein', 'Sein', 'Unser', 'Euer', 'Ihre', 'Dieser', 'Jener', 'Derjenige', 'Derselbe',
                    'Man', 'Welcher', 'Dies')
    sentenceWords = sentence.split(' ')
    answer = ''
    if sentenceWords[0] in disqualifier:
        answer = sentenceWords[0] + ' ' + sentenceWords[1]
        sentenceWords[0] = ''
        sentenceWords[1] = '<A>'
        sentenceWords.append('<A>' + answer)
    else:
        answer = sentenceWords[0]
        sentenceWords[0] = '<A>'
        sentenceWords.append('<A>' + answer)

    inputSentence = ''.join(sentenceWords)
    questionEncode = mlModel.generate(tf.constant([inputSentence]))
    question = SingleChoice().decode(questionEncode, [inputSentence])
    return {
        'answer': answer,
        'question': question[0],
        'sentence': sentence,
        'type': 'Single Choice',
    }