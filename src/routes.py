from flask import Flask, request
from questions import BlankQuestion, TrueFalseQuestion
from generator import SingleChoiceGenerator
from summarizer import Summarizer
from flask_cors import CORS
import tensorflow as tf

# Set Up the webserver
app = Flask(__name__)
app.config.from_pyfile('config.py')
CORS(app)


@app.route("/")
def hello_world():
    return generate("pdf/MDI.pdf")


@app.route("/all")
def all():
    return generateAll("pdf/MDI.pdf")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file found', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    file.save('uploads/' + file.filename)
    print(file.filename)
    return generate('uploads/' + file.filename), 200


def generate(url):
    summary = Summarizer(url)
    summary.extract_question_worthy_sentences()
    sentences = summary.get_sentences()
    blank_question_generator = BlankQuestion()

    questions = []
    for sentence in sentences:
        tfq = TrueFalseQuestion()
        tfq.generate_question(sentence)
        blank_question_generator.generate_question(sentence)
        questions.append(tfq.get_question())
        blankQuestion = blank_question_generator.get_question()
        questions.append(blankQuestion)
    return questions


def generateAll(url):
    summary = Summarizer(url)
    summary.extract_question_worthy_sentences()
    sentences = summary.get_sentences()
    blank_question_generator = BlankQuestion()
    QuestionRNN = tf.saved_model.load('generator')

    questions = []
    for sentence in sentences:
        tfq = TrueFalseQuestion()
        tfq.generate_question(sentence)
        blank_question_generator.generate_question(sentence)
        questions.append(tfq.get_question())
        blankQuestion = blank_question_generator.get_question()
        questions.append(blankQuestion)
        inputSentence = blankQuestion['question'].replace('/blank/', '<A>') + ' <A> ' + blankQuestion['answer']
        question = QuestionRNN.generate(tf.constant([inputSentence]))
        questions.append({
            'answer': blankQuestion['answer'],
            'question': question[0].numpy().decode(),
            'type': 'Single Choice',
        })
    print(question)
    return questions
