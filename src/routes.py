from flask import Flask
from questions import BlankQuestion
from generator import FalseGenerator
from summarizer import Summarizer

# Set Up the webserver
app = Flask(__name__)
app.config.from_pyfile('config.py')


@app.route("/")
def hello_world():
    summary = Summarizer("pdf/MDI.pdf")
    summary.extract_question_worthy_sentences()
    sentences = summary.get_sentences()
    blank_question_generator = BlankQuestion()

    questions = []
    #for sentence in sentences:
    blank_question_generator.generate_question(sentences[5])
    questions.append(blank_question_generator.get_question())
    fgen = FalseGenerator()
    fgen.train()

    return questions
